"""
Transaction Cost Analysis (TCA) — real-time fill quality monitoring.

Records slippage, fill rate, and latency for every fill event and exposes a
rolling-window view of execution quality.  When thresholds are breached the
monitor auto-pauses new orders (via RuntimeStateStore) or auto-throttles
position sizes.

Metrics (rolling window = last TCA_WINDOW fills):
  avg_slippage_bps      — deviation from VWAP benchmark in basis points
  fill_rate_pct         — filled_qty / requested_qty
  pct_filled_within_30s — fraction of fills arriving within 30 s

Auto-pause triggers (→ RuntimeStateStore.force_transition_internal(PAUSED)):
  avg_slippage_bps  > PAUSE_SLIPPAGE_BPS  (default 25 bps)
  fill_rate_pct     < PAUSE_FILL_RATE     (default 0.60)
  broker_latency_ms > PAUSE_LATENCY_MS    (default 2 000 ms)

Auto-throttle (reduces order sizes by 50 %, up to MAX_THROTTLE_STEPS):
  avg_slippage_bps  > THROTTLE_SLIPPAGE_BPS (default 12 bps)

Redis keys used:
  tca:fills          — JSON list of last TCA_WINDOW fill records
  tca:throttle_steps — int (current throttle step, 0–MAX_THROTTLE_STEPS)

Usage (called from event handler on every fill):
    tca = TcaMonitor(redis_client, state_store, alert_dispatcher)
    await tca.record_fill(
        order_id="xxx", symbol="AAPL",
        fill_price=182.50, vwap_benchmark=182.30,
        filled_qty=10, requested_qty=10,
        fill_latency_ms=120.0, side="buy",
    )
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Optional

import redis.asyncio as aioredis
from ..runtime_state import RuntimeStateStore, TradingState
from .alerts import AlertDispatcher, AlertLevel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

TCA_WINDOW: int = 100  # rolling window size (fills)
PAUSE_SLIPPAGE_BPS: float = 25.0  # auto-pause if avg slippage > this
PAUSE_FILL_RATE: float = 0.60  # auto-pause if fill rate < this
PAUSE_LATENCY_MS: float = 2_000.0  # auto-pause if broker latency p95 > this
THROTTLE_SLIPPAGE_BPS: float = 12.0  # auto-throttle if avg slippage > this
MAX_THROTTLE_STEPS: int = 3  # max position size reduction steps
THROTTLE_FACTOR: float = 0.50  # multiplier per throttle step

REDIS_KEY_FILLS = "tca:fills"
REDIS_KEY_THROTTLE = "tca:throttle_steps"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FillRecord:
    order_id: str
    symbol: str
    slippage_bps: float  # positive = paid more than benchmark
    fill_rate: float  # filled_qty / requested_qty  (0.0–1.0)
    fill_latency_ms: float
    side: str  # "buy" | "sell"
    timestamp: float  # Unix epoch (seconds)
    within_30s: bool  # fill_latency_ms <= 30_000


@dataclass
class TcaMetrics:
    sample_size: int
    avg_slippage_bps: float
    fill_rate_pct: float
    pct_filled_within_30s: float
    throttle_steps: int

    @property
    def throttle_multiplier(self) -> float:
        """Returns the position-size multiplier (1.0 = no throttle)."""
        return THROTTLE_FACTOR**self.throttle_steps


# ---------------------------------------------------------------------------
# TCA Monitor
# ---------------------------------------------------------------------------


class TcaMonitor:
    """
    Records fill events, computes rolling metrics, and enforces circuit-
    breaker thresholds.

    Thread-safety: async only; not safe to call from multiple tasks
    simultaneously on the same key.
    """

    def __init__(
        self,
        redis_client: aioredis.Redis,
        state_store: RuntimeStateStore,
        alert_dispatcher: Optional["AlertDispatcher"] = None,
    ) -> None:
        self._redis = redis_client
        self._state_store = state_store
        self._alerts = alert_dispatcher

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    async def record_fill(
        self,
        order_id: str,
        symbol: str,
        fill_price: float,
        vwap_benchmark: float,
        filled_qty: int,
        requested_qty: int,
        fill_latency_ms: float,
        side: str = "buy",
    ) -> None:
        """
        Record one fill and evaluate circuit-breaker thresholds.

        slippage_bps is sign-adjusted:
          BUY  → positive means we paid MORE than VWAP (bad)
          SELL → positive means we received LESS than VWAP (bad)
        """
        if vwap_benchmark <= 0 or requested_qty <= 0:
            logger.debug(
                "TcaMonitor.record_fill: invalid benchmark=%.4f or requested_qty=%d — skipping",
                vwap_benchmark,
                requested_qty,
            )
            return

        side_sign = 1.0 if side == "buy" else -1.0
        slippage_bps = (
            (fill_price - vwap_benchmark) / vwap_benchmark * 10_000 * side_sign
        )
        fill_rate = filled_qty / requested_qty

        record = FillRecord(
            order_id=order_id,
            symbol=symbol,
            slippage_bps=slippage_bps,
            fill_rate=fill_rate,
            fill_latency_ms=fill_latency_ms,
            side=side,
            timestamp=time.time(),
            within_30s=(fill_latency_ms <= 30_000),
        )

        # Push to Redis list and trim to window size
        await self._redis.rpush(REDIS_KEY_FILLS, json.dumps(asdict(record)))
        await self._redis.ltrim(REDIS_KEY_FILLS, -TCA_WINDOW, -1)

        logger.debug(
            "TCA fill recorded: %s %s slippage=%.1f bps fill_rate=%.2f latency=%.0f ms",
            side.upper(),
            symbol,
            slippage_bps,
            fill_rate,
            fill_latency_ms,
        )

        # Re-evaluate thresholds after every fill
        await self._evaluate_thresholds()

    async def record_broker_latency(self, latency_ms: float) -> None:
        """
        Optional: record a broker API latency sample for the circuit-breaker
        latency check.  Call from AlpacaBroker health_check or similar.
        """
        await self._evaluate_thresholds(latest_latency_ms=latency_ms)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def get_metrics(self) -> TcaMetrics:
        """Return rolling-window metrics from Redis."""
        raw_list = await self._redis.lrange(REDIS_KEY_FILLS, 0, -1)
        throttle_raw = await self._redis.get(REDIS_KEY_THROTTLE)
        throttle_steps = int(throttle_raw or 0)

        if not raw_list:
            return TcaMetrics(
                sample_size=0,
                avg_slippage_bps=0.0,
                fill_rate_pct=1.0,
                pct_filled_within_30s=1.0,
                throttle_steps=throttle_steps,
            )

        records: list[FillRecord] = []
        for raw in raw_list:
            try:
                data = json.loads(raw)
                records.append(FillRecord(**data))
            except Exception:
                continue

        n = len(records)
        avg_slippage = sum(r.slippage_bps for r in records) / n
        fill_rate = sum(r.fill_rate for r in records) / n
        pct_within_30s = sum(1 for r in records if r.within_30s) / n

        return TcaMetrics(
            sample_size=n,
            avg_slippage_bps=avg_slippage,
            fill_rate_pct=fill_rate,
            pct_filled_within_30s=pct_within_30s,
            throttle_steps=throttle_steps,
        )

    async def get_throttle_multiplier(self) -> float:
        """
        Returns the current position-size multiplier (1.0 = no throttle).
        Used by PortfolioManager / ExecutionRouter to scale order sizes.
        """
        metrics = await self.get_metrics()
        return metrics.throttle_multiplier

    # ------------------------------------------------------------------
    # Circuit-breaker / throttle evaluation
    # ------------------------------------------------------------------

    async def _evaluate_thresholds(
        self, latest_latency_ms: Optional[float] = None
    ) -> None:
        """
        Check current metrics against circuit-breaker thresholds.
        Auto-pauses or auto-throttles as required.
        """
        metrics = await self.get_metrics()

        # ── Throttle check ────────────────────────────────────────────
        if (
            metrics.sample_size >= 5  # need at least 5 fills to act
            and metrics.avg_slippage_bps > THROTTLE_SLIPPAGE_BPS
            and metrics.throttle_steps < MAX_THROTTLE_STEPS
        ):
            new_steps = min(metrics.throttle_steps + 1, MAX_THROTTLE_STEPS)
            await self._redis.set(REDIS_KEY_THROTTLE, new_steps)
            logger.warning(
                "TCA throttle step %d/%d activated: avg_slippage=%.1f bps > %.1f bps threshold. "
                "Position sizes reduced to %.0f%%.",
                new_steps,
                MAX_THROTTLE_STEPS,
                metrics.avg_slippage_bps,
                THROTTLE_SLIPPAGE_BPS,
                (THROTTLE_FACTOR**new_steps) * 100,
            )
            if self._alerts:
                await self._alerts.send(
                    f"TCA throttle step {new_steps}: avg slippage {metrics.avg_slippage_bps:.1f} bps "
                    f"→ sizes reduced to {(THROTTLE_FACTOR**new_steps)*100:.0f}%",
                    level=AlertLevel.WARNING,
                )

        # ── Auto-pause check (circuit breaker) ───────────────────────
        if metrics.sample_size < 5:
            return  # not enough data to pause

        pause_reason: Optional[str] = None

        if metrics.avg_slippage_bps > PAUSE_SLIPPAGE_BPS:
            pause_reason = (
                f"tca_slippage_breach:{metrics.avg_slippage_bps:.1f}_bps>"
                f"{PAUSE_SLIPPAGE_BPS}_bps"
            )
        elif metrics.fill_rate_pct < PAUSE_FILL_RATE:
            pause_reason = (
                f"tca_fill_rate_breach:{metrics.fill_rate_pct:.2f}<{PAUSE_FILL_RATE}"
            )
        elif latest_latency_ms is not None and latest_latency_ms > PAUSE_LATENCY_MS:
            pause_reason = (
                f"tca_latency_breach:{latest_latency_ms:.0f}ms>{PAUSE_LATENCY_MS:.0f}ms"
            )

        if pause_reason:
            logger.critical(
                "TCA CIRCUIT BREAKER ACTIVATED: %s — pausing new orders.",
                pause_reason,
            )
            success, _ = await self._state_store.force_transition_internal(
                target=TradingState.PAUSED,
                reason=pause_reason,
            )
            if self._alerts:
                await self._alerts.send(
                    f"TCA CIRCUIT BREAKER: {pause_reason}. New orders paused.",
                    level=AlertLevel.CRITICAL,
                )
            if not success:
                logger.error(
                    "TCA: force_transition_internal failed — state may already be HALTED."
                )

    # ------------------------------------------------------------------
    # Reset (manual, after operator review)
    # ------------------------------------------------------------------

    async def reset_throttle(self) -> None:
        """Reset throttle steps to 0. Call after operator review and manual resume."""
        await self._redis.set(REDIS_KEY_THROTTLE, 0)
        logger.info("TCA throttle reset to 0.")

    async def clear_history(self) -> None:
        """Erase fill history (dev/test use only)."""
        await self._redis.delete(REDIS_KEY_FILLS)
        await self._redis.set(REDIS_KEY_THROTTLE, 0)
