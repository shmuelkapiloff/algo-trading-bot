"""
Synthetic Canary Probe — connectivity and data freshness check.

Runs every 5 minutes during market hours (09:30–16:00 ET, Mon–Fri) and
verifies the trading infrastructure is alive before placing any orders.

Checks performed
----------------
1. API latency       — Alpaca responds within 2 seconds
2. Data freshness    — SPY latest bar is < 90 seconds stale
3. Price sanity      — SPY price within ±20% of last known price

The probe is READ-ONLY.  It never affects trading state directly, but a
FAIL status causes an CRITICAL alert so the operator can investigate.

Results are stored in Redis:
  canary:last_result     — JSON dict from most recent probe
  canary:fail_count      — consecutive failure count (reset on success)

Usage (wired by main.py into APScheduler):
    probe = CanaryProbe(fetcher, redis_client, alert_dispatcher)
    await probe.run()  # returns ProbeResult dict
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import timezone
from typing import Optional

import redis.asyncio as aioredis

from ..data.fetcher import MarketDataFetcher
from .alerts import AlertDispatcher, AlertLevel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANARY_SYMBOL = "SPY"
FRESHNESS_THRESHOLD_SEC: float = 90.0     # bar must be younger than this
PRICE_DEVIATION_THRESHOLD: float = 0.20   # ±20% from last known price
API_TIMEOUT_SEC: float = 2.0              # Alpaca must respond within 2 s
CONSECUTIVE_FAIL_ALERT_THRESHOLD: int = 2  # alert after N consecutive fails

REDIS_KEY_LAST_RESULT = "canary:last_result"
REDIS_KEY_FAIL_COUNT = "canary:fail_count"
REDIS_KEY_LAST_PRICE = "spy:last_price"


# ---------------------------------------------------------------------------
# Canary Probe
# ---------------------------------------------------------------------------

class CanaryProbe:
    """
    Runs the synthetic canary probe against Alpaca Market Data.

    Parameters
    ----------
    fetcher           : MarketDataFetcher wrapping the Alpaca SDK
    redis_client      : redis.asyncio connection
    alert_dispatcher  : AlertDispatcher for Telegram/log alerts (optional)
    """

    def __init__(
        self,
        fetcher: MarketDataFetcher,
        redis_client: aioredis.Redis,
        alert_dispatcher: Optional[AlertDispatcher] = None,
    ) -> None:
        self._fetcher = fetcher
        self._redis = redis_client
        self._alerts = alert_dispatcher

    async def run(self) -> dict:
        """
        Execute one canary probe cycle.

        Returns a result dict:
          {
            'status':              'ok' | 'warn' | 'fail',
            'latency_ms':          float,
            'staleness_sec':       float | None,
            'price_deviation_pct': float | None,
            'reason':              str | None,
          }
        """
        probe_start = time.monotonic()
        result: dict = {
            "status": "fail",
            "latency_ms": None,
            "staleness_sec": None,
            "price_deviation_pct": None,
            "reason": None,
        }

        try:
            bar = await asyncio.wait_for(
                self._fetcher.get_latest_bar(CANARY_SYMBOL),
                timeout=API_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            result["reason"] = "api_timeout"
            await self._on_result(result)
            return result
        except Exception as exc:
            result["reason"] = f"api_error:{type(exc).__name__}"
            await self._on_result(result)
            return result

        latency_ms = (time.monotonic() - probe_start) * 1_000
        result["latency_ms"] = latency_ms

        if bar is None:
            result["reason"] = "no_bar_returned"
            await self._on_result(result)
            return result

        # ── Freshness check ───────────────────────────────────────────
        staleness_sec = float(bar.get("lag_seconds", FRESHNESS_THRESHOLD_SEC))
        result["staleness_sec"] = staleness_sec
        is_fresh = staleness_sec < FRESHNESS_THRESHOLD_SEC

        # ── Price sanity check ────────────────────────────────────────
        close_price = float(bar.get("close_adj", 0.0))
        raw_last = await self._redis.get(REDIS_KEY_LAST_PRICE)
        last_known = float(raw_last) if raw_last else close_price

        if last_known > 0:
            price_deviation = abs(close_price - last_known) / last_known
        else:
            price_deviation = 0.0
        result["price_deviation_pct"] = price_deviation
        is_sane = price_deviation < PRICE_DEVIATION_THRESHOLD

        # ── Store current price for next cycle ────────────────────────
        if close_price > 0:
            await self._redis.set(REDIS_KEY_LAST_PRICE, close_price)

        # ── Determine status ──────────────────────────────────────────
        if not is_fresh and not is_sane:
            result["status"] = "fail"
            result["reason"] = (
                f"stale:{staleness_sec:.0f}s+price_deviation:{price_deviation:.1%}"
            )
        elif not is_fresh:
            result["status"] = "warn"
            result["reason"] = f"stale:{staleness_sec:.0f}s"
        elif not is_sane:
            result["status"] = "warn"
            result["reason"] = f"price_deviation:{price_deviation:.1%}"
        else:
            result["status"] = "ok"

        await self._on_result(result)
        return result

    # ------------------------------------------------------------------
    # Post-run bookkeeping and alerting
    # ------------------------------------------------------------------

    async def _on_result(self, result: dict) -> None:
        """Persist result, update fail counter, send alerts as needed."""
        status = result["status"]

        # Persist latest result
        await self._redis.set(REDIS_KEY_LAST_RESULT, json.dumps(result))

        if status == "ok":
            # Reset consecutive fail counter on success
            await self._redis.set(REDIS_KEY_FAIL_COUNT, 0)
            logger.debug(
                "Canary probe OK — latency=%.0f ms staleness=%.0f s",
                result.get("latency_ms") or 0,
                result.get("staleness_sec") or 0,
            )
        else:
            # Increment fail counter
            fail_count = await self._redis.incr(REDIS_KEY_FAIL_COUNT)

            log_fn = logger.warning if status == "warn" else logger.error
            log_fn(
                "Canary probe %s — reason=%s latency=%.0f ms (consecutive fails: %d)",
                status.upper(),
                result.get("reason"),
                result.get("latency_ms") or 0,
                fail_count,
            )

            # Alert on first fail or after threshold consecutive failures
            if self._alerts and (
                status == "fail" or fail_count >= CONSECUTIVE_FAIL_ALERT_THRESHOLD
            ):
                alert_level = (
                    AlertLevel.CRITICAL if status == "fail" else AlertLevel.WARNING
                )
                await self._alerts.send(
                    f"Canary probe {status.upper()}: {result.get('reason')} "
                    f"(consecutive: {fail_count})",
                    level=alert_level,
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def get_last_result(self) -> Optional[dict]:
        """Return the most recent probe result from Redis (or None)."""
        raw = await self._redis.get(REDIS_KEY_LAST_RESULT)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None

    async def get_fail_count(self) -> int:
        """Return current consecutive failure count."""
        raw = await self._redis.get(REDIS_KEY_FAIL_COUNT)
        return int(raw or 0)
