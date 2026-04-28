"""
Gate 5 — Execution Readiness (TCA metrics + broker health)

Blocks orders when execution conditions are degraded:

  1. Broker API latency p95 > broker_latency_p95_max_ms (default 1500 ms).
  2. Rolling average slippage > slippage_pause_bps (default 25 bps).
  3. Fill rate p95 < min_fill_rate_p95 (default 0.60 = 60%).

These thresholds match the tca_monitoring.auto_pause_new_orders_if block
in TRADING_BOT_PLAN.md §6гб.

From TRADING_BOT_PLAN.md:
    broker_latency_p95_ms_max: 1500
    avg_slippage_bps pause threshold: 25
    fill_rate_pct threshold: 0.60

Dependencies (constructor-injected):
    tca_metrics    Provides rolling execution metrics from the TCA module.
"""

from __future__ import annotations

import logging
from typing import Protocol

from .base_gate import GateResult, RiskGate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Injected collaborator protocol
# ---------------------------------------------------------------------------


class TCAMetrics(Protocol):
    def get_broker_latency_p95_ms(self) -> float:
        """Rolling p95 broker round-trip latency in milliseconds."""
        ...

    def get_avg_slippage_bps(self) -> float:
        """Rolling average slippage vs VWAP benchmark in basis points."""
        ...

    def get_fill_rate_p95(self) -> float:
        """
        Rolling p95 fill rate (fraction of qty filled per order, 0.0–1.0).
        """
        ...


# ---------------------------------------------------------------------------
# Gate implementation
# ---------------------------------------------------------------------------


class ExecutionReadinessGate(RiskGate):
    """
    Blocks orders when broker/execution conditions fall below SLO thresholds.
    """

    def __init__(
        self,
        tca_metrics: TCAMetrics,
        broker_latency_p95_max_ms: float = 1500.0,
        slippage_pause_bps: float = 25.0,
        min_fill_rate_p95: float = 0.60,
    ) -> None:
        self._tca = tca_metrics
        self._max_latency_ms = broker_latency_p95_max_ms
        self._slippage_pause = slippage_pause_bps
        self._min_fill_rate = min_fill_rate_p95

    async def evaluate(self, signal, portfolio_state: dict) -> GateResult:

        # ---- 1. Broker latency ----
        latency_p95 = self._tca.get_broker_latency_p95_ms()
        if latency_p95 > self._max_latency_ms:
            logger.warning(
                "[ExecutionReadinessGate] REJECTED %s  "
                "broker_latency_p95=%.0f ms  limit=%.0f ms",
                signal.symbol,
                latency_p95,
                self._max_latency_ms,
            )
            return GateResult.reject(
                f"broker_latency_breach:{latency_p95:.0f}ms>{self._max_latency_ms:.0f}ms"
            )

        # ---- 2. Rolling slippage ----
        avg_slippage = self._tca.get_avg_slippage_bps()
        if avg_slippage > self._slippage_pause:
            logger.warning(
                "[ExecutionReadinessGate] REJECTED %s  "
                "avg_slippage=%.1f bps  limit=%.1f bps",
                signal.symbol,
                avg_slippage,
                self._slippage_pause,
            )
            return GateResult.reject(
                f"slippage_breach:{avg_slippage:.1f}bps>{self._slippage_pause:.1f}bps"
            )

        # ---- 3. Fill rate ----
        fill_rate = self._tca.get_fill_rate_p95()
        if fill_rate < self._min_fill_rate:
            logger.warning(
                "[ExecutionReadinessGate] REJECTED %s  "
                "fill_rate_p95=%.2f  minimum=%.2f",
                signal.symbol,
                fill_rate,
                self._min_fill_rate,
            )
            return GateResult.reject(
                f"fill_rate_breach:{fill_rate:.2f}<{self._min_fill_rate:.2f}"
            )

        return GateResult.approve()
