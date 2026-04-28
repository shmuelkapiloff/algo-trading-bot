"""
Gate 1 — Signal Viability

Rejects signals whose expected value does not cover transaction costs
by the required multiple (min_cost_coverage_ratio, default 1.5×).

Cost model (round-trip):
    total_cost = (spread_bps + slippage_bps) / 10_000 × 2

Decision rule:
    signal.confidence × expected_win_per_signal > total_cost × min_ratio

Dependencies (constructor-injected, both are Protocols for easy mocking):
    cost_estimator      Provides live spread_bps and rolling avg slippage.
    performance_metrics Provides rolling expected win per signal by strategy.

Fallback:
    When trade history < min_sample_size (default 30), falls back to
    fallback_expected_win from config rather than dividing by zero.
"""

from __future__ import annotations

import logging
from typing import Protocol

from .base_gate import GateResult, RiskGate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Injected collaborator protocols
# ---------------------------------------------------------------------------


class CostEstimator(Protocol):
    def get_spread_bps(self, symbol: str) -> float:
        """Current bid/ask spread in basis points."""
        ...

    def get_avg_slippage_bps(self) -> float:
        """Rolling average slippage across all recent fills (bps)."""
        ...


class PerformanceMetrics(Protocol):
    def get_expected_win_per_signal(self, strategy_name: str) -> float:
        """
        Rolling average profit per trade for this strategy (as a fraction, e.g. 0.015).
        Returns fallback_expected_win when sample size < min_sample_size.
        """
        ...


# ---------------------------------------------------------------------------
# Gate implementation
# ---------------------------------------------------------------------------


class SignalViabilityGate(RiskGate):
    """
    Blocks signals whose EV is negative after transaction costs.

    From TRADING_BOT_PLAN.md §6דא:
        return expected_return > total_cost × min_cost_coverage_ratio
    """

    def __init__(
        self,
        cost_estimator: CostEstimator,
        performance_metrics: PerformanceMetrics,
        min_cost_coverage_ratio: float = 1.5,
    ) -> None:
        self._costs = cost_estimator
        self._perf = performance_metrics
        self._min_ratio = min_cost_coverage_ratio

    async def evaluate(self, signal, portfolio_state: dict) -> GateResult:
        # Expected return for this signal
        expected_win = self._perf.get_expected_win_per_signal(signal.strategy_name)
        expected_return = signal.confidence * expected_win

        # Round-trip transaction cost (entry + exit)
        spread_cost = self._costs.get_spread_bps(signal.symbol) / 10_000 * 2
        slippage_cost = self._costs.get_avg_slippage_bps() / 10_000 * 2
        total_cost = spread_cost + slippage_cost

        threshold = total_cost * self._min_ratio

        if expected_return <= threshold:
            logger.debug(
                "[SignalViabilityGate] REJECTED %s  "
                "EV=%.5f  threshold=%.5f  (spread=%.1f bps  slippage=%.1f bps)",
                signal.symbol,
                expected_return,
                threshold,
                self._costs.get_spread_bps(signal.symbol),
                self._costs.get_avg_slippage_bps(),
            )
            return GateResult.reject("signal_ev_negative_after_costs")

        return GateResult.approve()
