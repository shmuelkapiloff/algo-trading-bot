"""
Gate 4 — Tail-Risk Admission (ES / VaR + correlation crisis)

Two hard checks:
  1. Projected portfolio ES (Expected Shortfall at 95%) after adding the
     new position must remain below es_max_pct_portfolio (default 5%).
  2. Projected portfolio VaR (99%) must remain below var_max_pct_portfolio
     (default 3%).

One soft modifier:
  3. If average pairwise portfolio correlation exceeds
     correlation_crisis_threshold (default 0.75), the gate reduces
     signal.qty by crisis_size_reduction (default 50%) rather than
     blocking outright. The modified signal is forwarded to Gate 5.

From TRADING_BOT_PLAN.md §4 tail_risk_gate block:
    es_percentile: 0.95
    var_percentile: 0.99
    es_max_pct_portfolio: 0.05
    var_max_pct_portfolio: 0.03
    correlation_crisis_threshold: 0.75
    crisis_correlation_size_reduction: 0.50

ES/VaR estimation
-----------------
The RiskEngine.estimate_*_after_trade() methods receive the SignalIntent
directly and use it to compute a hypothetical marginal contribution to
portfolio risk. Internally they may use parametric VaR or historical
simulation — the gate does not care which method is used.

When signal.qty is None (signal arrives before Portfolio layer sizing),
the risk engine should estimate using its own worst-case sizing assumption
(e.g. absolute_max_position_pct). This keeps the gate conservative.

Dependencies (constructor-injected):
    risk_engine    Provides projected ES/VaR and correlation metrics.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Protocol

from .base_gate import GateResult, RiskGate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Injected collaborator protocol
# ---------------------------------------------------------------------------


class RiskEngine(Protocol):
    def estimate_es_after_trade(self, signal, portfolio_state: dict) -> float:
        """
        Projected portfolio ES (at 95%) as a fraction of NAV if the signal
        were to be executed at worst-case sizing.
        e.g. 0.04 = 4% of portfolio.
        """
        ...

    def estimate_var_after_trade(self, signal, portfolio_state: dict) -> float:
        """
        Projected portfolio VaR (at 99%) as a fraction of NAV.
        """
        ...

    def get_average_portfolio_correlation(self) -> float:
        """
        Current average pairwise Pearson correlation across all holdings
        over the rolling correlation_lookback_days window.
        Returns 0.0 if portfolio is empty.
        """
        ...


# ---------------------------------------------------------------------------
# Gate implementation
# ---------------------------------------------------------------------------


class TailRiskGate(RiskGate):
    """
    Blocks trades that push ES/VaR past limits; halves qty in crisis mode.
    """

    def __init__(
        self,
        risk_engine: RiskEngine,
        es_max_pct_portfolio: float = 0.05,
        var_max_pct_portfolio: float = 0.03,
        correlation_crisis_threshold: float = 0.75,
        crisis_size_reduction: float = 0.50,
    ) -> None:
        self._risk_engine = risk_engine
        self._es_max = es_max_pct_portfolio
        self._var_max = var_max_pct_portfolio
        self._crisis_threshold = correlation_crisis_threshold
        self._crisis_reduction = crisis_size_reduction

    async def evaluate(self, signal, portfolio_state: dict) -> GateResult:

        # ---- 1. Expected Shortfall check ----
        projected_es = self._risk_engine.estimate_es_after_trade(
            signal, portfolio_state
        )
        if projected_es > self._es_max:
            logger.warning(
                "[TailRiskGate] REJECTED %s  projected_ES=%.4f  limit=%.4f",
                signal.symbol,
                projected_es,
                self._es_max,
            )
            return GateResult.reject(f"es_breach:{projected_es:.4f}>{self._es_max:.4f}")

        # ---- 2. Value at Risk check ----
        projected_var = self._risk_engine.estimate_var_after_trade(
            signal, portfolio_state
        )
        if projected_var > self._var_max:
            logger.warning(
                "[TailRiskGate] REJECTED %s  projected_VaR=%.4f  limit=%.4f",
                signal.symbol,
                projected_var,
                self._var_max,
            )
            return GateResult.reject(
                f"var_breach:{projected_var:.4f}>{self._var_max:.4f}"
            )

        # ---- 3. Correlation crisis: reduce size, do not block ----
        avg_corr = self._risk_engine.get_average_portfolio_correlation()
        if avg_corr > self._crisis_threshold:
            if signal.qty is not None:
                reduced_qty = signal.qty * self._crisis_reduction
                modified = dataclasses.replace(signal, qty=reduced_qty)
                logger.info(
                    "[TailRiskGate] Crisis correlation %.2f > %.2f — "
                    "reducing %s qty %.2f → %.2f (%.0f%%)",
                    avg_corr,
                    self._crisis_threshold,
                    signal.symbol,
                    signal.qty,
                    reduced_qty,
                    self._crisis_reduction * 100,
                )
                return GateResult.approve(
                    reason="warning:crisis_correlation_active",
                    modified_signal=modified,
                )
            else:
                # qty not yet set — log the warning but cannot modify
                logger.info(
                    "[TailRiskGate] Crisis correlation %.2f > %.2f for %s "
                    "(qty not yet set — Portfolio layer must apply crisis reduction)",
                    avg_corr,
                    self._crisis_threshold,
                    signal.symbol,
                )
                return GateResult.approve(
                    reason="warning:crisis_correlation_qty_unknown"
                )

        return GateResult.approve()
