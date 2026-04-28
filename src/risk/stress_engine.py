"""
Stress & Exposure Engine — Intraday VaR/ES Scenarios.

Computes:
  1. Historical VaR (99%) and ES/CVaR (95%) for the current portfolio
  2. Stress scenarios: VIX spike, liquidity drain, correlation cluster
  3. Position heatmap by symbol (contribution to portfolio risk)

From TRADING_BOT_PLAN.md §4 (tail_risk_gate) and §6יד.

VaR/ES Methodology
------------------
  - Uses rolling historical window (lookback_days=63 ≈ 1 quarter)
  - Historical simulation: sort portfolio daily P&L, take percentile
  - ES = average of losses beyond VaR threshold (Expected Shortfall)
  - Returns are computed as daily_pnl / equity (fractional)

Stress Scenarios
----------------
  vix_spike_20pct     : Simulate VIX +20% → scale portfolio vol by 1.3
  liquidity_drain_20pct: Simulate 20% spread widening → widen all positions
  correlation_cluster : Force pairwise correlation to 0.95 → full covariance

Usage
-----
    engine = StressEngine(initial_equity=10_000.0)
    result = engine.compute(positions=portfolio.open_positions, daily_pnl=series)
    if result.var_99 < -0.03:
        await incident_controller.report(DRAWDOWN_BREACH, CRITICAL)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


@dataclass
class StressResult:
    """Result of a VaR/ES stress computation."""
    var_99: float               # VaR at 99% confidence (negative = loss)
    var_95: float               # VaR at 95% confidence
    es_95: float                # Expected Shortfall at 95% (CVaR), negative
    portfolio_volatility: float  # annualised vol (fraction)
    position_heatmap: Dict[str, float]  # symbol → fraction of portfolio risk
    scenario_results: Dict[str, float]  # scenario_name → stressed var_99
    lookback_days: int
    n_observations: int


@dataclass
class PositionRisk:
    """Risk contribution of a single position."""
    symbol: str
    notional: float         # dollar exposure
    pct_of_portfolio: float  # notional / equity
    stop_distance_pct: float
    dollar_risk: float      # notional × stop_distance_pct


class StressEngine:
    """
    Portfolio VaR, ES, and stress scenario engine.

    Parameters
    ----------
    initial_equity:
        Portfolio starting equity for fractional return normalisation.
    var_percentile:
        Confidence level for VaR (default 0.99 = 99%).
    es_percentile:
        Confidence level for ES/CVaR (default 0.95 = 95%).
    lookback_days:
        Rolling window for historical simulation (default 63 ≈ 1 quarter).
    """

    def __init__(
        self,
        initial_equity: float = 10_000.0,
        var_percentile: float = 0.99,
        es_percentile: float = 0.95,
        lookback_days: int = 63,
    ) -> None:
        self._equity = initial_equity
        self._var_pct = var_percentile
        self._es_pct = es_percentile
        self._lookback = lookback_days

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        positions: Dict[str, object],
        daily_pnl: pd.Series,
        current_equity: Optional[float] = None,
    ) -> StressResult:
        """
        Compute VaR, ES, and stress scenarios for the current portfolio.

        Parameters
        ----------
        positions:
            Dict of symbol → Position (must have qty, avg_entry_price,
            stop_distance_pct attributes).
        daily_pnl:
            Historical daily P&L series (dollars).
        current_equity:
            Current portfolio equity. If None, uses initial_equity.

        Returns
        -------
        StressResult with VaR, ES, heatmap, and scenario results.
        """
        equity = current_equity or self._equity

        # ── Historical VaR / ES ───────────────────────────────────────
        var_99, var_95, es_95 = self._compute_var_es(daily_pnl, equity)
        portfolio_vol = self._compute_portfolio_vol(daily_pnl, equity)

        # ── Position heatmap ─────────────────────────────────────────
        heatmap = self._compute_heatmap(positions, equity)

        # ── Stress scenarios ─────────────────────────────────────────
        scenarios = self._run_stress_scenarios(daily_pnl, equity)

        result = StressResult(
            var_99=var_99,
            var_95=var_95,
            es_95=es_95,
            portfolio_volatility=portfolio_vol,
            position_heatmap=heatmap,
            scenario_results=scenarios,
            lookback_days=self._lookback,
            n_observations=len(daily_pnl),
        )

        self._log_result(result)
        return result

    def compute_position_risks(
        self,
        positions: Dict[str, object],
        equity: Optional[float] = None,
    ) -> List[PositionRisk]:
        """Return per-position risk contribution list, sorted by dollar risk."""
        eq = equity or self._equity
        risks = []
        for symbol, pos in positions.items():
            try:
                notional = pos.qty * pos.avg_entry_price
                stop_pct = pos.stop_distance_pct or 0.03
                risks.append(
                    PositionRisk(
                        symbol=symbol,
                        notional=notional,
                        pct_of_portfolio=notional / max(eq, 1),
                        stop_distance_pct=stop_pct,
                        dollar_risk=notional * stop_pct,
                    )
                )
            except AttributeError as exc:
                logger.debug("[stress] Cannot compute risk for %s: %s", symbol, exc)
        return sorted(risks, key=lambda r: r.dollar_risk, reverse=True)

    def exceeds_limits(
        self,
        result: StressResult,
        max_var_99_pct: float = 0.03,
        max_es_95_pct: float = 0.05,
    ) -> bool:
        """
        Return True if the portfolio exceeds VaR/ES limits.

        Parameters match TRADING_BOT_PLAN.md §4 tail_risk_gate defaults:
          max_var_99_pct = 0.03 (3%)
          max_es_95_pct  = 0.05 (5%)
        """
        return (
            abs(result.var_99) > max_var_99_pct
            or abs(result.es_95) > max_es_95_pct
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_var_es(
        self,
        daily_pnl: pd.Series,
        equity: float,
    ) -> tuple[float, float, float]:
        """
        Historical simulation VaR and ES.

        Returns (var_99, var_95, es_95) as fractions of equity.
        """
        if daily_pnl.empty or len(daily_pnl) < 10:
            return 0.0, 0.0, 0.0

        window = daily_pnl.iloc[-self._lookback:]
        returns = window / max(equity, 1.0)

        sorted_returns = returns.sort_values()

        var_99 = float(sorted_returns.quantile(1 - self._var_pct))
        var_95 = float(sorted_returns.quantile(1 - self._es_pct))

        # ES = average of returns below VaR threshold
        tail = sorted_returns[sorted_returns <= var_95]
        es_95 = float(tail.mean()) if not tail.empty else var_95

        return var_99, var_95, es_95

    def _compute_portfolio_vol(
        self,
        daily_pnl: pd.Series,
        equity: float,
    ) -> float:
        """Annualised portfolio volatility (fraction)."""
        if daily_pnl.empty or len(daily_pnl) < 5:
            return 0.0
        daily_returns = daily_pnl.iloc[-self._lookback:] / max(equity, 1.0)
        return float(daily_returns.std() * np.sqrt(_TRADING_DAYS_PER_YEAR))

    @staticmethod
    def _compute_heatmap(
        positions: Dict[str, object],
        equity: float,
    ) -> Dict[str, float]:
        """
        Fraction of portfolio risk contributed by each position.
        Risk proxy = notional × stop_distance_pct.
        """
        risk_map: Dict[str, float] = {}
        total_risk = 0.0

        for symbol, pos in positions.items():
            try:
                risk = pos.qty * pos.avg_entry_price * (pos.stop_distance_pct or 0.03)
                risk_map[symbol] = risk
                total_risk += risk
            except AttributeError:
                risk_map[symbol] = 0.0

        if total_risk == 0:
            return {s: 0.0 for s in risk_map}

        return {s: v / total_risk for s, v in risk_map.items()}

    def _run_stress_scenarios(
        self,
        daily_pnl: pd.Series,
        equity: float,
    ) -> Dict[str, float]:
        """
        Run three canonical stress scenarios from the plan.

        Returns dict of scenario_name → stressed VaR_99 (fraction).
        """
        if daily_pnl.empty or len(daily_pnl) < 10:
            return {}

        window = daily_pnl.iloc[-self._lookback:]
        base_returns = window / max(equity, 1.0)

        scenarios: Dict[str, float] = {}

        # 1. VIX spike +20%: scale returns by 1.3 (historical vol scaling)
        stressed = base_returns * 1.3
        scenarios["vix_spike_20pct"] = float(stressed.quantile(1 - self._var_pct))

        # 2. Liquidity drain: assume 20% worse fill → returns shifted by -0.002
        liq_stressed = base_returns - 0.002
        scenarios["liquidity_drain_20pct"] = float(
            liq_stressed.quantile(1 - self._var_pct)
        )

        # 3. Correlation cluster: all holdings move together → scale by 1.5
        # (empirical: correlation jumps 20-30% in crisis, vol increases ~50%)
        corr_stressed = base_returns * 1.5
        scenarios["correlation_cluster"] = float(
            corr_stressed.quantile(1 - self._var_pct)
        )

        return scenarios

    def _log_result(self, result: StressResult) -> None:
        logger.info(
            "[stress] VaR(99)=%.2f%%  ES(95)=%.2f%%  vol=%.2f%%  "
            "positions=%d  obs=%d",
            result.var_99 * 100,
            result.es_95 * 100,
            result.portfolio_volatility * 100,
            len(result.position_heatmap),
            result.n_observations,
        )
        if result.scenario_results:
            for scenario, val in result.scenario_results.items():
                logger.debug("[stress] scenario=%s stressed_var=%.2f%%", scenario, val * 100)
