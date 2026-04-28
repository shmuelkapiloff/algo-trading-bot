"""Monte Carlo simulation and statistical projections.

Provides:
  - Bootstrap Monte Carlo simulation of portfolio equity curves
    (N paths over a 252-trading-day horizon, drawn from trade-return history)
  - Percentile confidence bands (5th, 25th, 50th, 75th, 95th)
  - Annualised return, Sharpe, and max-drawdown estimates per percentile

Usage
-----
    from src.monitoring.statistics import MonteCarloSimulator

    sim = MonteCarloSimulator(n_paths=1000, horizon_days=252, seed=42)
    result = sim.run(trade_returns=[0.012, -0.005, 0.023, ...])
    print(result.summary())

Design
------
  - Trade returns are bootstrapped in contiguous **blocks** (block_size=20)
    rather than i.i.d. draws. Block bootstrap preserves autocorrelation:
    consecutive losses during bear-market regimes cluster together, producing
    realistic drawdown depth. (Efron & Tibshirani, 1993; Politis & Romano, 1994)
  - Each path has ``horizon_days`` compounded steps.
  - An initial capital of 1.0 is assumed; all results are relative.
  - Paths are accumulated as compounded returns: equity[t] = equity[t-1] × (1 + r)
"""

from __future__ import annotations

import logging
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloResult:
    """Output of one Monte Carlo simulation run."""

    n_paths: int
    horizon_days: int
    n_trade_returns: int  # size of the input sample

    # Equity curve percentiles (each is a list of length horizon_days+1)
    # Index 0 = starting equity (1.0), index T = equity at day T
    p05: List[float] = field(default_factory=list)
    p25: List[float] = field(default_factory=list)
    p50: List[float] = field(default_factory=list)
    p75: List[float] = field(default_factory=list)
    p95: List[float] = field(default_factory=list)

    # Terminal equity statistics (day horizon_days)
    terminal_mean: float = 0.0
    terminal_p05: float = 0.0
    terminal_p50: float = 0.0
    terminal_p95: float = 0.0

    # Derived risk metrics (based on median path)
    cagr_p50: float = 0.0
    max_drawdown_p50: float = 0.0
    sharpe_p50: float = 0.0

    # Probability of loss (terminal equity < 1.0)
    prob_loss: float = 0.0

    def summary(self) -> str:
        return (
            f"Monte Carlo ({self.n_paths} paths, {self.horizon_days}d): "
            f"median_return={self.terminal_p50 - 1:.2%}  "
            f"p05={self.terminal_p05 - 1:.2%}  "
            f"p95={self.terminal_p95 - 1:.2%}  "
            f"prob_loss={self.prob_loss:.1%}  "
            f"CAGR(p50)={self.cagr_p50:.2%}  "
            f"MaxDD(p50)={self.max_drawdown_p50:.2%}  "
            f"Sharpe(p50)={self.sharpe_p50:.2f}"
        )


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class MonteCarloSimulator:
    """Bootstrap Monte Carlo simulator for trade-return sequences.

    Parameters
    ----------
    n_paths:
        Number of simulation paths (default 1000).
    horizon_days:
        Number of trading days to simulate per path (default 252 = 1 year).
    seed:
        Random seed for reproducibility (default 42).
    min_sample_size:
        Minimum number of trade returns required before running simulation.
        If fewer returns are available, returns a degenerate result.
    """

    def __init__(
        self,
        n_paths: int = 1000,
        horizon_days: int = 252,
        seed: int = 42,
        min_sample_size: int = 20,
        block_size: int = 20,
    ) -> None:
        self.n_paths = n_paths
        self.horizon_days = horizon_days
        self.seed = seed
        self.min_sample_size = min_sample_size
        self.block_size = block_size

    def run(self, trade_returns: List[float]) -> MonteCarloResult:
        """Run the simulation.

        Parameters
        ----------
        trade_returns:
            List of historical per-trade returns as fractions
            (e.g. 0.012 = +1.2%, -0.005 = -0.5%).
            These are trade-level returns, NOT daily equity returns.

        Returns
        -------
        :class:`MonteCarloResult` with populated statistics.
        """
        result = MonteCarloResult(
            n_paths=self.n_paths,
            horizon_days=self.horizon_days,
            n_trade_returns=len(trade_returns),
        )

        if len(trade_returns) < self.min_sample_size:
            logger.warning(
                "[statistics] Only %d trade returns; need >= %d for Monte Carlo. "
                "Returning degenerate result.",
                len(trade_returns),
                self.min_sample_size,
            )
            flat = [1.0] * (self.horizon_days + 1)
            result.p05 = result.p25 = result.p50 = result.p75 = result.p95 = flat
            result.terminal_mean = result.terminal_p05 = result.terminal_p50 = (
                result.terminal_p95
            ) = 1.0
            return result

        rng = random.Random(self.seed)

        # ── Generate N paths (Block Bootstrap) ───────────────────────────────
        # Samples contiguous blocks of `block_size` returns rather than i.i.d.
        # draws. This preserves regime autocorrelation: consecutive losses in a
        # bear-market phase will cluster together, producing realistic drawdown
        # depth that simple i.i.d. bootstrap underestimates.
        n_returns = len(trade_returns)
        all_paths: List[List[float]] = []
        for _ in range(self.n_paths):
            equity = 1.0
            path = [1.0]
            t = 0
            while t < self.horizon_days:
                start_idx = rng.randint(0, n_returns - 1)
                for offset in range(self.block_size):
                    if t >= self.horizon_days:
                        break
                    r = trade_returns[(start_idx + offset) % n_returns]
                    equity *= 1.0 + r
                    path.append(equity)
                    t += 1
            all_paths.append(path)

        # ── Compute percentile curves ─────────────────────────────────
        result.p05 = self._percentile_curve(all_paths, 5)
        result.p25 = self._percentile_curve(all_paths, 25)
        result.p50 = self._percentile_curve(all_paths, 50)
        result.p75 = self._percentile_curve(all_paths, 75)
        result.p95 = self._percentile_curve(all_paths, 95)

        # ── Terminal statistics ───────────────────────────────────────
        terminal_vals = sorted(p[-1] for p in all_paths)
        n = len(terminal_vals)
        result.terminal_mean = sum(terminal_vals) / n
        result.terminal_p05 = terminal_vals[max(0, int(0.05 * n) - 1)]
        result.terminal_p50 = terminal_vals[int(0.50 * n)]
        result.terminal_p95 = terminal_vals[min(n - 1, int(0.95 * n))]
        result.prob_loss = sum(1 for v in terminal_vals if v < 1.0) / n

        # ── Derived metrics from median path ─────────────────────────
        median_path = result.p50
        years = self.horizon_days / 252.0
        terminal = median_path[-1]
        if terminal > 0 and years > 0:
            result.cagr_p50 = terminal ** (1.0 / years) - 1.0

        # Max drawdown of median path
        peak = median_path[0]
        max_dd = 0.0
        for v in median_path:
            peak = max(peak, v)
            dd = (peak - v) / max(peak, 1e-9)
            max_dd = max(max_dd, dd)
        result.max_drawdown_p50 = max_dd

        # Sharpe of median path (daily returns)
        daily_rets = [
            (median_path[i] - median_path[i - 1]) / max(median_path[i - 1], 1e-9)
            for i in range(1, len(median_path))
        ]
        if len(daily_rets) > 1:
            mu = statistics.mean(daily_rets)
            sigma = statistics.stdev(daily_rets)
            result.sharpe_p50 = (mu / max(sigma, 1e-9)) * math.sqrt(252)

        logger.info("[statistics] %s", result.summary())
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _percentile_curve(paths: List[List[float]], pct: float) -> List[float]:
        """Extract a percentile curve from N simulation paths."""
        n_paths = len(paths)
        n_days = len(paths[0])
        idx = max(0, min(int(pct / 100 * (n_paths - 1)), n_paths - 1))
        curve = []
        for day in range(n_days):
            day_vals = sorted(p[day] for p in paths)
            curve.append(day_vals[idx])
        return curve
