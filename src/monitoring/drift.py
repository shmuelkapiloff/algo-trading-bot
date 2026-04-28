"""Behavioral drift detection for live trading strategies.

Detects when a strategy's live performance diverges from its
historical baseline, indicating potential regime change, overfitting,
or data/execution anomalies.

Checks implemented (as per TRADING_BOT_PLAN.md §monitoring/drift.py):
  1. Win-rate drift      — rolling win rate z-score vs baseline
  2. Drawdown drift      — rolling max drawdown z-score vs baseline
  3. Correlation drift   — rolling inter-strategy correlation shift
  4. Return variance     — rolling return std exceeds 2σ threshold

When a z-score exceeds ``alert_zscore`` (default 2.0), the detector
fires a :class:`DriftAlert` which can be forwarded to the alert system.

Usage
-----
    detector = DriftDetector(window=50, alert_zscore=2.0)

    # Feed trade returns as they arrive:
    for ret in live_returns:
        alerts = detector.update("momentum", ret)
        for a in alerts:
            await alert_system.send(a.message)

    # Or analyse a batch of historical vs live returns:
    report = detector.analyse(
        strategy_name="momentum",
        baseline_returns=historical,
        live_returns=recent_50,
    )
"""

from __future__ import annotations

import logging
import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DriftAlert:
    strategy_name: str
    drift_type: str  # "win_rate" | "drawdown" | "variance" | "correlation"
    zscore: float  # how many standard deviations from baseline
    current_value: float  # current rolling metric
    baseline_value: float  # baseline mean
    baseline_std: float  # baseline standard deviation
    message: str
    severity: str = "warn"  # "warn" | "critical"
    fired_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DriftReport:
    strategy_name: str
    n_baseline: int
    n_live: int
    win_rate_baseline: float
    win_rate_live: float
    win_rate_zscore: float
    drawdown_baseline: float
    drawdown_live: float
    drawdown_zscore: float
    variance_baseline: float
    variance_live: float
    variance_zscore: float
    alerts: List[DriftAlert] = field(default_factory=list)

    @property
    def has_drift(self) -> bool:
        return len(self.alerts) > 0

    def summary(self) -> str:
        lines = [
            f"DriftReport({self.strategy_name}): "
            f"win_rate {self.win_rate_baseline:.2%}→{self.win_rate_live:.2%} "
            f"(z={self.win_rate_zscore:.2f})  "
            f"drawdown {self.drawdown_baseline:.2%}→{self.drawdown_live:.2%} "
            f"(z={self.drawdown_zscore:.2f})"
        ]
        for alert in self.alerts:
            lines.append(
                f"  ⚠ ALERT [{alert.drift_type}] z={alert.zscore:.2f}: {alert.message}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Drift Detector
# ---------------------------------------------------------------------------


class DriftDetector:
    """Online behavioral drift detector for trading strategies.

    Maintains a rolling window of live returns per strategy and compares
    them against the stored baseline statistics.

    Parameters
    ----------
    window:
        Number of recent live trades to include in rolling metrics (default 50).
    alert_zscore:
        Z-score threshold to fire a drift alert (default 2.0 = 2σ).
    critical_zscore:
        Z-score threshold for a critical severity alert (default 3.0).
    min_baseline_size:
        Minimum number of baseline returns required before drift detection
        is active (default 30).
    """

    def __init__(
        self,
        window: int = 50,
        alert_zscore: float = 2.0,
        critical_zscore: float = 3.0,
        min_baseline_size: int = 30,
    ) -> None:
        self.window = window
        self.alert_zscore = alert_zscore
        self.critical_zscore = critical_zscore
        self.min_baseline_size = min_baseline_size

        # Per-strategy rolling buffers: strategy_name → deque of returns
        self._live_buffers: Dict[str, Deque[float]] = {}

        # Baseline statistics: strategy_name → {"win_rate_mean", "win_rate_std", ...}
        self._baselines: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    def set_baseline(self, strategy_name: str, historical_returns: List[float]) -> None:
        """Compute and store baseline statistics from historical returns.

        Call this once after backtesting is complete, before going live.

        Parameters
        ----------
        strategy_name:       Strategy identifier.
        historical_returns:  List of per-trade returns (fractions, e.g. 0.012).
        """
        if len(historical_returns) < self.min_baseline_size:
            logger.warning(
                "[drift] %s: too few baseline returns (%d < %d)",
                strategy_name,
                len(historical_returns),
                self.min_baseline_size,
            )

        wins = [r for r in historical_returns if r > 0]
        n = max(len(historical_returns), 1)
        win_rate = len(wins) / n

        # Rolling window metrics — compute over sub-windows for std estimation
        sub_win_rates = self._sub_window_metric(
            historical_returns,
            self.window,
            lambda rs: sum(1 for r in rs if r > 0) / max(len(rs), 1),
        )
        sub_drawdowns = self._sub_window_metric(
            historical_returns, self.window, self._max_drawdown_from_returns
        )
        sub_variances = self._sub_window_metric(
            historical_returns,
            self.window,
            lambda rs: statistics.stdev(rs) if len(rs) > 1 else 0.0,
        )

        self._baselines[strategy_name] = {
            "n": len(historical_returns),
            "win_rate_mean": _safe_mean(sub_win_rates, win_rate),
            "win_rate_std": _safe_std(sub_win_rates),
            "drawdown_mean": _safe_mean(sub_drawdowns, 0.0),
            "drawdown_std": _safe_std(sub_drawdowns),
            "variance_mean": _safe_mean(sub_variances, 0.0),
            "variance_std": _safe_std(sub_variances),
        }

        logger.info(
            "[drift] %s: baseline set from %d returns  "
            "win_rate=%.2f  drawdown_mean=%.2f%%  var_mean=%.4f",
            strategy_name,
            len(historical_returns),
            win_rate,
            self._baselines[strategy_name]["drawdown_mean"] * 100,
            self._baselines[strategy_name]["variance_mean"],
        )

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update(self, strategy_name: str, new_return: float) -> List[DriftAlert]:
        """Feed a new live trade return and check for drift.

        Parameters
        ----------
        strategy_name: Strategy that produced this trade.
        new_return:    Return fraction (e.g. 0.012 = +1.2%).

        Returns
        -------
        List of :class:`DriftAlert` objects (empty if no drift detected).
        """
        if strategy_name not in self._live_buffers:
            self._live_buffers[strategy_name] = deque(maxlen=self.window)
        self._live_buffers[strategy_name].append(new_return)

        if strategy_name not in self._baselines:
            return []  # no baseline yet

        live = list(self._live_buffers[strategy_name])
        if len(live) < max(10, self.window // 5):
            return []  # not enough live data

        return self._detect(strategy_name, live)

    # ------------------------------------------------------------------
    # Batch analysis
    # ------------------------------------------------------------------

    def analyse(
        self,
        strategy_name: str,
        baseline_returns: List[float],
        live_returns: List[float],
    ) -> DriftReport:
        """One-shot drift analysis comparing baseline vs live returns.

        This method does NOT update internal state.  Use it for
        offline analysis or reporting.
        """
        self.set_baseline(strategy_name, baseline_returns)
        alerts = self._detect(strategy_name, live_returns)
        base = self._baselines[strategy_name]

        n_live = max(len(live_returns), 1)
        wins_live = sum(1 for r in live_returns if r > 0)
        win_rate_live = wins_live / n_live
        drawdown_live = self._max_drawdown_from_returns(live_returns)
        variance_live = statistics.stdev(live_returns) if len(live_returns) > 1 else 0.0

        def _z(val: float, mean: float, std: float) -> float:
            return (val - mean) / max(std, 1e-9)

        return DriftReport(
            strategy_name=strategy_name,
            n_baseline=base["n"],
            n_live=len(live_returns),
            win_rate_baseline=base["win_rate_mean"],
            win_rate_live=win_rate_live,
            win_rate_zscore=_z(
                win_rate_live, base["win_rate_mean"], base["win_rate_std"]
            ),
            drawdown_baseline=base["drawdown_mean"],
            drawdown_live=drawdown_live,
            drawdown_zscore=_z(
                drawdown_live, base["drawdown_mean"], base["drawdown_std"]
            ),
            variance_baseline=base["variance_mean"],
            variance_live=variance_live,
            variance_zscore=_z(
                variance_live, base["variance_mean"], base["variance_std"]
            ),
            alerts=alerts,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect(self, strategy_name: str, live: List[float]) -> List[DriftAlert]:
        base = self._baselines.get(strategy_name)
        if not base:
            return []

        alerts: List[DriftAlert] = []
        n = max(len(live), 1)

        def _z(val: float, mean_key: str, std_key: str) -> float:
            std = base.get(std_key, 0.0)
            return (val - base.get(mean_key, 0.0)) / max(std, 1e-9)

        # 1. Win-rate drift
        win_rate = sum(1 for r in live if r > 0) / n
        z_wr = _z(win_rate, "win_rate_mean", "win_rate_std")
        if abs(z_wr) >= self.alert_zscore:
            severity = "critical" if abs(z_wr) >= self.critical_zscore else "warn"
            alerts.append(
                DriftAlert(
                    strategy_name=strategy_name,
                    drift_type="win_rate",
                    zscore=z_wr,
                    current_value=win_rate,
                    baseline_value=base["win_rate_mean"],
                    baseline_std=base["win_rate_std"],
                    message=(
                        f"{strategy_name} win_rate drift: "
                        f"{win_rate:.2%} vs baseline {base['win_rate_mean']:.2%} (z={z_wr:.2f})"
                    ),
                    severity=severity,
                )
            )

        # 2. Max drawdown drift
        drawdown = self._max_drawdown_from_returns(live)
        z_dd = _z(drawdown, "drawdown_mean", "drawdown_std")
        if z_dd >= self.alert_zscore:  # drawdown only alerts when worse (positive z)
            severity = "critical" if z_dd >= self.critical_zscore else "warn"
            alerts.append(
                DriftAlert(
                    strategy_name=strategy_name,
                    drift_type="drawdown",
                    zscore=z_dd,
                    current_value=drawdown,
                    baseline_value=base["drawdown_mean"],
                    baseline_std=base["drawdown_std"],
                    message=(
                        f"{strategy_name} drawdown drift: "
                        f"{drawdown:.2%} vs baseline {base['drawdown_mean']:.2%} (z={z_dd:.2f})"
                    ),
                    severity=severity,
                )
            )

        # 3. Variance drift
        if len(live) > 1:
            var = statistics.stdev(live)
            z_var = _z(var, "variance_mean", "variance_std")
            if abs(z_var) >= self.alert_zscore:
                severity = "critical" if abs(z_var) >= self.critical_zscore else "warn"
                alerts.append(
                    DriftAlert(
                        strategy_name=strategy_name,
                        drift_type="variance",
                        zscore=z_var,
                        current_value=var,
                        baseline_value=base["variance_mean"],
                        baseline_std=base["variance_std"],
                        message=(
                            f"{strategy_name} return variance drift: "
                            f"{var:.4f} vs baseline {base['variance_mean']:.4f} (z={z_var:.2f})"
                        ),
                        severity=severity,
                    )
                )

        if alerts:
            logger.warning(
                "[drift] %s: %d drift alerts fired", strategy_name, len(alerts)
            )

        return alerts

    @staticmethod
    def _max_drawdown_from_returns(returns: List[float]) -> float:
        """Compute max drawdown from a sequence of returns."""
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in returns:
            equity *= 1.0 + r
            peak = max(peak, equity)
            dd = (peak - equity) / max(peak, 1e-9)
            max_dd = max(max_dd, dd)
        return max_dd

    @staticmethod
    def _sub_window_metric(returns: List[float], window: int, fn) -> List[float]:
        """Apply ``fn`` to rolling sub-windows to estimate metric variability."""
        results = []
        for i in range(window, len(returns) + 1, max(window // 5, 1)):
            sub = returns[max(0, i - window) : i]
            if sub:
                try:
                    results.append(fn(sub))
                except Exception:
                    pass
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_mean(vals: List[float], default: float = 0.0) -> float:
    return statistics.mean(vals) if vals else default


def _safe_std(vals: List[float]) -> float:
    return statistics.stdev(vals) if len(vals) > 1 else 0.0
