"""
Kelly Statistics — Incremental O(1) EWMA-based stats for Bayesian Kelly sizing.

Replaces sorted() O(N log N) with EWMA O(1) per trade update.
Emits "strategy_edge_zero" metric / log when b_ratio ≈ 0 (no edge detected).

Phase 2 feature — used by BayesianKellySizer once 90+ real trades available.
Phase 1 uses FixedFractional sizing (see portfolio/risk.py).

Usage
-----
    stats = KellyStats(strategy_name="momentum")
    stats.update(pnl=125.0)   # win
    stats.update(pnl=-40.0)   # loss
    fraction = stats.kelly_fraction(max_fraction=0.01)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_EDGE_ZERO_THRESHOLD = 1e-6  # b_ratio below this → emit edge=0 alert


@dataclass
class KellyStats:
    """
    Incremental online statistics — no re-sort on every call.
    O(1) update, O(1) Kelly fraction computation.

    Parameters
    ----------
    strategy_name:
        Used in log / metric emissions for edge=0 alert.
    lam:
        EWMA decay factor (0–1). Higher = slower decay, more history weight.
        Default 0.97 ≈ ~33 trades half-life.
    min_sample_size:
        Below this number of trades, return a conservative prior Kelly fraction
        instead of a data-driven one.
    """

    strategy_name: str = "unknown"
    lam: float = 0.97  # decay factor — recent trades matter more
    min_sample_size: int = 30  # conservative prior until enough data

    n_wins: int = field(default=0, init=False)
    n_losses: int = field(default=0, init=False)
    ewma_win: float = field(default=0.0, init=False)  # EWMA avg win (USD)
    ewma_loss: float = field(default=0.0, init=False)  # EWMA avg loss (USD, positive)

    def update(self, pnl: float) -> None:
        """
        Register a completed trade P&L.

        Parameters
        ----------
        pnl:
            Trade profit/loss in dollars. Positive = win, negative = loss.
        """
        if pnl > 0:
            self.n_wins += 1
            self.ewma_win = self.lam * self.ewma_win + (1 - self.lam) * pnl
        else:
            self.n_losses += 1
            self.ewma_loss = self.lam * self.ewma_loss + (1 - self.lam) * abs(pnl)

    # ------------------------------------------------------------------
    # Derived properties — O(1)
    # ------------------------------------------------------------------

    @property
    def n_trades(self) -> int:
        return self.n_wins + self.n_losses

    @property
    def win_rate(self) -> float:
        """Empirical win rate; 0.5 prior before min_sample_size trades."""
        if self.n_trades < self.min_sample_size:
            return 0.5
        return self.n_wins / self.n_trades

    @property
    def b_ratio(self) -> float:
        """avg_win / avg_loss (odds ratio, b in Kelly formula).

        Returns 0 when avg_loss is effectively zero (no losses observed yet),
        which means the Kelly fraction is undefined / edge unproven.
        """
        if self.ewma_loss < _EDGE_ZERO_THRESHOLD:
            return 0.0
        return self.ewma_win / self.ewma_loss

    @property
    def kelly_fraction(self) -> float:
        """
        Full Kelly fraction: p - q/b

        Returns 0.0 (no bet) when:
          - fewer than min_sample_size trades recorded
          - b_ratio ≈ 0 (no edge)

        Emits 'strategy_edge_zero' log/metric when b=0 is detected.
        """
        if self.n_trades < self.min_sample_size:
            logger.debug(
                "[kelly:%s] Insufficient data (%d/%d trades) — returning 0",
                self.strategy_name,
                self.n_trades,
                self.min_sample_size,
            )
            return 0.0

        b = self.b_ratio
        if b < _EDGE_ZERO_THRESHOLD:
            # ★ NEW P2: emit "strategy_edge_zero" metric + log when b=0
            logger.warning(
                "[kelly:%s] strategy_edge_zero detected: b_ratio=%.6f "
                "(wins=%d losses=%d ewma_win=%.4f ewma_loss=%.4f) — "
                "returning kelly_fraction=0; investigate strategy performance",
                self.strategy_name,
                b,
                self.n_wins,
                self.n_losses,
                self.ewma_win,
                self.ewma_loss,
            )
            # Emit structured metric for monitoring pipeline consumption
            _emit_edge_zero_metric(self.strategy_name, b, self.n_trades)
            return 0.0

        p = self.win_rate
        q = 1.0 - p
        fraction = p - q / b

        if fraction <= 0:
            logger.info(
                "[kelly:%s] Kelly fraction non-positive (%.4f) — no bet",
                self.strategy_name,
                fraction,
            )
            return 0.0

        return fraction

    def capped_kelly(self, max_fraction: float = 0.01) -> float:
        """Return kelly_fraction capped at max_fraction (default 1%)."""
        return min(self.kelly_fraction, max_fraction)

    def reset(self) -> None:
        """Clear all accumulated statistics (e.g., after strategy parameter change)."""
        self.n_wins = 0
        self.n_losses = 0
        self.ewma_win = 0.0
        self.ewma_loss = 0.0


# ---------------------------------------------------------------------------
# Metric emission helper (lightweight; replace with OTel counter in Phase 6)
# ---------------------------------------------------------------------------


def _emit_edge_zero_metric(strategy_name: str, b_ratio: float, n_trades: int) -> None:
    """
    Emit a structured 'strategy_edge_zero' metric.

    In Phase 1-5 this logs a structured record. In Phase 6 swap for:
        otel_counter.add(1, {"strategy": strategy_name})
    """
    logger.warning(
        "METRIC strategy_edge_zero strategy=%s b_ratio=%.8f n_trades=%d",
        strategy_name,
        b_ratio,
        n_trades,
    )
