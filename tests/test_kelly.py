"""
Tests for Kelly Statistics and Bayesian Kelly sizing.

Tests verify:
  - KellyStats EWMA update mechanics
  - kelly_fraction() with wins/losses
  - Edge-zero detection (b_ratio ≈ 0)
  - Conservative prior with insufficient data
  - Winsorization caps (max_fraction enforced)
"""

from __future__ import annotations

import pytest

from src.portfolio.kelly_stats import KellyStats


class TestKellyStats:
    def test_initial_fraction_conservative(self):
        """With no data, should return the conservative prior (0.0)."""
        stats = KellyStats(strategy_name="test")
        frac = stats.capped_kelly(max_fraction=0.05)
        assert frac == 0.0

    def test_update_wins_increases_fraction(self):
        """Adding wins should result in positive Kelly fraction."""
        stats = KellyStats(strategy_name="test", min_sample_size=3)
        for _ in range(5):
            stats.update(pnl=100.0)
        for _ in range(2):
            stats.update(pnl=-50.0)
        frac = stats.capped_kelly(max_fraction=0.25)
        assert frac > 0.0

    def test_max_fraction_cap_enforced(self):
        """Kelly fraction must never exceed max_fraction."""
        stats = KellyStats(strategy_name="test", min_sample_size=2)
        for _ in range(20):
            stats.update(pnl=500.0)
        for _ in range(2):
            stats.update(pnl=-10.0)
        frac = stats.capped_kelly(max_fraction=0.02)
        assert frac <= 0.02

    def test_edge_zero_when_no_wins(self):
        """All losses → zero Kelly fraction."""
        stats = KellyStats(strategy_name="test", min_sample_size=3)
        for _ in range(10):
            stats.update(pnl=-50.0)
        frac = stats.capped_kelly(max_fraction=0.05)
        assert frac == 0.0

    def test_win_loss_counts(self):
        """n_wins and n_losses should track correctly."""
        stats = KellyStats()
        stats.update(100.0)
        stats.update(-50.0)
        stats.update(200.0)
        assert stats.n_wins == 2
        assert stats.n_losses == 1

    def test_ewma_recent_trades_weighted_more(self):
        """EWMA should weight recent trades more heavily than old ones."""
        stats = KellyStats(strategy_name="test", lam=0.5, min_sample_size=3)
        # Old losses
        for _ in range(5):
            stats.update(pnl=-100.0)
        # Recent wins (large)
        for _ in range(5):
            stats.update(pnl=500.0)
        frac_high_lam = stats.capped_kelly(max_fraction=1.0)
        # With lam=0.5 (fast decay), recent wins dominate → positive fraction
        assert frac_high_lam > 0.0

    def test_fraction_non_negative(self):
        """Kelly fraction should never be negative."""
        stats = KellyStats(strategy_name="test", min_sample_size=1)
        stats.update(-100.0)
        stats.update(-50.0)
        frac = stats.capped_kelly(max_fraction=0.05)
        assert frac >= 0.0
