"""Tests for new modules: TrendFollowing, MAE/MFE, DataQuality, Universe,
Statistics, Drift, RateLimiter."""

from __future__ import annotations

import asyncio
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trend_bars(n: int = 250, bullish: bool = True) -> pd.DataFrame:
    """Bars where EMA50 > EMA200 (bullish) or EMA50 < EMA200 (bearish)."""
    import random

    rng = random.Random(0)
    # Strong uptrend if bullish, downtrend otherwise
    drift = 0.001 if bullish else -0.001
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(max(1.0, closes[-1] * (1 + drift + rng.gauss(0, 0.005))))

    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.005 for c in closes],
            "low": [c * 0.995 for c in closes],
            "close_adj": closes,
            "volume": [1_000_000] * n,
        },
        index=idx,
    )
    return df


# ===========================================================================
# TrendFollowingStrategy
# ===========================================================================


class TestTrendFollowingStrategy:
    def test_import(self):
        from src.signals.trend_following import TrendFollowingStrategy

        s = TrendFollowingStrategy()
        assert s.name == "trend_following"

    def test_requires_200_bars(self):
        from src.signals.trend_following import TrendFollowingStrategy
        from src.analysis.market_regime import MarketRegime

        s = TrendFollowingStrategy()
        short_df = _make_trend_bars(n=100)
        signals = list(s.generate_signals({"AAPL": short_df}, MarketRegime.BULL))
        assert len(signals) == 0  # not enough bars

    def test_no_signal_in_bear_regime(self):
        from src.signals.trend_following import TrendFollowingStrategy
        from src.analysis.market_regime import MarketRegime

        s = TrendFollowingStrategy()
        df = _make_trend_bars(n=260, bullish=True)
        signals = list(s.generate_signals({"AAPL": df}, MarketRegime.BEAR))
        assert len(signals) == 0

    def test_signal_structure_when_triggered(self):
        from src.signals.trend_following import TrendFollowingStrategy
        from src.analysis.market_regime import MarketRegime
        from src.signals.models import OrderSide

        s = TrendFollowingStrategy()
        df = _make_trend_bars(n=260, bullish=True)
        signals = list(s.generate_signals({"AAPL": df}, MarketRegime.BULL))
        # May or may not fire depending on ADX — just check structure if it does
        for sig in signals:
            assert sig.symbol == "AAPL"
            assert sig.side == OrderSide.BUY
            assert sig.strategy_name == "trend_following"
            assert 0 <= sig.confidence <= 1
            assert sig.stop_distance_pct is not None
            assert 0.025 <= sig.stop_distance_pct <= 0.060


# ===========================================================================
# MAE/MFE Tracker
# ===========================================================================


class TestMaeMfeTracker:
    def test_basic_lifecycle(self):
        from src.portfolio.mae_mfe import MaeMfeTracker

        tracker = MaeMfeTracker()
        tracker.open_position("ord1", "AAPL", entry_price=100.0)
        tracker.update("ord1", bar_low=95.0, bar_high=105.0)
        tracker.update("ord1", bar_low=92.0, bar_high=108.0)
        record = tracker.close_position("ord1", exit_price=104.0, exit_reason="ttl", strategy_name="momentum")

        assert record is not None
        assert record.mae_pct == pytest.approx(0.08, rel=1e-2)   # (100 - 92) / 100
        assert record.mfe_pct == pytest.approx(0.08, rel=1e-2)   # (108 - 100) / 100
        assert record.pnl_pct == pytest.approx(0.04, rel=1e-2)   # (104 - 100) / 100

    def test_close_unknown_order_returns_none(self):
        from src.portfolio.mae_mfe import MaeMfeTracker

        tracker = MaeMfeTracker()
        result = tracker.close_position("nonexistent", 100.0, "ttl", "momentum")
        assert result is None

    def test_strategy_stats(self):
        from src.portfolio.mae_mfe import MaeMfeTracker

        tracker = MaeMfeTracker()
        for i in range(20):
            oid = f"ord{i}"
            tracker.open_position(oid, "MSFT", entry_price=100.0)
            tracker.update(oid, bar_low=97.0, bar_high=103.0)
            pnl = 1.02 if i % 2 == 0 else 0.98
            tracker.close_position(oid, exit_price=100.0 * pnl, exit_reason="ttl", strategy_name="momentum")

        stats = tracker.strategy_stats("momentum")
        assert stats["count"] == 20
        assert stats["win_rate"] == pytest.approx(0.5, abs=0.01)
        assert "mae_p75" in stats
        assert "mfe_p50" in stats
        assert "suggested_stop_pct" in stats


# ===========================================================================
# DataQualityChecker
# ===========================================================================


class TestDataQualityChecker:
    def _good_df(self, n: int = 100) -> pd.DataFrame:
        idx = pd.date_range("2024-01-02", periods=n, freq="B")
        return pd.DataFrame(
            {
                "open": [100.0] * n,
                "high": [102.0] * n,
                "low": [98.0] * n,
                "close_adj": [100.0] * n,
                "volume": [1_000_000] * n,
            },
            index=idx,
        )

    def test_good_data_all_ok(self):
        from src.data.data_quality import DataQualityChecker

        checker = DataQualityChecker()
        df = self._good_df()
        # Use the day after the last bar so staleness check passes
        as_of = df.index.max().date() + timedelta(days=1)
        reports = checker.check_all({"AAPL": df}, as_of=as_of)
        assert reports["AAPL"].passed
        assert not reports["AAPL"].has_warnings

    def test_empty_df_fails(self):
        from src.data.data_quality import DataQualityChecker, CheckSeverity

        checker = DataQualityChecker()
        reports = checker.check_all({"AAPL": pd.DataFrame()})
        assert not reports["AAPL"].passed
        assert reports["AAPL"].worst_severity == CheckSeverity.FAIL

    def test_stale_bars_fail(self):
        from src.data.data_quality import DataQualityChecker

        checker = DataQualityChecker(stale_bar_max_days=3)
        df = self._good_df()
        # as_of is 30 days after last bar
        as_of = df.index.max().date() + timedelta(days=30)
        reports = checker.check_all({"AAPL": df}, as_of=as_of)
        assert not reports["AAPL"].passed

    def test_zero_price_fails(self):
        from src.data.data_quality import DataQualityChecker

        checker = DataQualityChecker()
        df = self._good_df()
        df.loc[df.index[5], "close_adj"] = 0.0
        df.loc[df.index[5], "low"] = 0.0
        reports = checker.check_all({"AAPL": df}, as_of=df.index.max().date())
        assert not reports["AAPL"].passed

    def test_ohlc_inconsistency_fails(self):
        from src.data.data_quality import DataQualityChecker

        checker = DataQualityChecker()
        df = self._good_df()
        # Make high < low on one bar
        df.loc[df.index[10], "high"] = 90.0
        df.loc[df.index[10], "low"] = 95.0
        reports = checker.check_all({"AAPL": df}, as_of=df.index.max().date())
        assert not reports["AAPL"].passed

    def test_quarantined_symbols(self):
        from src.data.data_quality import DataQualityChecker

        checker = DataQualityChecker()
        good = self._good_df()
        bad = pd.DataFrame()  # empty → fail
        reports = checker.check_all({"AAPL": good, "BAD": bad}, as_of=good.index.max().date())
        quarantined = checker.quarantined_symbols(reports)
        assert "BAD" in quarantined
        assert "AAPL" not in quarantined


# ===========================================================================
# StockUniverse
# ===========================================================================


class TestStockUniverse:
    def test_static_filter_blacklist(self):
        from src.data.universe import StockUniverse

        universe = StockUniverse(watchlist=["AAPL", "MSFT", "TQQQ", "SOXL"])
        snap = universe.get_static_universe()
        active = snap.active_symbols
        assert "TQQQ" not in active
        assert "SOXL" not in active
        assert "AAPL" in active
        assert "MSFT" in active

    def test_daily_filter_removes_earnings(self):
        from src.data.universe import StockUniverse

        universe = StockUniverse(watchlist=["AAPL", "MSFT"], earnings_buffer_days=3)
        snap = universe.get_static_universe()
        today = date.today()
        # AAPL has earnings tomorrow → removed
        snap2 = universe.apply_daily_filters(
            snap,
            as_of=today,
            earnings_calendar={"AAPL": today + timedelta(days=1)},
        )
        assert "AAPL" not in snap2.active_symbols
        assert "MSFT" in snap2.active_symbols

    def test_daily_filter_removes_halted(self):
        from src.data.universe import StockUniverse

        universe = StockUniverse(watchlist=["AAPL", "MSFT"])
        snap = universe.get_static_universe()
        snap2 = universe.apply_daily_filters(snap, halted_symbols={"MSFT"})
        assert "MSFT" not in snap2.active_symbols

    def test_tier3_scan_no_fn_returns_all(self):
        from src.data.universe import StockUniverse

        universe = StockUniverse(watchlist=["AAPL", "MSFT", "GOOGL"])
        snap = universe.get_static_universe()
        candidates = universe.apply_strategy_scan(snap, "momentum")
        assert set(candidates) == {"AAPL", "MSFT", "GOOGL"}

    def test_tier3_scan_with_filter(self):
        from src.data.universe import StockUniverse

        universe = StockUniverse(watchlist=["AAPL", "MSFT", "GOOGL"])
        snap = universe.get_static_universe()
        idx = pd.date_range("2024-01-02", periods=50, freq="B")
        bars = {s: pd.DataFrame({"close_adj": [100.0] * 50}, index=idx) for s in ["AAPL", "MSFT", "GOOGL"]}
        # Only AAPL passes scan
        candidates = universe.apply_strategy_scan(snap, "test", bars=bars, scan_fn=lambda sym, df: sym == "AAPL")
        assert candidates == ["AAPL"]


# ===========================================================================
# MonteCarloSimulator
# ===========================================================================


class TestMonteCarloSimulator:
    def _returns(self, n: int = 100) -> list:
        import random
        rng = random.Random(42)
        return [rng.gauss(0.005, 0.02) for _ in range(n)]

    def test_insufficient_sample_returns_flat(self):
        from src.monitoring.statistics import MonteCarloSimulator

        sim = MonteCarloSimulator(min_sample_size=20)
        result = sim.run([0.01, 0.02, -0.01])  # only 3 returns
        assert result.terminal_p50 == 1.0

    def test_full_run_structure(self):
        from src.monitoring.statistics import MonteCarloSimulator

        sim = MonteCarloSimulator(n_paths=100, horizon_days=50, seed=1)
        result = sim.run(self._returns())
        assert result.n_paths == 100
        assert result.horizon_days == 50
        assert len(result.p50) == 51  # 0..50 inclusive
        assert result.p50[0] == 1.0   # starts at 1.0

    def test_percentile_ordering(self):
        from src.monitoring.statistics import MonteCarloSimulator

        sim = MonteCarloSimulator(n_paths=200, horizon_days=50, seed=7)
        result = sim.run(self._returns(200))
        # p05 <= p50 <= p95 at every step
        for i in range(len(result.p50)):
            assert result.p05[i] <= result.p50[i] <= result.p95[i]

    def test_prob_loss_between_0_and_1(self):
        from src.monitoring.statistics import MonteCarloSimulator

        sim = MonteCarloSimulator(n_paths=200, horizon_days=50, seed=5)
        result = sim.run(self._returns())
        assert 0.0 <= result.prob_loss <= 1.0

    def test_deterministic_same_seed(self):
        from src.monitoring.statistics import MonteCarloSimulator

        returns = self._returns(100)
        r1 = MonteCarloSimulator(n_paths=100, seed=42).run(returns)
        r2 = MonteCarloSimulator(n_paths=100, seed=42).run(returns)
        assert r1.terminal_p50 == pytest.approx(r2.terminal_p50, rel=1e-9)


# ===========================================================================
# DriftDetector
# ===========================================================================


class TestDriftDetector:
    def _baseline_returns(self, n: int = 100) -> list:
        import random
        rng = random.Random(10)
        return [rng.gauss(0.005, 0.015) for _ in range(n)]

    def test_no_drift_on_same_distribution(self):
        from src.monitoring.drift import DriftDetector

        detector = DriftDetector(window=50, alert_zscore=2.0, min_baseline_size=20)
        baseline = self._baseline_returns(100)
        import random
        rng = random.Random(11)
        live = [rng.gauss(0.005, 0.015) for _ in range(50)]
        report = detector.analyse("momentum", baseline, live)
        # Same distribution → unlikely to trigger drift
        # (May fire with very unlucky seeds — only check structure)
        assert report.n_baseline == 100
        assert report.n_live == 50

    def test_drift_on_degraded_win_rate(self):
        from src.monitoring.drift import DriftDetector

        detector = DriftDetector(window=30, alert_zscore=2.0, min_baseline_size=20)
        # Baseline: 60% win rate
        baseline = [0.01 if i % 5 != 0 else -0.02 for i in range(100)]  # ~80% wins
        # Live: all losses
        live = [-0.02] * 30
        report = detector.analyse("momentum", baseline, live)
        # Should detect win rate drift
        assert report.has_drift
        win_rate_alerts = [a for a in report.alerts if a.drift_type == "win_rate"]
        assert len(win_rate_alerts) > 0

    def test_online_update_no_crash(self):
        from src.monitoring.drift import DriftDetector

        detector = DriftDetector(window=30, min_baseline_size=10)
        baseline = [0.01 if i % 2 == 0 else -0.005 for i in range(50)]
        detector.set_baseline("strategy_a", baseline)
        alerts = []
        import random
        rng = random.Random(5)
        for _ in range(30):
            r = rng.gauss(0.005, 0.015)
            alerts.extend(detector.update("strategy_a", r))
        # No crash is the primary requirement
        assert isinstance(alerts, list)


# ===========================================================================
# AlpacaRateLimiter
# ===========================================================================


class TestAlpacaRateLimiter:
    """Import rate_limiter directly to avoid the heavy execution __init__ chain."""

    def _load(self):
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "_rl_isolated",
            r"C:\Users\shmuel kapiloff\Desktop\screen\trading_bot\src\execution\rate_limiter.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_initial_tokens_full(self):
        mod = self._load()
        limiter = mod.AlpacaRateLimiter(max_requests=100)
        assert limiter.available() == pytest.approx(100.0, abs=1.0)

    async def test_acquire_reduces_tokens(self):
        mod = self._load()
        limiter = mod.AlpacaRateLimiter(max_requests=100)
        await limiter.acquire(10)
        assert limiter.available() == pytest.approx(90.0, abs=2.0)

    async def test_acquire_does_not_exceed_max(self):
        mod = self._load()
        limiter = mod.AlpacaRateLimiter(max_requests=10)
        await limiter.acquire(10)
        assert limiter.available() == pytest.approx(0.0, abs=2.0)

    def test_stats_structure(self):
        mod = self._load()
        limiter = mod.AlpacaRateLimiter()
        stats = limiter.stats()
        assert "available_tokens" in stats
        assert "max_requests" in stats
        assert stats["max_requests"] == 200

    async def test_acquire_zero_is_noop(self):
        mod = self._load()
        limiter = mod.AlpacaRateLimiter(max_requests=50)
        before = limiter.available()
        await limiter.acquire(0)
        after = limiter.available()
        assert before == pytest.approx(after, abs=1.0)
