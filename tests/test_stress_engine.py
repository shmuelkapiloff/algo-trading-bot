"""Tests for src/risk/stress_engine.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.stress_engine import StressEngine, StressResult


def _make_pnl(n: int = 100, seed: int = 42, daily_vol: float = 100.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, daily_vol, n))


def _make_positions(n_symbols: int = 3, equity: float = 10_000.0) -> dict:
    """Create dummy position objects for testing."""
    positions = {}
    for i in range(n_symbols):
        symbol = f"SYM{i}"
        pos = type("Position", (), {
            "qty": 10,
            "avg_entry_price": equity / (n_symbols * 10),
            "stop_distance_pct": 0.03,
        })()
        positions[symbol] = pos
    return positions


class TestStressEngineInit:
    def test_default_parameters(self):
        engine = StressEngine()
        assert engine._var_pct == 0.99
        assert engine._es_pct == 0.95
        assert engine._lookback == 63

    def test_custom_parameters(self):
        engine = StressEngine(
            initial_equity=50_000.0,
            var_percentile=0.95,
            es_percentile=0.90,
            lookback_days=30,
        )
        assert engine._var_pct == 0.95
        assert engine._lookback == 30


class TestVaRComputation:
    def test_var_99_is_negative_loss(self):
        engine = StressEngine(initial_equity=10_000.0)
        pnl = _make_pnl(200)
        result = engine.compute({}, pnl)
        # VaR at 99% should be a loss (negative return)
        assert result.var_99 < 0.0

    def test_var_99_worse_than_var_95(self):
        engine = StressEngine(initial_equity=10_000.0)
        pnl = _make_pnl(200)
        result = engine.compute({}, pnl)
        # 99% VaR is more extreme (more negative) than 95%
        assert result.var_99 <= result.var_95

    def test_es_95_worse_than_var_95(self):
        """ES/CVaR should be more extreme than VaR at same confidence."""
        engine = StressEngine(initial_equity=10_000.0)
        pnl = _make_pnl(200)
        result = engine.compute({}, pnl)
        assert result.es_95 <= result.var_95

    def test_empty_pnl_returns_zeros(self):
        engine = StressEngine(initial_equity=10_000.0)
        result = engine.compute({}, pd.Series([], dtype=float))
        assert result.var_99 == 0.0
        assert result.es_95 == 0.0


class TestPortfolioVol:
    def test_nonzero_vol_for_noisy_returns(self):
        engine = StressEngine(initial_equity=10_000.0)
        pnl = _make_pnl(200, daily_vol=100.0)
        result = engine.compute({}, pnl)
        assert result.portfolio_volatility > 0.0

    def test_zero_vol_for_constant_returns(self):
        engine = StressEngine(initial_equity=10_000.0)
        pnl = pd.Series([50.0] * 100)  # constant daily P&L
        result = engine.compute({}, pnl)
        # Floating point: near zero (< 1e-10) counts as zero vol
        assert result.portfolio_volatility < 1e-10


class TestPositionHeatmap:
    def test_heatmap_sums_to_one(self):
        engine = StressEngine(initial_equity=10_000.0)
        positions = _make_positions(3)
        pnl = _make_pnl(100)
        result = engine.compute(positions, pnl)
        total = sum(result.position_heatmap.values())
        assert abs(total - 1.0) < 1e-6

    def test_heatmap_keys_are_symbols(self):
        engine = StressEngine(initial_equity=10_000.0)
        positions = _make_positions(3)
        pnl = _make_pnl(100)
        result = engine.compute(positions, pnl)
        assert set(result.position_heatmap.keys()) == set(positions.keys())

    def test_empty_positions_heatmap(self):
        engine = StressEngine(initial_equity=10_000.0)
        pnl = _make_pnl(100)
        result = engine.compute({}, pnl)
        assert result.position_heatmap == {}


class TestStressScenarios:
    def test_all_three_scenarios_present(self):
        engine = StressEngine(initial_equity=10_000.0)
        pnl = _make_pnl(100)
        result = engine.compute({}, pnl)
        assert "vix_spike_20pct" in result.scenario_results
        assert "liquidity_drain_20pct" in result.scenario_results
        assert "correlation_cluster" in result.scenario_results

    def test_stressed_var_worse_than_base(self):
        """VIX spike scenario should produce worse VaR than base."""
        engine = StressEngine(initial_equity=10_000.0)
        pnl = _make_pnl(200)
        result = engine.compute({}, pnl)
        assert result.scenario_results["vix_spike_20pct"] <= result.var_99

    def test_empty_pnl_no_scenarios(self):
        engine = StressEngine(initial_equity=10_000.0)
        result = engine.compute({}, pd.Series([], dtype=float))
        assert result.scenario_results == {}


class TestExceedsLimits:
    def test_exceeds_when_var_large(self):
        engine = StressEngine(initial_equity=10_000.0)
        result = StressResult(
            var_99=-0.05,  # 5% loss > 3% limit
            var_95=-0.04,
            es_95=-0.06,
            portfolio_volatility=0.2,
            position_heatmap={},
            scenario_results={},
            lookback_days=63,
            n_observations=100,
        )
        assert engine.exceeds_limits(result) is True

    def test_within_limits(self):
        engine = StressEngine(initial_equity=10_000.0)
        result = StressResult(
            var_99=-0.02,  # 2% loss < 3% limit
            var_95=-0.01,
            es_95=-0.04,   # 4% < 5% limit
            portfolio_volatility=0.15,
            position_heatmap={},
            scenario_results={},
            lookback_days=63,
            n_observations=100,
        )
        assert engine.exceeds_limits(result) is False

    def test_es_breach_triggers_limit(self):
        engine = StressEngine(initial_equity=10_000.0)
        result = StressResult(
            var_99=-0.02,  # within VaR limit
            var_95=-0.01,
            es_95=-0.06,   # 6% > 5% ES limit
            portfolio_volatility=0.15,
            position_heatmap={},
            scenario_results={},
            lookback_days=63,
            n_observations=100,
        )
        assert engine.exceeds_limits(result) is True


class TestComputePositionRisks:
    def test_returns_sorted_by_dollar_risk(self):
        engine = StressEngine(initial_equity=10_000.0)
        # Create positions with different risk levels
        pos_big = type("P", (), {"qty": 100, "avg_entry_price": 100.0, "stop_distance_pct": 0.05})()
        pos_small = type("P", (), {"qty": 10, "avg_entry_price": 50.0, "stop_distance_pct": 0.02})()
        positions = {"BIG": pos_big, "SMALL": pos_small}

        risks = engine.compute_position_risks(positions, equity=10_000.0)
        assert len(risks) == 2
        # BIG should have higher dollar risk
        assert risks[0].symbol == "BIG"
        assert risks[0].dollar_risk > risks[1].dollar_risk

    def test_pct_of_portfolio_sums_approx(self):
        engine = StressEngine(initial_equity=10_000.0)
        positions = _make_positions(2, equity=10_000.0)
        risks = engine.compute_position_risks(positions, equity=10_000.0)
        total_pct = sum(r.pct_of_portfolio for r in risks)
        # Should be <= 1.0 (can be less if positions don't fill all equity)
        assert 0 < total_pct <= 1.5  # allow a little over due to test positions
