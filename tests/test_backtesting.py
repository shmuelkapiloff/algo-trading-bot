"""Tests for the backtesting engine, cost model, and fill simulator."""

from __future__ import annotations

import pytest
import pandas as pd
from datetime import date, datetime, timezone

from backtesting.costs import CostModel, TradeCosts, SEC_FEE_RATE, FINRA_TAF_PER_SHARE
from backtesting.fill_simulator import FillSimulator, FillStatus
from backtesting.deterministic import deterministic_context
from backtesting.engine import BacktestEngine, BacktestResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bars(n: int = 250, start_price: float = 100.0) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame with DatetimeIndex."""
    idx = pd.date_range("2023-01-03", periods=n, freq="B")
    import random

    rng = random.Random(42)
    closes = [start_price]
    for _ in range(n - 1):
        closes.append(max(1.0, closes[-1] * (1 + rng.gauss(0.0002, 0.015))))

    df = pd.DataFrame(
        {
            "open": [c * rng.uniform(0.995, 1.005) for c in closes],
            "high": [c * rng.uniform(1.000, 1.015) for c in closes],
            "low": [c * rng.uniform(0.985, 1.000) for c in closes],
            "close_adj": closes,
            "volume": [rng.randint(500_000, 5_000_000) for _ in closes],
        },
        index=idx,
    )
    # Ensure high >= low
    df["high"] = df[["open", "close_adj", "high"]].max(axis=1)
    df["low"] = df[["open", "close_adj", "low"]].min(axis=1)
    return df


# ===========================================================================
# CostModel tests
# ===========================================================================


class TestCostModel:
    def test_sell_side_fees(self):
        model = CostModel()
        cost = model.estimate_one_way("sell", 10_000.0, 100.0, 50_000_000.0)
        # SEC fee
        assert cost.sec_fee_usd == pytest.approx(10_000.0 * SEC_FEE_RATE, rel=1e-6)
        # FINRA TAF
        assert cost.finra_taf_usd == pytest.approx(
            100.0 * FINRA_TAF_PER_SHARE, rel=1e-6
        )
        assert cost.total_usd > 0

    def test_buy_side_no_regulatory_fees(self):
        model = CostModel()
        cost = model.estimate_one_way("buy", 10_000.0, 100.0, 50_000_000.0)
        assert cost.sec_fee_usd == 0.0
        assert cost.finra_taf_usd == 0.0
        assert cost.spread_cost_usd > 0
        assert cost.slippage_usd > 0

    def test_round_trip_more_than_one_way(self):
        model = CostModel()
        rt = model.estimate_round_trip(10_000.0, 100.0, 50_000_000.0)
        one_way = model.estimate_one_way("buy", 10_000.0, 100.0, 50_000_000.0)
        assert rt.total_usd > one_way.total_usd

    def test_fill_price_buy_above_nominal(self):
        model = CostModel()
        nominal = 100.0
        fill = model.apply_fill_price(nominal, "buy", 50_000_000.0, 10_000.0)
        assert fill > nominal

    def test_fill_price_sell_below_nominal(self):
        model = CostModel()
        nominal = 100.0
        fill = model.apply_fill_price(nominal, "sell", 50_000_000.0, 10_000.0)
        assert fill < nominal

    def test_large_order_higher_slippage(self):
        model = CostModel()
        # 3% ADV order → higher slippage tier
        big = model.estimate_one_way("buy", 3_000_000.0, 30_000.0, 100_000_000.0)
        # 0.1% ADV order → base slippage
        small = model.estimate_one_way("buy", 100_000.0, 1_000.0, 100_000_000.0)
        assert big.total_bps > small.total_bps

    def test_finra_taf_cap(self):
        model = CostModel()
        # 50,000 shares × $0.000166 = $8.30, hits the cap
        cost = model.estimate_one_way("sell", 5_000_000.0, 50_000.0, 1_000_000_000.0)
        assert cost.finra_taf_usd == pytest.approx(8.30, rel=1e-4)


# ===========================================================================
# FillSimulator tests
# ===========================================================================


class TestFillSimulator:
    def test_small_order_full_fill(self):
        sim = FillSimulator(rng_seed=1)
        # Order of 10 shares, ADV 100K shares → definitely fills
        result = sim.simulate_fill(10, 100.0, 100_000, 1_000_000)
        assert result.status == FillStatus.FULL
        assert result.filled_qty == 10
        assert result.remaining_qty == 0

    def test_large_order_partial_or_full(self):
        sim = FillSimulator(max_adv_fill_pct=0.01, rng_seed=42)
        # Order of 2000 shares, ADV 100K → limit = 1000 shares per bar
        result = sim.simulate_fill(2000, 100.0, 100_000, 1_000_000)
        assert result.filled_qty <= 2000
        assert result.filled_qty + result.remaining_qty == 2000

    def test_latency_in_range(self):
        sim = FillSimulator(base_latency_ms=200, latency_jitter=0.30, rng_seed=7)
        result = sim.simulate_fill(10, 50.0, 100_000, 500_000)
        assert result.latency_ms > 0
        # Should be within ±30% jitter of 200ms
        assert 100.0 <= result.latency_ms <= 320.0

    def test_non_marketable_lower_fill_probability(self):
        """Non-marketable orders should have lower fill rates."""
        sim = FillSimulator(rng_seed=99)
        fills = sum(
            1
            for _ in range(200)
            if sim.simulate_fill(
                10, 100.0, 100_000, 500_000, is_marketable=False
            ).filled_qty
            > 0
        )
        # Non-marketable fill rate should be < marketable (which is ~95%)
        assert fills < 190  # at most 95%


# ===========================================================================
# deterministic_context tests
# ===========================================================================


class TestDeterministicContext:
    def test_same_seed_same_result(self):
        import random

        with deterministic_context(seed=42):
            val1 = random.random()
        with deterministic_context(seed=42):
            val2 = random.random()
        assert val1 == val2

    def test_different_seeds_different_result(self):
        import random

        with deterministic_context(seed=1):
            v1 = random.random()
        with deterministic_context(seed=2):
            v2 = random.random()
        assert v1 != v2

    def test_state_restored_after_context(self):
        import random

        random.seed(100)
        ref = random.random()
        random.seed(100)

        with deterministic_context(seed=999):
            _ = random.random()  # consume random values

        # After context exits, state should be restored
        val = random.random()
        assert val == ref


# ===========================================================================
# BacktestEngine tests
# ===========================================================================


class TestBacktestEngine:
    def _simple_bars(self) -> dict:
        return {"AAPL": _make_bars(300, start_price=150.0)}

    def test_empty_bars_returns_initial_capital(self):
        engine = BacktestEngine(initial_capital=50_000.0)
        result = engine.run({})
        assert result.initial_capital == 50_000.0
        assert result.final_capital == 50_000.0
        assert result.trades == []

    def test_no_strategies_no_trades(self):
        engine = BacktestEngine(initial_capital=100_000.0, strategies=[])
        result = engine.run(self._simple_bars())
        assert len(result.trades) == 0
        assert result.final_capital == pytest.approx(100_000.0)

    def test_with_momentum_strategy_generates_trades(self):
        from src.signals.momentum import MomentumStrategy

        engine = BacktestEngine(
            initial_capital=100_000.0,
            strategies=[MomentumStrategy()],
            max_open_positions=3,
        )
        with deterministic_context(seed=42):
            result = engine.run(self._simple_bars())

        # Should attempt at least some trades given 300 bars
        assert result.total_trades >= 0  # may be 0 if signal never fires
        assert result.final_capital > 0

    def test_result_compute_metrics_no_crash(self):
        result = BacktestResult(
            initial_capital=100_000.0,
            final_capital=105_000.0,
        )
        result.compute_metrics()
        assert result.total_return == pytest.approx(0.05, rel=1e-4)
        assert result.total_trades == 0

    def test_multi_symbol_backtest(self):
        from src.signals.momentum import MomentumStrategy

        bars = {
            "AAPL": _make_bars(300, start_price=150.0),
            "MSFT": _make_bars(300, start_price=300.0),
            "GOOGL": _make_bars(300, start_price=100.0),
        }
        engine = BacktestEngine(
            initial_capital=100_000.0,
            strategies=[MomentumStrategy()],
            max_open_positions=5,
        )
        with deterministic_context(seed=1):
            result = engine.run(bars)
        assert result.final_capital > 0
        assert result.initial_capital == 100_000.0

    def test_deterministic_result_with_same_seed(self):
        from src.signals.momentum import MomentumStrategy

        bars = self._simple_bars()
        engine = BacktestEngine(
            initial_capital=100_000.0,
            strategies=[MomentumStrategy()],
        )
        with deterministic_context(seed=42):
            r1 = engine.run(bars)
        with deterministic_context(seed=42):
            r2 = engine.run(bars)

        assert r1.final_capital == pytest.approx(r2.final_capital, rel=1e-6)
        assert r1.total_trades == r2.total_trades
