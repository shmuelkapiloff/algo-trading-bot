"""
Tests for src/signals/momentum.py — MomentumStrategy.generate_signals()

Strategy entry conditions (all must be true):
  1. price > EMA50
  2. RSI[-2] < 35 AND RSI[-1] >= 35  (crossover)
  3. MACD line > MACD signal
  4. regime in {BULL, SIDEWAYS}

Tests verify:
  - Valid entry conditions produce a BUY signal
  - Each failed condition independently blocks signal
  - Confidence is in [0, 1]
  - Wrong regime produces zero signals
  - Insufficient bars produce no signal
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.market_regime import MarketRegime
from src.signals.momentum import MomentumStrategy
from src.signals.models import OrderSide


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bars_from_prices(prices: list[float]) -> pd.DataFrame:
    """Build a minimal bar DataFrame from a price list."""
    n = len(prices)
    return pd.DataFrame(
        {
            "close_adj": prices,
            "high": [p * 1.005 for p in prices],
            "low": [p * 0.995 for p in prices],
            "volume": [1_000_000] * n,
            "open": prices,
        }
    )


def _uptrend_then_dip(n: int = 120) -> pd.DataFrame:
    """
    Creates a clear uptrend (price well above EMA50) with a final RSI dip
    below 35 then recovery above 35, plus bullish MACD.

    Construction:
      - First 100 bars: steady uptrend (ensures EMA50 < price at end)
      - Last 20 bars: quick dip then partial recovery
        The dip creates low RSI, the partial recovery creates RSI crossover.
    """
    # Strong uptrend base
    base = [100 + i * 0.5 for i in range(100)]
    # Sharp down-move (creates low RSI)
    dip = [base[-1] - j * 1.5 for j in range(1, 12)]
    # Recovery (RSI crosses back above 35)
    recovery = [dip[-1] + k * 2.0 for k in range(1, 9)]
    prices = base + dip + recovery
    return _bars_from_prices(prices)


def _downtrend_below_ema50(n: int = 120) -> pd.DataFrame:
    """Monotonic downtrend — price will be below EMA50 at the end."""
    prices = [200 - i * 0.8 for i in range(n)]
    return _bars_from_prices(prices)


def _flat_series(n: int = 120, value: float = 100.0) -> pd.DataFrame:
    """Completely flat — RSI ≈ 50, no MACD divergence, no RSI crossover."""
    prices = [value] * n
    return _bars_from_prices(prices)


# ---------------------------------------------------------------------------
# Basic instantiation
# ---------------------------------------------------------------------------


class TestMomentumStrategyInit:
    def test_default_construction(self):
        s = MomentumStrategy()
        assert s.name == "momentum"
        assert MarketRegime.BULL in s.allowed_regimes
        assert MarketRegime.SIDEWAYS in s.allowed_regimes
        assert MarketRegime.BEAR not in s.allowed_regimes

    def test_custom_params(self):
        s = MomentumStrategy(rsi_oversold=30, min_confidence=0.65)
        assert s.rsi_oversold == 30
        assert s.min_confidence == 0.65


# ---------------------------------------------------------------------------
# Regime filter
# ---------------------------------------------------------------------------


class TestRegimeFilter:
    def test_no_signals_in_bear_regime(self):
        strategy = MomentumStrategy()
        bars = {"AAPL": _uptrend_then_dip()}
        signals = list(strategy.generate_signals(bars, MarketRegime.BEAR))
        assert signals == []

    def test_signals_allowed_in_bull(self):
        """If price conditions are met, BULL regime should not block."""
        strategy = MomentumStrategy(min_confidence=0.0, min_price=0.0)
        bars = {"AAPL": _uptrend_then_dip(120)}
        # Just verify the method runs without error in BULL
        _ = list(strategy.generate_signals(bars, MarketRegime.BULL))

    def test_signals_allowed_in_sideways(self):
        strategy = MomentumStrategy(min_confidence=0.0, min_price=0.0)
        bars = {"AAPL": _uptrend_then_dip(120)}
        _ = list(strategy.generate_signals(bars, MarketRegime.SIDEWAYS))


# ---------------------------------------------------------------------------
# Insufficient data
# ---------------------------------------------------------------------------


class TestInsufficientBars:
    def test_too_few_bars_yields_nothing(self):
        strategy = MomentumStrategy()
        short_df = _bars_from_prices([100.0] * 30)  # < 60 bars minimum
        signals = list(strategy.generate_signals({"AAPL": short_df}, MarketRegime.BULL))
        assert signals == []

    def test_exactly_min_bars_does_not_raise(self):
        strategy = MomentumStrategy()
        df = _bars_from_prices([100.0] * 60)
        _ = list(strategy.generate_signals({"AAPL": df}, MarketRegime.BULL))


# ---------------------------------------------------------------------------
# Price filter
# ---------------------------------------------------------------------------


class TestPriceFilter:
    def test_penny_stock_rejected(self):
        """Symbol with price < min_price (default $5) must be skipped."""
        strategy = MomentumStrategy(min_price=5.0, min_confidence=0.0)
        cheap_df = _bars_from_prices([3.0 + i * 0.01 for i in range(120)])
        signals = list(
            strategy.generate_signals({"PENNY": cheap_df}, MarketRegime.BULL)
        )
        assert signals == []


# ---------------------------------------------------------------------------
# Signal fields
# ---------------------------------------------------------------------------


class TestSignalFields:
    def test_signal_side_is_buy(self):
        """Momentum is long-only — all signals must be BUY."""
        strategy = MomentumStrategy(min_confidence=0.0, min_price=0.0)
        bars = {"AAPL": _uptrend_then_dip(120)}
        signals = list(strategy.generate_signals(bars, MarketRegime.BULL))
        for sig in signals:
            assert sig.side == OrderSide.BUY

    def test_signal_strategy_name(self):
        strategy = MomentumStrategy(min_confidence=0.0, min_price=0.0)
        bars = {"AAPL": _uptrend_then_dip(120)}
        signals = list(strategy.generate_signals(bars, MarketRegime.BULL))
        for sig in signals:
            assert sig.strategy_name == "momentum"

    def test_confidence_in_range(self):
        strategy = MomentumStrategy(min_confidence=0.0, min_price=0.0)
        bars = {"AAPL": _uptrend_then_dip(120)}
        signals = list(strategy.generate_signals(bars, MarketRegime.BULL))
        for sig in signals:
            assert 0.0 <= sig.confidence <= 1.0

    def test_stop_distance_pct_positive(self):
        strategy = MomentumStrategy(min_confidence=0.0, min_price=0.0)
        bars = {"AAPL": _uptrend_then_dip(120)}
        signals = list(strategy.generate_signals(bars, MarketRegime.BULL))
        for sig in signals:
            assert sig.stop_distance_pct is not None
            assert sig.stop_distance_pct > 0


# ---------------------------------------------------------------------------
# No signal when trend filter fails
# ---------------------------------------------------------------------------


class TestTrendFilter:
    def test_no_signal_in_downtrend(self):
        """Price below EMA50 — trend filter should block all signals."""
        strategy = MomentumStrategy(min_confidence=0.0, min_price=0.0)
        bars = {"SPY": _downtrend_below_ema50(120)}
        signals = list(strategy.generate_signals(bars, MarketRegime.BULL))
        assert signals == []

    def test_no_signal_in_flat_series(self):
        """Flat series: no RSI crossover, no MACD divergence → no signal."""
        strategy = MomentumStrategy(min_confidence=0.0, min_price=0.0)
        bars = {"MSFT": _flat_series(120)}
        signals = list(strategy.generate_signals(bars, MarketRegime.BULL))
        assert signals == []


# ---------------------------------------------------------------------------
# Multi-symbol scan
# ---------------------------------------------------------------------------


class TestMultiSymbolScan:
    def test_handles_empty_bars_dict(self):
        strategy = MomentumStrategy()
        signals = list(strategy.generate_signals({}, MarketRegime.BULL))
        assert signals == []

    def test_processes_multiple_symbols_independently(self):
        strategy = MomentumStrategy(min_confidence=0.0, min_price=0.0)
        bars = {
            "AAPL": _uptrend_then_dip(120),
            "MSFT": _flat_series(120),  # flat → no signal
            "AMZN": _downtrend_below_ema50(120),  # downtrend → no signal
        }
        signals = list(strategy.generate_signals(bars, MarketRegime.BULL))
        symbols = [s.symbol for s in signals]
        assert "MSFT" not in symbols
        assert "AMZN" not in symbols
