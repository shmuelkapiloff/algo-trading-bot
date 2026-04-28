"""
Tests for src/analysis/market_regime.py — detect_regime()

All tests use synthetic DataFrames with controlled SMA/RSI conditions
to verify each classification branch independently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.market_regime import MarketRegime, detect_regime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n: int, prices: list[float] | None = None) -> pd.DataFrame:
    """
    Build a minimal DataFrame accepted by detect_regime().
    If prices is None, a flat uptrend is generated.
    """
    if prices is None:
        prices = [100.0 + i * 0.1 for i in range(n)]
    assert len(prices) == n
    return pd.DataFrame({"close_adj": prices})


def _make_sim_prices(
    n: int, drift: float, vol: float, seed: int, start: float = 100.0
) -> pd.Series:
    """Generate GBM-style daily prices with given drift and volatility."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=drift, scale=vol, size=n - 1)
    prices = [start]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return pd.Series(prices)


def _trending_up(
    n: int = 250, start: float = 100.0, step: float = 0.001
) -> pd.DataFrame:
    """Bull market: upward drift with realistic noise so RSI stays in 50-70 range."""
    close = _make_sim_prices(n, drift=0.001, vol=0.008, seed=0, start=start)
    return pd.DataFrame({"close_adj": close.tolist()})


def _trending_down(
    n: int = 250, start: float = 100.0, step: float = 0.001
) -> pd.DataFrame:
    """Bear market: downward drift, price ends below SMA200."""
    close = _make_sim_prices(n, drift=-0.001, vol=0.008, seed=0, start=start)
    return pd.DataFrame({"close_adj": close.tolist()})


def _flat(n: int = 250, value: float = 100.0) -> pd.DataFrame:
    """Completely flat — price == SMA200 == SMA50, RSI undefined (sideways)."""
    prices = [value] * n
    return pd.DataFrame({"close_adj": prices})


# ---------------------------------------------------------------------------
# Insufficient data → default to SIDEWAYS
# ---------------------------------------------------------------------------


class TestInsufficientData:
    def test_empty_df_returns_sideways(self):
        df = _make_df(0, [])
        assert detect_regime(df) == MarketRegime.SIDEWAYS

    def test_199_bars_returns_sideways(self):
        df = _make_df(199)
        assert detect_regime(df) == MarketRegime.SIDEWAYS

    def test_exactly_200_bars_does_not_raise(self):
        df = _make_df(200)
        result = detect_regime(df)
        assert isinstance(result, MarketRegime)


# ---------------------------------------------------------------------------
# BULL regime
# ---------------------------------------------------------------------------


class TestBullRegime:
    def test_steady_uptrend_is_bull(self):
        df = _trending_up(n=250, start=100.0)
        result = detect_regime(df)
        assert result == MarketRegime.BULL

    def test_price_above_both_smas_and_rsi_below_75(self):
        """Explicitly construct a series where price > SMA50 > SMA200 and RSI < 75."""
        df = _trending_up(n=250, start=80.0)
        result = detect_regime(df)
        assert result == MarketRegime.BULL


# ---------------------------------------------------------------------------
# BEAR regime
# ---------------------------------------------------------------------------


class TestBearRegime:
    def test_steady_downtrend_is_bear(self):
        df = _trending_down(n=250, start=100.0)
        result = detect_regime(df)
        assert result == MarketRegime.BEAR

    def test_price_below_sma200_is_bear(self):
        """Any series where price < SMA200 at the end should be BEAR."""
        df = _trending_down(n=250, start=200.0)
        result = detect_regime(df)
        assert result == MarketRegime.BEAR


# ---------------------------------------------------------------------------
# SIDEWAYS regime
# ---------------------------------------------------------------------------


class TestSidewaysRegime:
    def test_flat_series_is_sideways(self):
        df = _flat(n=250, value=100.0)
        # Flat: price == SMA200 == SMA50; RSI is undefined (no change) → defaults
        # The function should return SIDEWAYS (neither BULL nor BEAR)
        result = detect_regime(df)
        assert result == MarketRegime.SIDEWAYS

    def test_oscillating_series_is_sideways(self):
        """Zero-drift noisy series → neither BULL nor BEAR (sideways)."""
        close = _make_sim_prices(250, drift=0.0, vol=0.008, seed=3)
        df = pd.DataFrame({"close_adj": close.tolist()})
        result = detect_regime(df)
        assert result == MarketRegime.SIDEWAYS


# ---------------------------------------------------------------------------
# Return type is always MarketRegime
# ---------------------------------------------------------------------------


class TestReturnType:
    @pytest.mark.parametrize(
        "df_factory",
        [
            lambda: _trending_up(250),
            lambda: _trending_down(250),
            lambda: _flat(250),
            lambda: _make_df(50),  # insufficient data
        ],
    )
    def test_always_returns_market_regime(self, df_factory):
        result = detect_regime(df_factory())
        assert isinstance(result, MarketRegime)
        assert result in (MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS)
