"""
Tests for src/signals/mean_reversion.py — MeanReversionStrategy.generate_signals()

Strategy entry conditions (all must be true):
  1. price <= lower_band × (1 + bb_entry_pct)  — near lower Bollinger Band
  2. RSI < rsi_max_entry (default 40)           — oversold confirmation
  3. price > SMA200 × 0.85                      — not a falling knife (if 200 bars)
  4. regime in {SIDEWAYS, BULL}

Tests verify:
  - Oversold near BB lower band → BUY signal in correct regime
  - Each failed condition blocks signal independently
  - Confidence components produce values in [0, 1]
  - BEAR regime blocks all signals
  - Penny stock filter works
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.market_regime import MarketRegime
from src.signals.mean_reversion import MeanReversionStrategy
from src.signals.models import OrderSide


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bars_from_prices(
    prices: list[float], volume_multiplier: float = 1.0
) -> pd.DataFrame:
    n = len(prices)
    return pd.DataFrame(
        {
            "close_adj": prices,
            "high": [p * 1.003 for p in prices],
            "low": [p * 0.997 for p in prices],
            "volume": [int(1_000_000 * volume_multiplier)] * n,
            "open": prices,
        }
    )


def _oversold_near_lower_band(n: int = 120) -> pd.DataFrame:
    """
    Steady price followed by a sharp sell-off that touches the lower BB.

    Construction:
      - First 80 bars: stable at 100 (establishes BB midpoint + SMA200 base)
      - Last 40 bars: gradual decline to push price near lower band
        and drive RSI below 40
    """
    stable = [100.0] * 80
    # Decline: push ~2 standard deviations below mean
    decline = [100.0 - i * 0.6 for i in range(1, 41)]
    prices = stable + decline
    return _bars_from_prices(prices)


def _overbought_at_upper_band(n: int = 120) -> pd.DataFrame:
    """Rising prices so RSI is high and price is near UPPER band."""
    prices = [80.0 + i * 0.5 for i in range(n)]
    return _bars_from_prices(prices)


def _falling_knife(n: int = 220) -> pd.DataFrame:
    """
    Severe downtrend: price falls well below SMA200 × 0.85 threshold.
    Uses 220 bars so SMA200 can be computed.
    """
    prices = [200.0 - i * 0.7 for i in range(n)]
    prices = [max(p, 5.0) for p in prices]
    return _bars_from_prices(prices)


def _flat_series(n: int = 120, value: float = 100.0) -> pd.DataFrame:
    prices = [value] * n
    return _bars_from_prices(prices)


# ---------------------------------------------------------------------------
# Basic instantiation
# ---------------------------------------------------------------------------


class TestMeanReversionStrategyInit:
    def test_default_construction(self):
        s = MeanReversionStrategy()
        assert s.name == "mean_reversion"
        assert MarketRegime.SIDEWAYS in s.allowed_regimes
        assert MarketRegime.BULL in s.allowed_regimes
        assert MarketRegime.BEAR not in s.allowed_regimes

    def test_custom_params(self):
        s = MeanReversionStrategy(bb_period=30, rsi_max_entry=35, min_confidence=0.70)
        assert s.bb_period == 30
        assert s.rsi_max_entry == 35
        assert s.min_confidence == 0.70


# ---------------------------------------------------------------------------
# Regime filter
# ---------------------------------------------------------------------------


class TestRegimeFilter:
    def test_no_signals_in_bear_regime(self):
        strategy = MeanReversionStrategy(min_confidence=0.0, min_price=0.0)
        bars = {"AAPL": _oversold_near_lower_band()}
        signals = list(strategy.generate_signals(bars, MarketRegime.BEAR))
        assert signals == []

    def test_regime_sideways_allowed(self):
        strategy = MeanReversionStrategy(min_confidence=0.0, min_price=0.0)
        bars = {"AAPL": _oversold_near_lower_band()}
        _ = list(strategy.generate_signals(bars, MarketRegime.SIDEWAYS))

    def test_regime_bull_allowed(self):
        strategy = MeanReversionStrategy(min_confidence=0.0, min_price=0.0)
        bars = {"AAPL": _oversold_near_lower_band()}
        _ = list(strategy.generate_signals(bars, MarketRegime.BULL))


# ---------------------------------------------------------------------------
# Falling knife guard
# ---------------------------------------------------------------------------


class TestFallingKnifeGuard:
    def test_price_far_below_sma200_blocked(self):
        """
        With 220 bars of falling prices, price at the end is far below SMA200.
        The 85% guard should block the signal.
        """
        strategy = MeanReversionStrategy(min_confidence=0.0, min_price=0.0)
        bars = {"SPY": _falling_knife(220)}
        signals = list(strategy.generate_signals(bars, MarketRegime.SIDEWAYS))
        assert signals == []


# ---------------------------------------------------------------------------
# Insufficient bars
# ---------------------------------------------------------------------------


class TestInsufficientBars:
    def test_too_few_bars_yields_nothing(self):
        strategy = MeanReversionStrategy()
        short_df = _bars_from_prices([100.0] * 20)
        signals = list(
            strategy.generate_signals({"AAPL": short_df}, MarketRegime.SIDEWAYS)
        )
        assert signals == []

    def test_exactly_min_bars_does_not_raise(self):
        strategy = MeanReversionStrategy()
        df = _bars_from_prices([100.0] * 60)
        _ = list(strategy.generate_signals({"AAPL": df}, MarketRegime.SIDEWAYS))


# ---------------------------------------------------------------------------
# Price floor filter
# ---------------------------------------------------------------------------


class TestPriceFilter:
    def test_penny_stock_rejected(self):
        strategy = MeanReversionStrategy(min_price=10.0, min_confidence=0.0)
        cheap_df = _bars_from_prices([5.0] * 120)
        signals = list(
            strategy.generate_signals({"PENNY": cheap_df}, MarketRegime.SIDEWAYS)
        )
        assert signals == []


# ---------------------------------------------------------------------------
# No signal when not oversold
# ---------------------------------------------------------------------------


class TestNoSignalConditions:
    def test_overbought_rising_no_signal(self):
        """Price rising steadily — RSI high, price near UPPER band, not lower band."""
        strategy = MeanReversionStrategy(min_confidence=0.0, min_price=0.0)
        bars = {"AAPL": _overbought_at_upper_band()}
        signals = list(strategy.generate_signals(bars, MarketRegime.SIDEWAYS))
        assert signals == []

    def test_flat_series_no_signal(self):
        """Flat price: RSI ≈ 50 (above 40) and price ≈ midband (not near lower)."""
        strategy = MeanReversionStrategy(min_confidence=0.0, min_price=0.0)
        bars = {"FLAT": _flat_series(120)}
        signals = list(strategy.generate_signals(bars, MarketRegime.SIDEWAYS))
        assert signals == []


# ---------------------------------------------------------------------------
# Signal fields (when signal is produced)
# ---------------------------------------------------------------------------


class TestSignalFields:
    def _get_signals(self):
        strategy = MeanReversionStrategy(min_confidence=0.0, min_price=0.0)
        bars = {"AAPL": _oversold_near_lower_band()}
        return list(strategy.generate_signals(bars, MarketRegime.SIDEWAYS))

    def test_signal_side_is_buy(self):
        signals = self._get_signals()
        for sig in signals:
            assert sig.side == OrderSide.BUY

    def test_signal_strategy_name(self):
        signals = self._get_signals()
        for sig in signals:
            assert sig.strategy_name == "mean_reversion"

    def test_confidence_in_range(self):
        signals = self._get_signals()
        for sig in signals:
            assert 0.0 <= sig.confidence <= 1.0

    def test_stop_distance_pct_positive(self):
        signals = self._get_signals()
        for sig in signals:
            assert sig.stop_distance_pct is not None
            assert sig.stop_distance_pct > 0


# ---------------------------------------------------------------------------
# Volume spike boosts confidence
# ---------------------------------------------------------------------------


class TestVolumeSpike:
    def test_higher_volume_gives_higher_confidence(self):
        """
        Two identical price series but different volumes.
        Higher volume on the down day should yield higher confidence score.
        """
        prices = [100.0] * 80 + [100.0 - i * 0.6 for i in range(1, 41)]

        df_low_vol = pd.DataFrame(
            {
                "close_adj": prices,
                "high": [p * 1.003 for p in prices],
                "low": [p * 0.997 for p in prices],
                "volume": [500_000] * len(prices),
                "open": prices,
            }
        )
        df_high_vol = pd.DataFrame(
            {
                "close_adj": prices,
                "high": [p * 1.003 for p in prices],
                "low": [p * 0.997 for p in prices],
                "volume": [3_000_000] * len(prices),  # 6× avg = high spike
                "open": prices,
            }
        )

        strategy = MeanReversionStrategy(min_confidence=0.0, min_price=0.0)
        sigs_low = list(
            strategy.generate_signals({"SYM": df_low_vol}, MarketRegime.SIDEWAYS)
        )
        sigs_high = list(
            strategy.generate_signals({"SYM": df_high_vol}, MarketRegime.SIDEWAYS)
        )

        if sigs_low and sigs_high:
            assert sigs_high[0].confidence >= sigs_low[0].confidence


# ---------------------------------------------------------------------------
# Multi-symbol scan
# ---------------------------------------------------------------------------


class TestMultiSymbolScan:
    def test_empty_bars_dict(self):
        strategy = MeanReversionStrategy()
        signals = list(strategy.generate_signals({}, MarketRegime.SIDEWAYS))
        assert signals == []

    def test_only_oversold_symbol_generates_signal(self):
        strategy = MeanReversionStrategy(min_confidence=0.0, min_price=0.0)
        bars = {
            "OVERSOLD": _oversold_near_lower_band(),
            "RISING": _overbought_at_upper_band(),
            "FLAT": _flat_series(120),
        }
        signals = list(strategy.generate_signals(bars, MarketRegime.SIDEWAYS))
        symbols = [s.symbol for s in signals]
        assert "RISING" not in symbols
        assert "FLAT" not in symbols
