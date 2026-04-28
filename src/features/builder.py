"""
Feature Builder — computes point-in-time feature sets from OHLCV DataFrames.

Rules:
- Input DataFrame must have a DatetimeIndex and columns: open, high, low, close, volume
- All prices MUST be split-adjusted before being passed here
- Features are computed from data[:as_of_date] only (strict PIT boundary)
- Returns a FeatureSnapshot that can be stored in FeatureStore
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from .store import FeatureSnapshot, FeatureStore

logger = logging.getLogger(__name__)

try:
    import ta
    _TA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TA_AVAILABLE = False


class FeatureBuilder:
    """
    Builds a FeatureSnapshot for (symbol, as_of_date) from a price DataFrame.

    Usage::

        builder = FeatureBuilder()
        snap = builder.build(symbol="AAPL", bars=df, as_of_date=date(2024, 1, 15))
        store.put(snap)
    """

    def __init__(self, store: FeatureStore | None = None) -> None:
        self._store = store

    def build(
        self,
        symbol: str,
        bars: pd.DataFrame,
        as_of_date: date,
    ) -> FeatureSnapshot:
        """
        Compute features using only bars up to and including as_of_date.
        Stores into self._store if one was provided.
        """
        # Strict PIT slice — no future data
        pit_bars = bars[bars.index.date <= as_of_date]
        if pit_bars.empty:
            raise ValueError(f"No bars available for {symbol} up to {as_of_date}")

        features: dict = {}

        close = pit_bars["close"]
        high = pit_bars["high"]
        low = pit_bars["low"]
        volume = pit_bars["volume"]

        # --- Price features ---
        features["close"] = float(close.iloc[-1])
        features["pct_change_1d"] = float(close.pct_change(1).iloc[-1])
        features["pct_change_5d"] = float(close.pct_change(5).iloc[-1])
        features["pct_change_20d"] = float(close.pct_change(20).iloc[-1])

        # --- Moving averages ---
        features["ema_20"] = float(close.ewm(span=20).mean().iloc[-1])
        features["ema_50"] = float(close.ewm(span=50).mean().iloc[-1])
        features["ema_200"] = float(close.ewm(span=200).mean().iloc[-1])
        features["price_above_ema50"] = float(close.iloc[-1]) > features["ema_50"]

        # --- ATR (14) ---
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        features["atr_14"] = float(tr.rolling(14).mean().iloc[-1])
        features["atr_pct"] = features["atr_14"] / features["close"] if features["close"] else 0.0

        # --- Volume features ---
        features["volume"] = int(volume.iloc[-1])
        features["adv_20"] = float(volume.rolling(20).mean().iloc[-1])
        features["vol_ratio"] = (features["volume"] / features["adv_20"]
                                  if features["adv_20"] else 0.0)

        # --- Momentum (ta library if available) ---
        if _TA_AVAILABLE and len(pit_bars) >= 26:
            features["rsi_14"] = float(
                ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
            )
            macd_obj = ta.trend.MACD(close)
            features["macd"] = float(macd_obj.macd().iloc[-1])
            features["macd_signal"] = float(macd_obj.macd_signal().iloc[-1])
            features["macd_histogram"] = features["macd"] - features["macd_signal"]
        else:
            features["rsi_14"] = None
            features["macd"] = None
            features["macd_signal"] = None
            features["macd_histogram"] = None

        snapshot = FeatureSnapshot(symbol=symbol, as_of_date=as_of_date, features=features)
        if self._store is not None:
            self._store.put(snapshot)
        logger.debug("feature_builder.built symbol=%s date=%s hash=%s",
                     symbol, as_of_date, snapshot.content_hash)
        return snapshot
