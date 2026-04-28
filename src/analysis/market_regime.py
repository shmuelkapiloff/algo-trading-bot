"""
Market Regime Detection — Phase 1.

Classifies the current market into one of three regimes:
  bull     — trending up; momentum strategies perform well
  bear     — trending down; long entries suppressed
  sideways — mean-reverting; mean-reversion strategies preferred

Classification method (lean Phase 1)
--------------------------------------
Uses SPY (S&P 500 ETF) as the market proxy:
  1. 200-day SMA (long-term trend)
  2. 50-day SMA  (medium-term trend)
  3. RSI(14) of SPY

Rules:
  bull:     price > SMA200 AND price > SMA50 AND RSI < 75
  bear:     price < SMA200 OR (price < SMA50 AND RSI < 45)
  sideways: everything else

The regime is re-computed once per day (EOD scan trigger) and cached
in Redis with a 30-minute TTL. The cache is in regime_cache.py.
"""

from __future__ import annotations

import logging
from enum import Enum

import pandas as pd
import ta

logger = logging.getLogger(__name__)

MARKET_PROXY = "SPY"


class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


def detect_regime(spy_df: pd.DataFrame) -> MarketRegime:
    """
    Classify the current market regime from SPY daily bars.

    Parameters
    ----------
    spy_df : DataFrame with at least 200 rows and a 'close_adj' column,
             sorted ascending by date.

    Returns
    -------
    MarketRegime
    """
    if len(spy_df) < 200:
        logger.warning(
            "Insufficient bars for regime detection (%d < 200). "
            "Defaulting to SIDEWAYS (conservative).",
            len(spy_df),
        )
        return MarketRegime.SIDEWAYS

    close = spy_df["close_adj"]

    sma200 = close.rolling(200).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    rsi14 = ta.momentum.RSIIndicator(close=close, window=14).rsi().iloc[-1]
    price = close.iloc[-1]

    logger.debug(
        "Regime inputs: price=%.2f SMA50=%.2f SMA200=%.2f RSI=%.1f",
        price,
        sma50,
        sma200,
        rsi14,
    )

    if price > sma200 and price > sma50 and rsi14 < 75:
        regime = MarketRegime.BULL
    elif price < sma200 or (price < sma50 and rsi14 < 45):
        regime = MarketRegime.BEAR
    else:
        regime = MarketRegime.SIDEWAYS

    logger.info("Market regime detected: %s", regime.value)
    return regime
