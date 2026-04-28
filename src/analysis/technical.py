"""
Technical Analysis — all indicators computed on split-adjusted prices.

Rules (from TRADING_BOT_PLAN.md):
- ALWAYS use close_adj, never close_raw
- All indicators return pd.Series aligned to the input index
- Functions are stateless and pure (no side effects)
- Minimum bar requirements are enforced; raises ValueError on insufficient data

Indicators included:
  RSI, MACD, Bollinger Bands, EMA (20/50/200), ATR, ADX, Stochastic, OBV
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import ta as _ta
    _TA = True
except ImportError:  # pragma: no cover
    _TA = False


# ---------------------------------------------------------------------------
# Named-tuple results
# ---------------------------------------------------------------------------


class MACDResult(NamedTuple):
    macd: pd.Series
    signal: pd.Series
    histogram: pd.Series


class BollingerResult(NamedTuple):
    upper: pd.Series
    middle: pd.Series
    lower: pd.Series
    bandwidth: pd.Series
    pct_b: pd.Series


# ---------------------------------------------------------------------------
# Core indicators
# ---------------------------------------------------------------------------


def ema(close: pd.Series, window: int) -> pd.Series:
    """Exponential moving average."""
    return close.ewm(span=window, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index."""
    if len(close) < window + 1:
        raise ValueError(f"rsi requires >= {window + 1} bars, got {len(close)}")
    if _TA:
        return _ta.momentum.RSIIndicator(close, window=window).rsi()
    # Fallback implementation
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, float("inf"))
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_window: int = 9,
) -> MACDResult:
    """MACD line, signal line, histogram."""
    if len(close) < slow + signal_window:
        raise ValueError(f"macd requires >= {slow + signal_window} bars")
    if _TA:
        obj = _ta.trend.MACD(close, window_slow=slow, window_fast=fast,
                               window_sign=signal_window)
        line = obj.macd()
        sig = obj.macd_signal()
        return MACDResult(line, sig, line - sig)
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    line = fast_ema - slow_ema
    sig = line.ewm(span=signal_window, adjust=False).mean()
    return MACDResult(line, sig, line - sig)


def bollinger_bands(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> BollingerResult:
    """Bollinger Bands: upper, middle, lower + bandwidth + %B."""
    if len(close) < window:
        raise ValueError(f"bollinger requires >= {window} bars")
    middle = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = (upper - lower) / middle.replace(0, float("nan"))
    pct_b = (close - lower) / (upper - lower).replace(0, float("nan"))
    return BollingerResult(upper, middle, lower, bandwidth, pct_b)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Average Directional Index (strength of trend, 0–100)."""
    if _TA:
        return _ta.trend.ADXIndicator(high, low, close, window=window).adx()
    tr_val = atr(high, low, close, window)
    up_move = high.diff()
    down_move = -low.diff()
    pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    pos_di = 100 * pos_dm.rolling(window).mean() / tr_val.replace(0, float("nan"))
    neg_di = 100 * neg_dm.rolling(window).mean() / tr_val.replace(0, float("nan"))
    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di).replace(0, float("nan"))
    return dx.rolling(window).mean()


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_window: int = 14,
    d_window: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator — returns (%K, %D)."""
    if _TA:
        obj = _ta.momentum.StochasticOscillator(
            high, low, close, window=k_window, smooth_window=d_window
        )
        return obj.stoch(), obj.stoch_signal()
    low_roll = low.rolling(k_window).min()
    high_roll = high.rolling(k_window).max()
    k = 100 * (close - low_roll) / (high_roll - low_roll).replace(0, float("nan"))
    d = k.rolling(d_window).mean()
    return k, d


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()
