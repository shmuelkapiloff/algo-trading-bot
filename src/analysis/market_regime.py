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
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

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


# ===========================================================================
# V2 — 4-Dimensional Regime Detection (Production)
# ===========================================================================
# Fixes two known V1 failures:
#   - 2022: false BULL signals during bear-market rallies (SMA cross never
#     triggered because rally stayed below SMA200)
#   - 2020: BEAR held too long into the V-shaped recovery
#
# Four weighted dimensions (TRADING_BOT_PLAN.md §3):
#   trend 0.35 | volatility 0.30 | breadth 0.25 | liquidity 0.10
#
# When optional inputs (universe_closes, spread_bps) are absent, their weights
# are redistributed proportionally across the available dimensions.

_V2_WEIGHTS: Dict[str, float] = {
    "trend": 0.35,
    "volatility": 0.30,
    "breadth": 0.25,
    "liquidity": 0.10,
}
_BULL_THRESHOLD: float = 0.60
_BEAR_THRESHOLD: float = 0.40
# Volatility (annualised realised vol)
_LOW_VOL_ANN: float = 0.16    # ≤ 16% ann → vol_score 1.0 (calm)
_HIGH_VOL_ANN: float = 0.28   # ≥ 28% ann → vol_score 0.0 (stress)
# Breadth (% universe above EMA50)
_STRONG_BREADTH: float = 0.65
_WEAK_BREADTH: float = 0.40
# Liquidity (bid-ask spread bps)
_LIQ_NORMAL_BPS: float = 8.0
_LIQ_STRESS_BPS: float = 18.0
_LIQ_WIDE_FRACTION: float = 0.25


@dataclass
class RegimeScoreV2:
    """Decomposed 4-dimensional regime score returned by detect_regime_v2()."""

    trend_score: float               # 0.0 bearish → 1.0 bullish
    volatility_score: float          # 0.0 high-vol → 1.0 low-vol
    breadth_score: Optional[float]   # None when universe_closes not provided
    liquidity_score: Optional[float] # None when spread_bps not provided
    composite_score: float           # weighted average of available dimensions
    regime: MarketRegime
    n_breadth_symbols: int = 0

    def summary(self) -> str:
        parts = [
            f"trend={self.trend_score:.2f}",
            f"vol={self.volatility_score:.2f}",
        ]
        if self.breadth_score is not None:
            parts.append(f"breadth={self.breadth_score:.2f}({self.n_breadth_symbols})")
        if self.liquidity_score is not None:
            parts.append(f"liq={self.liquidity_score:.2f}")
        parts.append(f"→ composite={self.composite_score:.3f} [{self.regime.value.upper()}]")
        return "RegimeV2[" + "  ".join(parts) + "]"


# --- Dimension calculators (private) ----------------------------------------


def _v2_trend_score(close: pd.Series) -> float:
    """EMA50 vs EMA200 gap magnitude (60%) + 20-day breakout ratio (40%)."""
    if len(close) < 200:
        return 0.5
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    ema50_val = float(ema50.iloc[-1])
    ema200_val = float(ema200.iloc[-1])
    # Map gap [-10%, +10%] → [0, 1]
    gap_pct = (ema50_val - ema200_val) / max(ema200_val, 1e-9)
    cross_score = max(0.0, min(1.0, (gap_pct + 0.10) / 0.20))
    # Breakout ratio: fraction of last 20 closes above EMA50
    breakout_ratio = float((close.values[-20:] > ema50.values[-20:]).mean())
    return 0.60 * cross_score + 0.40 * breakout_ratio


def _v2_volatility_score(close: pd.Series) -> float:
    """EWMA(λ=0.94)+20D realised vol blend. Low vol → 1.0, high vol → 0.0."""
    if len(close) < 21:
        return 0.5
    daily_ret = close.pct_change().dropna()
    ewma_var = daily_ret.ewm(alpha=1 - 0.94, adjust=False).var()
    ewma_vol = float(ewma_var.iloc[-1] ** 0.5) * (252 ** 0.5)
    vol_20d = (
        float(daily_ret.iloc[-20:].std()) * (252 ** 0.5)
        if len(daily_ret) >= 20
        else ewma_vol
    )
    realized_vol = 0.60 * ewma_vol + 0.40 * vol_20d
    if realized_vol <= _LOW_VOL_ANN:
        return 1.0
    if realized_vol >= _HIGH_VOL_ANN:
        return 0.0
    return 1.0 - (realized_vol - _LOW_VOL_ANN) / (_HIGH_VOL_ANN - _LOW_VOL_ANN)


def _v2_breadth_score(
    universe_closes: Dict[str, pd.Series],
) -> Tuple[float, int]:
    """Fraction of universe with close > EMA50. Returns (score 0–1, n_symbols)."""
    above = total = 0
    for sym_close in universe_closes.values():
        if len(sym_close) < 50:
            continue
        ema50 = sym_close.ewm(span=50, adjust=False).mean()
        if float(sym_close.iloc[-1]) > float(ema50.iloc[-1]):
            above += 1
        total += 1
    if total == 0:
        return 0.5, 0
    pct = above / total
    if pct >= _STRONG_BREADTH:
        score = 1.0
    elif pct <= _WEAK_BREADTH:
        score = 0.0
    else:
        score = (pct - _WEAK_BREADTH) / (_STRONG_BREADTH - _WEAK_BREADTH)
    return score, total


def _v2_liquidity_score(spread_bps_dict: Dict[str, float]) -> float:
    """Median spread + fraction of wide spreads. 0 = stressed, 1 = healthy."""
    if not spread_bps_dict:
        return 0.5
    spreads = list(spread_bps_dict.values())
    n = len(spreads)
    median_spread = sorted(spreads)[n // 2]
    fraction_wide = sum(1 for s in spreads if s > _LIQ_STRESS_BPS) / n
    if median_spread <= _LIQ_NORMAL_BPS:
        median_comp = 1.0
    elif median_spread >= _LIQ_STRESS_BPS:
        median_comp = 0.0
    else:
        median_comp = 1.0 - (
            (median_spread - _LIQ_NORMAL_BPS) / (_LIQ_STRESS_BPS - _LIQ_NORMAL_BPS)
        )
    fraction_comp = max(0.0, 1.0 - fraction_wide / _LIQ_WIDE_FRACTION)
    return 0.60 * median_comp + 0.40 * fraction_comp


def _apply_regime_hysteresis(
    target: MarketRegime,
    composite: float,
    prev: Optional[MarketRegime],
    days_in_current: int,
    min_days: int,
    buffer: float,
) -> MarketRegime:
    """Prevent whipsawing: require sustained evidence before switching regime."""
    if prev is None or target == prev:
        return target
    # Too few days in current regime — stay
    if days_in_current < min_days:
        return prev
    # Switching FROM BULL: need composite clearly below threshold
    if prev == MarketRegime.BULL and composite >= _BULL_THRESHOLD - buffer:
        return prev
    # Switching FROM BEAR: need composite clearly above threshold
    if prev == MarketRegime.BEAR and composite <= _BEAR_THRESHOLD + buffer:
        return prev
    # Switching FROM SIDEWAYS: need composite clearly past a threshold
    if prev == MarketRegime.SIDEWAYS:
        if _BEAR_THRESHOLD - buffer < composite < _BULL_THRESHOLD + buffer:
            return prev
    return target


# --- Public V2 entry point --------------------------------------------------


def detect_regime_v2(
    spy_df: pd.DataFrame,
    universe_closes: Optional[Dict[str, pd.Series]] = None,
    spread_bps: Optional[Dict[str, float]] = None,
    prev_regime: Optional[MarketRegime] = None,
    days_in_current_regime: int = 0,
    min_days_in_regime: int = 3,
    hysteresis_buffer: float = 0.05,
) -> Tuple[MarketRegime, RegimeScoreV2]:
    """Classify market regime using a 4-dimensional weighted composite score.

    Parameters
    ----------
    spy_df:
        DataFrame with 'close_adj' column, sorted ascending (200+ rows ideal).
    universe_closes:
        Optional {symbol: close_Series} for the breadth dimension.
        When absent, breadth weight is redistributed to trend + volatility.
    spread_bps:
        Optional {symbol: spread_in_bps} for the liquidity dimension.
        When absent, liquidity weight is redistributed to other dimensions.
    prev_regime:
        Previous regime for hysteresis logic. None = no hysteresis applied.
    days_in_current_regime:
        Trading days spent in prev_regime (for hysteresis min_days check).
    min_days_in_regime:
        Minimum days in current regime before switching is allowed.
    hysteresis_buffer:
        Score buffer past threshold required to switch away from prev_regime.

    Returns
    -------
    (MarketRegime, RegimeScoreV2)
    """
    if len(spy_df) < 200:
        logger.warning(
            "V2: Insufficient SPY bars (%d < 200). Defaulting to SIDEWAYS.",
            len(spy_df),
        )
        return MarketRegime.SIDEWAYS, RegimeScoreV2(
            trend_score=0.5,
            volatility_score=0.5,
            breadth_score=None,
            liquidity_score=None,
            composite_score=0.5,
            regime=MarketRegime.SIDEWAYS,
        )

    close = spy_df["close_adj"]

    # Compute each dimension
    t_score = _v2_trend_score(close)
    v_score = _v2_volatility_score(close)

    b_score: Optional[float] = None
    n_breadth = 0
    if universe_closes:
        b_score, n_breadth = _v2_breadth_score(universe_closes)

    l_score: Optional[float] = None
    if spread_bps:
        l_score = _v2_liquidity_score(spread_bps)

    # Weighted composite (redistribute absent-dimension weights)
    available: Dict[str, float] = {"trend": t_score, "volatility": v_score}
    if b_score is not None:
        available["breadth"] = b_score
    if l_score is not None:
        available["liquidity"] = l_score
    total_weight = sum(_V2_WEIGHTS[d] for d in available)
    composite = sum(_V2_WEIGHTS[d] * s for d, s in available.items()) / total_weight

    # Raw regime from composite
    if composite >= _BULL_THRESHOLD:
        raw_regime = MarketRegime.BULL
    elif composite <= _BEAR_THRESHOLD:
        raw_regime = MarketRegime.BEAR
    else:
        raw_regime = MarketRegime.SIDEWAYS

    # Apply hysteresis
    final_regime = _apply_regime_hysteresis(
        target=raw_regime,
        composite=composite,
        prev=prev_regime,
        days_in_current=days_in_current_regime,
        min_days=min_days_in_regime,
        buffer=hysteresis_buffer,
    )

    result = RegimeScoreV2(
        trend_score=t_score,
        volatility_score=v_score,
        breadth_score=b_score,
        liquidity_score=l_score,
        composite_score=composite,
        regime=final_regime,
        n_breadth_symbols=n_breadth,
    )
    logger.info("Market regime V2: %s", result.summary())
    return final_regime, result
