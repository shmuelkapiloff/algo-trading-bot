"""
Trend Following Strategy — Phase 2 (Long-Only, EOD).

Entry logic
-----------
All conditions must be true at EOD for a BUY signal:
  1. Golden Cross active: EMA50 > EMA200  (long-term uptrend filter)
  2. ADX(14) > 25  (trend strength — avoids whipsaw in sideways markets)
  3. Price above EMA50  (confirmation: we are in the trend, not below it)
  4. Market regime is "bull" only (conservative — trend strategies underperform in ranging)

Exit / Stop
-----------
  Stop: ATR(14) × 1.5 below entry (volatility-adaptive)
  Minimum stop: 2.5% of entry price (floor, in case ATR is tiny)
  Maximum stop: 6% of entry price (cap, to limit catastrophic loss)

Confidence score
----------------
  0.0–0.40 : ADX strength above 25 (higher ADX = stronger trend)
  0.0–0.35 : EMA50/EMA200 gap pct (wider gap = clearer trend)
  0.0–0.25 : Price distance above EMA50 (but not too far = not overextended)

References to plan §6 "Strategy Scan Level 3":
    → Trend Following: Golden Cross + ADX > 25
"""

from __future__ import annotations

import logging
from typing import Iterator

import pandas as pd
import ta

from .base_strategy import BaseStrategy
from .models import OrderSide, SignalIntent
from ..analysis.market_regime import MarketRegime

logger = logging.getLogger(__name__)

_MIN_BARS = 210  # need 200 bars to compute EMA200 reliably


class TrendFollowingStrategy(BaseStrategy):
    """
    Golden Cross + ADX trend-following strategy.

    Parameters (all tunable via config/strategies.yaml)
    ---------------------------------------------------
    ema_fast_period   : Fast EMA period for Golden Cross (default 50)
    ema_slow_period   : Slow EMA period for Golden Cross (default 200)
    adx_period        : ADX lookback period (default 14)
    adx_threshold     : Minimum ADX for trend strength gate (default 25)
    stop_atr_mult     : ATR multiplier for stop distance (default 1.5)
    stop_min_pct      : Minimum stop-loss distance (default 0.025 = 2.5%)
    stop_max_pct      : Maximum stop-loss distance (default 0.06 = 6%)
    ttl_seconds       : Signal TTL (default 5 days = 5 × 86400)
    min_confidence    : Drop signals below this score (default 0.55)
    min_price         : Reject penny stocks below this price (default 5.0)
    allowed_regimes   : Defaults to BULL only
    """

    def __init__(
        self,
        ema_fast_period: int = 50,
        ema_slow_period: int = 200,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        stop_atr_mult: float = 1.5,
        stop_min_pct: float = 0.025,
        stop_max_pct: float = 0.060,
        ttl_seconds: int = 5 * 86_400,
        min_confidence: float = 0.55,
        min_price: float = 5.0,
        allowed_regimes: set[MarketRegime] | None = None,
    ) -> None:
        super().__init__(
            name="trend_following",
            allowed_regimes=allowed_regimes or {MarketRegime.BULL},
            min_confidence=min_confidence,
            min_price=min_price,
        )
        self.ema_fast_period = ema_fast_period
        self.ema_slow_period = ema_slow_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.stop_atr_mult = stop_atr_mult
        self.stop_min_pct = stop_min_pct
        self.stop_max_pct = stop_max_pct
        self.ttl_seconds = ttl_seconds

    # ------------------------------------------------------------------
    # Strategy implementation
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        bars: dict[str, pd.DataFrame],
        regime: MarketRegime,
    ) -> Iterator[SignalIntent]:
        if not self._regime_allowed(regime):
            return

        for symbol, df in bars.items():
            signal = self._evaluate(symbol, df)
            if signal is not None:
                yield signal

    def _evaluate(self, symbol: str, df: pd.DataFrame) -> SignalIntent | None:
        if not self._need_bars(df, _MIN_BARS, symbol):
            return None

        close = df["close_adj"]
        high = df["high"]
        low = df["low"]

        # ── Indicators ────────────────────────────────────────────────
        ema_fast = ta.trend.EMAIndicator(
            close=close, window=self.ema_fast_period
        ).ema_indicator()
        ema_slow = ta.trend.EMAIndicator(
            close=close, window=self.ema_slow_period
        ).ema_indicator()
        adx_obj = ta.trend.ADXIndicator(
            high=high, low=low, close=close, window=self.adx_period
        )
        atr14 = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range()

        ema_fast_now = ema_fast.iloc[-1]
        ema_slow_now = ema_slow.iloc[-1]
        adx_now = adx_obj.adx().iloc[-1]
        price = close.iloc[-1]
        atr = atr14.iloc[-1]

        # ── Entry conditions ──────────────────────────────────────────
        # 1. Golden Cross: EMA50 > EMA200
        golden_cross = ema_fast_now > ema_slow_now
        if not golden_cross:
            return None

        # 2. ADX > threshold (trend must be strong)
        strong_trend = adx_now >= self.adx_threshold
        if not strong_trend:
            return None

        # 3. Price above EMA50 (we are riding the trend, not a laggard)
        above_ema_fast = price > ema_fast_now
        if not above_ema_fast:
            return None

        # 4. Sanity: price above minimum
        if price < self.min_price:
            return None

        # ── Confidence score ──────────────────────────────────────────
        # Component 1: ADX strength above threshold (0.0–0.40)
        adx_excess = max(0.0, adx_now - self.adx_threshold)
        # ADX > 25: 0 pts;  ADX >= 50: full 0.40
        adx_component = min(adx_excess / 25.0, 1.0) * 0.40

        # Component 2: EMA gap pct (EMA50 above EMA200) — (0.0–0.35)
        ema_gap_pct = (ema_fast_now - ema_slow_now) / max(ema_slow_now, 1e-6)
        # 0% gap = 0 pts; 5% gap = full 0.35
        ema_component = min(ema_gap_pct / 0.05, 1.0) * 0.35

        # Component 3: Price vs EMA50 distance (0.0–0.25)
        # Reward being above trend, penalise being too extended (> 10%)
        price_gap_pct = (price - ema_fast_now) / max(ema_fast_now, 1e-6)
        # Ideal: 1–5% above EMA50.  Too extended (>10%) = lower confidence.
        if price_gap_pct <= 0.0:
            price_component = 0.0
        elif price_gap_pct <= 0.05:
            price_component = (price_gap_pct / 0.05) * 0.25
        else:
            # Fade as price extends beyond 5%
            overextension = min((price_gap_pct - 0.05) / 0.05, 1.0)
            price_component = max(0.0, 0.25 * (1.0 - overextension))

        confidence = adx_component + ema_component + price_component

        # ── Viability check ───────────────────────────────────────────
        if not self._check_viability(symbol, confidence, price):
            return None

        # ── Stop distance (ATR-based, with floor and cap) ────────────
        atr_safe = max(atr, 1e-6)
        atr_stop_pct = (atr_safe * self.stop_atr_mult) / price
        stop_pct = max(atr_stop_pct, self.stop_min_pct)
        stop_pct = min(stop_pct, self.stop_max_pct)

        logger.debug(
            "[trend_following] Signal: %s  confidence=%.3f  price=%.2f  "
            "adx=%.1f  ema_gap=%.3f  stop_pct=%.3f",
            symbol,
            confidence,
            price,
            adx_now,
            ema_gap_pct,
            stop_pct,
        )

        return SignalIntent(
            symbol=symbol,
            side=OrderSide.BUY,
            strategy_name=self.name,
            confidence=confidence,
            stop_distance_pct=stop_pct,
            ttl_seconds=self.ttl_seconds,
            reason=f"GoldenCross EMA{self.ema_fast_period}>{self.ema_slow_period}, ADX={adx_now:.1f}",
        )
