"""
Momentum Strategy — Phase 1 (Long-Only, EOD).

Entry logic
-----------
All conditions must be true at EOD for a BUY signal:
  1. Price > EMA50   (long-term uptrend filter — no buying in downtrends)
  2. RSI(14) crossed above the oversold threshold recently
     (specifically: RSI[-2] < rsi_oversold AND RSI[-1] >= rsi_oversold)
  3. MACD line > Signal line (bullish momentum confirmation)
  4. Market regime is "bull" or "sideways" (from regime_cache)

Confidence score
----------------
Confidence is scored 0.0–1.0 from the following components:
  - RSI distance from oversold:  how much room before overbought (0.0–0.40)
  - MACD histogram strength:     normalised by recent ATR (0.0–0.35)
  - Price vs EMA50 gap:          how far above the trend line (0.0–0.25)

Stop-loss
---------
Initial stop = entry_price × (1 - stop_distance_pct).
stop_distance_pct defaults to ATR(14)/price or the config floor.

Hold period
-----------
ttl_seconds = 86400 (1 trading day). The portfolio manager will exit
positions that have been open for > 1 day and no renewal signal arrives.
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

_MIN_BARS = 60  # minimum bars needed to compute EMA50 + MACD reliably


class MomentumStrategy(BaseStrategy):
    """
    RSI-crossover + MACD confirmation momentum strategy.

    Parameters (all tunable via config/strategies.yaml)
    ---------------------------------------------------
    rsi_period        : RSI lookback (default 14)
    rsi_oversold      : Entry threshold for RSI crossover (default 35)
    rsi_overbought    : Upper bound — no entry when RSI > this (default 70)
    macd_fast/slow/signal : MACD parameters
    ema_trend_period  : EMA period for trend filter (default 50)
    stop_distance_pct : Initial stop-loss distance (default 0.03 = 3%)
    ttl_seconds       : How long the signal is valid (default 86400 = 1 day)
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 70.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        ema_trend_period: int = 50,
        stop_distance_pct: float = 0.03,
        ttl_seconds: int = 86_400,
        min_confidence: float = 0.55,
        min_price: float = 5.0,
        allowed_regimes: set[MarketRegime] | None = None,
    ) -> None:
        super().__init__(
            name="momentum",
            allowed_regimes=allowed_regimes
            or {MarketRegime.BULL, MarketRegime.SIDEWAYS},
            min_confidence=min_confidence,
            min_price=min_price,
        )
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.ema_trend_period = ema_trend_period
        self.stop_distance_pct = stop_distance_pct
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

        # ── Indicators ────────────────────────────────────────────────
        rsi = ta.momentum.RSIIndicator(close=close, window=self.rsi_period).rsi()
        macd_obj = ta.trend.MACD(
            close=close,
            window_slow=self.macd_slow,
            window_fast=self.macd_fast,
            window_sign=self.macd_signal,
        )
        ema50 = ta.trend.EMAIndicator(
            close=close, window=self.ema_trend_period
        ).ema_indicator()
        atr14 = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=close, window=14
        ).average_true_range()

        rsi_now = rsi.iloc[-1]
        rsi_prev = rsi.iloc[-2]
        macd_line = macd_obj.macd().iloc[-1]
        macd_sig = macd_obj.macd_signal().iloc[-1]
        macd_hist = macd_obj.macd_diff().iloc[-1]
        price = close.iloc[-1]
        ema_now = ema50.iloc[-1]
        atr = atr14.iloc[-1]

        # ── Entry conditions ──────────────────────────────────────────
        # 1. Trend filter: price above EMA50
        if price < ema_now:
            return None

        # 2. RSI crossover: crossed above oversold on the last bar
        rsi_crossed_up = rsi_prev < self.rsi_oversold <= rsi_now

        # 3. Not overbought
        not_overbought = rsi_now < self.rsi_overbought

        # 4. MACD bullish
        macd_bullish = macd_line > macd_sig

        if not (rsi_crossed_up and not_overbought and macd_bullish):
            return None

        # ── Confidence score ──────────────────────────────────────────
        # Component 1: RSI room to overbought (0.0–0.40)
        rsi_room = max(0.0, self.rsi_overbought - rsi_now)
        rsi_component = (
            min(rsi_room / (self.rsi_overbought - self.rsi_oversold), 1.0) * 0.40
        )

        # Component 2: MACD histogram strength vs ATR (0.0–0.35)
        atr_safe = max(atr, 1e-6)
        hist_norm = min(abs(macd_hist) / (atr_safe * 0.1), 1.0)  # 10% ATR = full score
        macd_component = hist_norm * 0.35

        # Component 3: Price above EMA50 (0.0–0.25)
        ema_gap_pct = (price - ema_now) / max(ema_now, 1e-6)
        ema_component = min(ema_gap_pct / 0.05, 1.0) * 0.25  # 5% gap = full score

        confidence = rsi_component + macd_component + ema_component

        # ── Viability check ───────────────────────────────────────────
        if not self._check_viability(symbol, confidence, price):
            return None

        # ── Stop distance ─────────────────────────────────────────────
        # Use ATR-based stop if available, else config default
        atr_stop_pct = (atr / price) if price > 0 else self.stop_distance_pct
        stop_pct = max(atr_stop_pct, 0.015)  # floor at 1.5% to avoid tiny stops
        stop_pct = min(stop_pct, self.stop_distance_pct * 1.5)  # cap at 1.5× config

        logger.debug(
            "[momentum] Signal: %s  confidence=%.3f  price=%.2f  "
            "rsi=%.1f  macd_hist=%.4f  stop_pct=%.3f",
            symbol,
            confidence,
            price,
            rsi_now,
            macd_hist,
            stop_pct,
        )

        return SignalIntent(
            symbol=symbol,
            side=OrderSide.BUY,
            strategy_name=self.name,
            confidence=confidence,
            stop_distance_pct=stop_pct,
            ttl_seconds=self.ttl_seconds,
        )
