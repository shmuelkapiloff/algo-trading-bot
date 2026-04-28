"""
Mean Reversion Strategy — Phase 1 (Long-Only, EOD).

Entry logic
-----------
All conditions must be true at EOD for a BUY signal:
  1. Price touched or pierced the lower Bollinger Band (BB 20,2)
     (price <= lower_band * (1 + bb_entry_pct))
  2. RSI(14) < rsi_max_entry (depressed RSI confirms oversold condition)
  3. Price is not in a strong downtrend:
     (price > SMA200 OR price is within 15% below SMA200)
     — avoids catching falling knives in bear markets
  4. Market regime is "sideways" or "bull" (from regime_cache)

Confidence score
----------------
  - Distance from lower band to mid-band (mean reversion potential): 0.0–0.50
  - RSI depth below entry level:                                      0.0–0.30
  - Volume spike (today vs 20-day avg):                               0.0–0.20

Stop-loss
---------
stop_distance_pct from config (default 2.5%) — tighter than momentum
because mean-reversion trades have a clearly invalidating level (break of band).

Hold period
-----------
ttl_seconds = 86400 (1 day). Exit when price returns to midline (BB center)
or after TTL expires. The portfolio manager handles the exit.
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

_MIN_BARS = 60  # BB(20) + RSI(14) + SMA(200) safety buffer


class MeanReversionStrategy(BaseStrategy):
    """
    Bollinger Band lower-touch + RSI confirmation mean-reversion strategy.

    Parameters (tunable via config/strategies.yaml)
    ------------------------------------------------
    bb_period       : Bollinger Band lookback (default 20)
    bb_std          : Number of standard deviations for bands (default 2.0)
    bb_entry_pct    : Buy within this % of the lower band (default 0.02 = 2%)
    rsi_period      : RSI lookback (default 14)
    rsi_max_entry   : Only enter when RSI < this (default 40)
    stop_distance_pct : Initial stop (default 0.025 = 2.5%)
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        bb_entry_pct: float = 0.02,
        rsi_period: int = 14,
        rsi_max_entry: float = 40.0,
        stop_distance_pct: float = 0.025,
        ttl_seconds: int = 86_400,
        min_confidence: float = 0.60,
        min_price: float = 10.0,
        allowed_regimes: set[MarketRegime] | None = None,
    ) -> None:
        super().__init__(
            name="mean_reversion",
            allowed_regimes=allowed_regimes
            or {MarketRegime.SIDEWAYS, MarketRegime.BULL},
            min_confidence=min_confidence,
            min_price=min_price,
        )
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.bb_entry_pct = bb_entry_pct
        self.rsi_period = rsi_period
        self.rsi_max_entry = rsi_max_entry
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
        volume = df["volume"]

        # ── Indicators ────────────────────────────────────────────────
        bb = ta.volatility.BollingerBands(
            close=close, window=self.bb_period, window_dev=self.bb_std
        )
        rsi = ta.momentum.RSIIndicator(close=close, window=self.rsi_period).rsi()

        lower_band = bb.bollinger_lband().iloc[-1]
        mid_band = bb.bollinger_mavg().iloc[-1]
        price = close.iloc[-1]
        rsi_now = rsi.iloc[-1]

        # SMA200 for downtrend guard (need at least 200 bars, else skip)
        sma200 = close.rolling(200).mean().iloc[-1] if len(df) >= 200 else None

        # Volume: today vs 20-day avg
        vol_avg20 = volume.rolling(20).mean().iloc[-1]
        vol_today = volume.iloc[-1]
        vol_ratio = vol_today / max(vol_avg20, 1)

        # ── Entry conditions ──────────────────────────────────────────
        # 1. Price near lower Bollinger Band
        entry_threshold = lower_band * (1 + self.bb_entry_pct)
        near_lower_band = price <= entry_threshold

        # 2. RSI oversold
        rsi_oversold = rsi_now < self.rsi_max_entry

        # 3. Not a falling knife.
        # Tightened from 15% → 10% below SMA200: at -14% a stock is already in
        # structural decline and mean-reversion rarely works (2000-02, 2008-09, 2022).
        # Additionally, require SMA200 to be rising or flat (slope over 20 days > -0.5%):
        # a declining SMA200 means the trend is definitionally bearish at a multi-month
        # horizon; buying dips into a declining long-term average is the falling knife trap.
        if sma200 is not None:
            sma200_20d_ago = (
                close.rolling(200).mean().iloc[-21]
                if len(df) >= 221
                else sma200
            )
            sma200_slope_pct = (sma200 - sma200_20d_ago) / max(sma200_20d_ago, 1e-9)
            sma200_rising_or_flat = sma200_slope_pct >= -0.005  # allow ≤ -0.5% over 20 days
            not_knife = (price > sma200 * 0.90) and sma200_rising_or_flat
        else:
            not_knife = True  # no SMA200 data — pass through

        if not (near_lower_band and rsi_oversold and not_knife):
            return None

        # ── Confidence score ──────────────────────────────────────────
        # Component 1: Mean-reversion potential (distance to mid-band) (0.0–0.50)
        band_width = mid_band - lower_band
        if band_width > 0:
            reversion_potential = min((mid_band - price) / band_width, 1.0)
        else:
            reversion_potential = 0.0
        reversion_component = reversion_potential * 0.50

        # Component 2: RSI depth (lower = more oversold = higher confidence) (0.0–0.30)
        rsi_depth = max(0.0, self.rsi_max_entry - rsi_now) / self.rsi_max_entry
        rsi_component = min(rsi_depth, 1.0) * 0.30

        # Component 3: Volume spike (higher volume on the down day = capitulation) (0.0–0.20)
        vol_component = min((vol_ratio - 1.0) / 2.0, 1.0) * 0.20  # 3× avg = full score
        vol_component = max(vol_component, 0.0)

        confidence = reversion_component + rsi_component + vol_component

        # ── Viability check ───────────────────────────────────────────
        if not self._check_viability(symbol, confidence, price):
            return None

        logger.debug(
            "[mean_reversion] Signal: %s  confidence=%.3f  price=%.2f  "
            "lower_band=%.2f  rsi=%.1f  vol_ratio=%.2f",
            symbol,
            confidence,
            price,
            lower_band,
            rsi_now,
            vol_ratio,
        )

        return SignalIntent(
            symbol=symbol,
            side=OrderSide.BUY,
            strategy_name=self.name,
            confidence=confidence,
            stop_distance_pct=self.stop_distance_pct,
            ttl_seconds=self.ttl_seconds,
        )
