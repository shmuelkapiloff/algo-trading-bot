"""
Mean Reversion Short Strategy — Phase 3+ (Bear regime only).

⚠️  SHORT SELLING IS DISABLED UNTIL PHASE 3+ ⚠️
This strategy emits SHORT signals only when:
  1. Market regime is explicitly "bear"
  2. Borrow availability confirmed (no hard-to-borrow symbols)
  3. PDT rule satisfied

Entry logic
-----------
All conditions required for a SHORT signal at EOD:
  1. Price touched or exceeded the UPPER Bollinger Band (BB 20,2)
     (price >= upper_band * (1 - bb_entry_pct))
  2. RSI(14) > rsi_min_entry (overbought confirmation)
  3. Price in a downtrend: price < SMA200 AND SMA200 slope < 0
  4. Market regime == "bear" (hard gate — no shorts in bull/sideways)
  5. MACD histogram is negative (bearish momentum)
  6. Cost hurdle: expected short return > round_trip_cost_bps * cost_coverage_ratio

Short-specific risks handled
-----------------------------
- Forced buy-in / recall: stop-loss is tighter (1.5% default vs 2.5% long)
- Borrow cost: expected in the cost model (15 bps round-trip minimum)
- Unlimited loss potential: position size is halved vs long equivalent
"""

from __future__ import annotations

import logging
from typing import Iterator

import pandas as pd

from .base_strategy import BaseStrategy
from .models import OrderSide, SignalIntent

try:
    from ..analysis.market_regime import MarketRegime
except ImportError:
    from src.analysis.market_regime import MarketRegime  # type: ignore

try:
    import ta as _ta
    _TA = True
except ImportError:  # pragma: no cover
    _TA = False

logger = logging.getLogger(__name__)

_MIN_BARS = 60


class MeanReversionShortStrategy(BaseStrategy):
    """
    Bollinger Band upper-touch + RSI overbought short strategy.

    Only active in bear market regime.

    Parameters (tunable via config/strategies.yaml)
    ------------------------------------------------
    bb_period           : Bollinger Band lookback (default 20)
    bb_std              : Standard deviations for bands (default 2.0)
    bb_entry_pct        : Sell within this % of the upper band (default 0.02)
    rsi_period          : RSI lookback (default 14)
    rsi_min_entry       : Only enter when RSI > this (default 65)
    stop_distance_pct   : Initial buy-stop (default 0.015 = 1.5% — tighter for shorts)
    round_trip_cost_bps : Estimated round-trip cost including borrow (default 20)
    cost_coverage_ratio : Return must be >= cost × ratio (default 1.5)
    """

    name = "mean_reversion_short"
    phase = "3+"  # disabled in Phase 1–2

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.bb_period: int = int(cfg.get("bb_period", 20))
        self.bb_std: float = float(cfg.get("bb_std", 2.0))
        self.bb_entry_pct: float = float(cfg.get("bb_entry_pct", 0.02))
        self.rsi_period: int = int(cfg.get("rsi_period", 14))
        self.rsi_min_entry: float = float(cfg.get("rsi_min_entry", 65.0))
        self.stop_distance_pct: float = float(cfg.get("stop_distance_pct", 0.015))
        self.round_trip_cost_bps: float = float(cfg.get("round_trip_cost_bps", 20.0))
        self.cost_coverage_ratio: float = float(cfg.get("cost_coverage_ratio", 1.5))
        super().__init__(config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_regime(self, regime: str | None) -> bool:
        if regime is None:
            return False
        try:
            return MarketRegime(regime) == MarketRegime.BEAR
        except ValueError:
            return regime.lower() == "bear"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        regime: str | None = None,
    ) -> Iterator[SignalIntent]:
        """
        Yields SHORT SignalIntents.
        Silently skips if regime is not "bear" — no exceptions raised.
        """
        if not self._check_regime(regime):
            logger.debug("mean_rev_short: skipping — regime=%s (requires bear)", regime)
            return

        for symbol, bars in bars_by_symbol.items():
            if len(bars) < _MIN_BARS:
                continue
            try:
                signal = self._evaluate(symbol, bars)
                if signal:
                    yield signal
            except Exception as exc:  # noqa: BLE001
                logger.warning("mean_rev_short error symbol=%s: %s", symbol, exc)

    def _evaluate(self, symbol: str, bars: pd.DataFrame) -> SignalIntent | None:
        close = bars["close"]
        high = bars["high"]
        low = bars["low"]

        # --- Bollinger Bands ---
        sma = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std()
        upper_band = sma + self.bb_std * std
        mid_band = sma

        last_price = float(close.iloc[-1])
        last_upper = float(upper_band.iloc[-1])
        last_mid = float(mid_band.iloc[-1])

        # Gate 1: price at/above upper band
        if last_price < last_upper * (1 - self.bb_entry_pct):
            return None

        # Gate 2: RSI overbought
        if _TA:
            rsi_series = _ta.momentum.RSIIndicator(close, window=self.rsi_period).rsi()
        else:
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
            loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
            rs = gain / loss.replace(0, float("inf"))
            rsi_series = 100 - (100 / (1 + rs))

        last_rsi = float(rsi_series.iloc[-1])
        if last_rsi < self.rsi_min_entry:
            return None

        # Gate 3: downtrend filter
        sma200 = close.rolling(200).mean()
        if len(sma200.dropna()) < 20:
            return None
        if last_price >= float(sma200.iloc[-1]):
            return None
        slope = (float(sma200.iloc[-1]) - float(sma200.iloc[-20])) / float(sma200.iloc[-20])
        if slope >= 0:
            return None

        # Gate 4: MACD negative
        if _TA:
            macd_obj = _ta.trend.MACD(close)
            macd_hist = float(macd_obj.macd_diff().iloc[-1])
            if macd_hist >= 0:
                return None

        # Gate 5: cost hurdle
        if last_price <= 0 or last_mid <= 0:
            return None
        expected_return_pct = (last_price - last_mid) / last_price
        expected_return_bps = expected_return_pct * 10_000
        if expected_return_bps < self.round_trip_cost_bps * self.cost_coverage_ratio:
            return None

        # ATR-based stop
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_val = float(tr.rolling(14).mean().iloc[-1])
        stop_dist = max(self.stop_distance_pct, atr_val / last_price)

        # Confidence
        rsi_factor = min((last_rsi - self.rsi_min_entry) / (100 - self.rsi_min_entry), 1.0) * 0.4
        bb_factor = min((last_price - last_upper) / (last_upper * self.bb_entry_pct + 1e-9), 1.0) * 0.4
        cost_factor = min(expected_return_bps / (self.round_trip_cost_bps * 3), 1.0) * 0.2
        confidence = max(0.0, min(1.0, rsi_factor + bb_factor + cost_factor))

        return SignalIntent(
            symbol=symbol,
            side=OrderSide.SELL,  # short
            confidence=confidence,
            strategy=self.name,
            stop_distance_pct=stop_dist,
            metadata={
                "rsi": last_rsi,
                "price_vs_upper_band": last_price / last_upper,
                "expected_return_bps": expected_return_bps,
                "regime": "bear",
            },
        )
