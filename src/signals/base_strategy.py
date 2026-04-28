"""
Abstract base class for all trading strategies.

Each concrete strategy:
  1. Receives a DataFrame of OHLCV bars + current market regime.
  2. Computes technical indicators.
  3. Emits zero or more SignalIntent objects.

Separation of Concerns
-----------------------
Strategies are PURE signal generators. They do NOT:
  - Size positions  (Portfolio layer)
  - Check risk limits  (PreTradeGateway)
  - Submit orders  (Execution layer)
  - Read from Redis or DB

This makes strategies independently testable with synthetic bar data.

Subclass contract
-----------------
Implement `generate_signals(bars, regime)`.
Call `_check_viability(signal)` before yielding each signal — this
runs the fast pre-check (price floor, min_adv, confidence threshold)
without requiring the full Pre-Trade Gateway.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Iterator

import pandas as pd

from .models import OrderSide, SignalIntent
from ..analysis.market_regime import MarketRegime

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base for all EOD long-only strategies.

    Parameters
    ----------
    name            : Unique identifier (used as signal.strategy_name).
    allowed_regimes : Set of regimes where this strategy is active.
                      Defaults to all regimes.
    min_confidence  : Signals below this threshold are dropped silently.
    min_price       : Reject symbols below this price (penny-stock filter).
    """

    def __init__(
        self,
        name: str,
        allowed_regimes: set[MarketRegime] | None = None,
        min_confidence: float = 0.50,
        min_price: float = 5.0,
    ) -> None:
        self.name = name
        self.allowed_regimes: set[MarketRegime] = allowed_regimes or set(MarketRegime)
        self.min_confidence = min_confidence
        self.min_price = min_price

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_signals(
        self,
        bars: dict[str, pd.DataFrame],
        regime: MarketRegime,
    ) -> Iterator[SignalIntent]:
        """
        Analyse bars for all symbols and yield SignalIntent objects.

        Parameters
        ----------
        bars   : {symbol: DataFrame(timestamp, open, high, low, close_adj, volume)}
                 DataFrames are sorted ascending by date and split-adjusted.
        regime : Current market regime (from regime_cache).
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _regime_allowed(self, regime: MarketRegime) -> bool:
        """Return True if the current regime permits this strategy."""
        allowed = regime in self.allowed_regimes
        if not allowed:
            logger.debug(
                "[%s] Skipping scan — regime=%s not in allowed=%s",
                self.name,
                regime.value,
                {r.value for r in self.allowed_regimes},
            )
        return allowed

    def _check_viability(
        self,
        symbol: str,
        confidence: float,
        last_price: float,
    ) -> bool:
        """
        Fast pre-filter before constructing a SignalIntent.
        Returns False (and logs DEBUG) if the signal should be dropped.
        """
        if last_price < self.min_price:
            logger.debug(
                "[%s] %s rejected: price %.2f < min_price %.2f",
                self.name,
                symbol,
                last_price,
                self.min_price,
            )
            return False
        if confidence < self.min_confidence:
            logger.debug(
                "[%s] %s rejected: confidence %.3f < min_confidence %.3f",
                self.name,
                symbol,
                confidence,
                self.min_confidence,
            )
            return False
        return True

    def _need_bars(self, df: pd.DataFrame, min_bars: int, symbol: str) -> bool:
        """Return True if df has enough rows for indicator computation."""
        if len(df) < min_bars:
            logger.debug(
                "[%s] %s skipped: only %d bars (need %d)",
                self.name,
                symbol,
                len(df),
                min_bars,
            )
            return False
        return True
