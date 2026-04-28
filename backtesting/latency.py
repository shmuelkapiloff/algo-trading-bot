"""
Latency & Fill Simulation for Backtesting.

Simulates realistic execution delays and fill probabilities based on:
  - Order size relative to ADV (larger orders = higher latency + worse fills)
  - Market regime (volatile markets = wider spreads, higher slippage)
  - Time of day (open/close = higher latency)
  - Order type (limit vs market)

Used by the backtest engine to apply realistic transaction cost assumptions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class FillSimResult:
    filled: bool
    fill_price: Optional[float]
    fill_latency_ms: float     # simulated latency in milliseconds
    slippage_bps: float        # simulated slippage in basis points
    partial_pct: float         # fraction filled (0.0–1.0); 1.0 = full fill


class LatencyModel:
    """
    Simulates order submission-to-fill latency.

    Latency components:
      - Network + broker processing: 5–15 ms baseline
      - Queue position at open/close: +50–200 ms
      - Large order premium: +10 ms per 1% of ADV

    Parameters
    ----------
    base_latency_ms      : Minimum latency (default 8 ms)
    open_close_premium_ms: Extra latency at market open/close (default 150 ms)
    seed                 : Random seed for reproducibility
    """

    def __init__(
        self,
        base_latency_ms: float = 8.0,
        open_close_premium_ms: float = 150.0,
        seed: Optional[int] = None,
    ) -> None:
        self.base_latency_ms = base_latency_ms
        self.open_close_premium_ms = open_close_premium_ms
        self._rng = random.Random(seed)

    def sample_latency_ms(
        self,
        order_size_pct_adv: float = 0.01,
        is_open_close: bool = False,
    ) -> float:
        """Sample a latency value in milliseconds."""
        base = self.base_latency_ms + self._rng.uniform(0, 7)
        size_premium = order_size_pct_adv * 1000  # 10 ms per 1% ADV
        oc_premium = self.open_close_premium_ms * self._rng.uniform(0.5, 1.5) if is_open_close else 0
        return base + size_premium + oc_premium


class FillSimulator:
    """
    Simulates fills for backtest orders, applying latency and slippage.

    Parameters
    ----------
    base_slippage_bps  : Minimum one-way slippage (default 5 bps)
    spread_bps         : Bid-ask spread (default 5 bps)
    partial_fill_prob  : Probability of partial fill for large orders (default 0.10)
    seed               : Random seed for reproducibility
    """

    def __init__(
        self,
        base_slippage_bps: float = 5.0,
        spread_bps: float = 5.0,
        partial_fill_prob: float = 0.10,
        seed: Optional[int] = None,
    ) -> None:
        self.base_slippage_bps = base_slippage_bps
        self.spread_bps = spread_bps
        self.partial_fill_prob = partial_fill_prob
        self._rng = random.Random(seed)
        self._latency_model = LatencyModel(seed=seed)

    def simulate(
        self,
        symbol: str,
        side: str,              # "buy" or "sell"
        qty: float,
        limit_price: Optional[float],
        market_price: float,
        order_size_pct_adv: float = 0.01,
        is_open_close: bool = False,
        regime: str = "bull",
    ) -> FillSimResult:
        """
        Simulate a single order fill.

        For limit orders: filled only if market_price crosses the limit.
        For market orders: always filled with slippage.
        """
        latency_ms = self._latency_model.sample_latency_ms(order_size_pct_adv, is_open_close)

        # Regime multiplier — volatile markets have 2× slippage
        regime_mult = 2.0 if regime == "bear" else 1.0

        # Slippage
        slippage_bps = (
            self.base_slippage_bps
            + self.spread_bps / 2
            + order_size_pct_adv * 50  # 50 bps per 100% ADV
        ) * regime_mult * self._rng.uniform(0.5, 1.5)

        slippage_pct = slippage_bps / 10_000
        if side.lower() == "buy":
            fill_price = market_price * (1 + slippage_pct)
        else:
            fill_price = market_price * (1 - slippage_pct)

        # Limit price check
        if limit_price is not None:
            if side.lower() == "buy" and fill_price > limit_price:
                return FillSimResult(False, None, latency_ms, slippage_bps, 0.0)
            if side.lower() == "sell" and fill_price < limit_price:
                return FillSimResult(False, None, latency_ms, slippage_bps, 0.0)

        # Partial fill simulation
        partial_pct = 1.0
        if order_size_pct_adv > 0.05 and self._rng.random() < self.partial_fill_prob:
            partial_pct = self._rng.uniform(0.3, 0.9)

        return FillSimResult(
            filled=True,
            fill_price=round(fill_price, 4),
            fill_latency_ms=latency_ms,
            slippage_bps=round(slippage_bps, 2),
            partial_pct=partial_pct,
        )
