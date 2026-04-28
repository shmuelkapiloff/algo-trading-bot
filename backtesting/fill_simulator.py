"""Latency-aware fill simulation for backtesting.

Models realistic fill behaviour:
  - Fill latency:    orders are not filled instantaneously (50–500 ms jitter)
  - Capacity limit:  max fill per bar = min(ADV_limit, bar_volume_limit)
  - Partial fills:   large orders near the ADV cap receive only a fraction
  - Non-marketable:  limit orders that miss the bar close → NO_FILL

Fill outcome constants
----------------------
  FULL    : entire qty filled at simulated price
  PARTIAL : fraction filled, remaining carries to next bar
  NO_FILL : market conditions prevented any fill
  EXPIRED : order TTL elapsed without fill
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FillStatus(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    NO_FILL = "no_fill"
    EXPIRED = "expired"


@dataclass
class FillResult:
    status: FillStatus
    filled_qty: int
    remaining_qty: int
    fill_price: float
    latency_ms: float


class FillSimulator:
    """Simulates fill behaviour for backtesting.

    Parameters
    ----------
    max_adv_fill_pct:
        Maximum fraction of ADV (in shares) that can be filled per bar.
        Default: 2 % of ADV per bar.
    max_bar_vol_fill_pct:
        Maximum fraction of the bar's own volume fillable per bar (default 1 %).
    base_latency_ms:
        Median fill latency in milliseconds (default 120 ms).
    latency_jitter:
        Fraction of base_latency_ms for ± uniform jitter (default 0.30 = ±30 %).
    partial_fill_prob:
        Probability of receiving a partial fill when the order exceeds ADV cap
        (default 0.40).
    marketable_fill_prob:
        Baseline probability that a marketable order fills in a given bar
        (default 0.95, very high for liquid stocks).
    rng_seed:
        Optional integer seed for deterministic replay.
    """

    def __init__(
        self,
        max_adv_fill_pct: float = 0.02,
        max_bar_vol_fill_pct: float = 0.01,
        base_latency_ms: float = 120.0,
        latency_jitter: float = 0.30,
        partial_fill_prob: float = 0.40,
        marketable_fill_prob: float = 0.95,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.max_adv_fill_pct = max_adv_fill_pct
        self.max_bar_vol_fill_pct = max_bar_vol_fill_pct
        self.base_latency_ms = base_latency_ms
        self.latency_jitter = latency_jitter
        self.partial_fill_prob = partial_fill_prob
        self.marketable_fill_prob = marketable_fill_prob
        self._rng = random.Random(rng_seed)

    # ------------------------------------------------------------------

    def _latency(self) -> float:
        jitter = self._rng.uniform(-self.latency_jitter, self.latency_jitter)
        return max(1.0, self.base_latency_ms * (1.0 + jitter))

    def _max_fillable(self, adv_shares: float, bar_volume: float) -> int:
        adv_limit = max(int(adv_shares * self.max_adv_fill_pct), 1)
        bar_limit = max(int(bar_volume * self.max_bar_vol_fill_pct), 1)
        return min(adv_limit, bar_limit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate_fill(
        self,
        order_qty: int,
        fill_price: float,
        adv_shares: float,
        bar_volume: float,
        is_marketable: bool = True,
    ) -> FillResult:
        """Simulate one fill attempt for a single bar.

        Parameters
        ----------
        order_qty:     total shares requested
        fill_price:    post-slippage price (from CostModel.apply_fill_price)
        adv_shares:    average daily volume in shares for this symbol
        bar_volume:    actual volume traded in this bar
        is_marketable: True for marketable limit orders; False for passive limits
        """
        latency_ms = self._latency()
        order_qty = max(order_qty, 0)

        # ── Non-marketable: probabilistic fill ────────────────────────
        if not is_marketable:
            if self._rng.random() > 0.70:
                return FillResult(
                    status=FillStatus.NO_FILL,
                    filled_qty=0,
                    remaining_qty=order_qty,
                    fill_price=fill_price,
                    latency_ms=latency_ms,
                )

        # ── Marketable: baseline fill probability ──────────────────────
        if self._rng.random() > self.marketable_fill_prob:
            return FillResult(
                status=FillStatus.NO_FILL,
                filled_qty=0,
                remaining_qty=order_qty,
                fill_price=fill_price,
                latency_ms=latency_ms,
            )

        # ── Capacity check ─────────────────────────────────────────────
        max_fillable = self._max_fillable(adv_shares, bar_volume)

        if order_qty <= max_fillable:
            # Full fill — order fits within capacity
            return FillResult(
                status=FillStatus.FULL,
                filled_qty=order_qty,
                remaining_qty=0,
                fill_price=fill_price,
                latency_ms=latency_ms,
            )

        # ── Order exceeds capacity — partial fill decision ────────────
        if self._rng.random() < self.partial_fill_prob:
            # Partial: fill between 50 % and 100 % of max_fillable
            partial_frac = self._rng.uniform(0.50, 1.0)
            filled = max(1, min(int(max_fillable * partial_frac), order_qty))
            return FillResult(
                status=FillStatus.PARTIAL,
                filled_qty=filled,
                remaining_qty=order_qty - filled,
                fill_price=fill_price,
                latency_ms=latency_ms,
            )

        # Full fill capped at max_fillable (accepts remaining next bar)
        filled = min(max_fillable, order_qty)
        status = FillStatus.FULL if filled == order_qty else FillStatus.PARTIAL
        return FillResult(
            status=status,
            filled_qty=filled,
            remaining_qty=order_qty - filled,
            fill_price=fill_price,
            latency_ms=latency_ms,
        )
