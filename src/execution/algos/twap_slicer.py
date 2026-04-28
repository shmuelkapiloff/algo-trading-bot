"""
TWAP Slicer — time-weighted average price execution for large orders.

Used for large orders (> 5% ADV) where market impact must be minimized
by spreading execution evenly over a longer time window.

Algorithm:
  - Divide total quantity into equal child orders
  - Submit one child order per time interval
  - Each child: marketable limit at mid ± 1 tick (never aggressive cross)
  - If a child is not filled within its interval, carry the remainder
    forward to the next slice (no abandonment of unfilled qty)

Design constraints:
  - Max order size per slice: 5% of ADV
  - Default window: 60 minutes
  - Minimum slice qty: 1 share (fractional shares not supported)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TwapSlice:
    """One child order in a TWAP execution plan."""
    slice_index: int
    symbol: str
    qty: float
    target_time: datetime
    limit_price: Optional[float] = None
    submitted: bool = False
    filled_qty: float = 0.0
    avg_fill_price: Optional[float] = None
    carried_from_prev: float = 0.0  # unfilled qty carried from previous slice


@dataclass
class TwapPlan:
    """Full TWAP execution plan for a parent order."""
    symbol: str
    total_qty: float
    slices: list[TwapSlice] = field(default_factory=list)
    completed: bool = False

    @property
    def total_filled(self) -> float:
        return sum(s.filled_qty for s in self.slices)

    @property
    def remaining_qty(self) -> float:
        return self.total_qty - self.total_filled

    @property
    def avg_fill_price(self) -> Optional[float]:
        fills = [(s.avg_fill_price, s.filled_qty)
                 for s in self.slices if s.avg_fill_price and s.filled_qty > 0]
        if not fills:
            return None
        total = sum(q for _, q in fills)
        return sum(p * q for p, q in fills) / total if total > 0 else None


class TwapSlicer:
    """
    Builds a TWAP execution plan.

    Parameters
    ----------
    window_minutes  : Total execution window in minutes (default 60)
    num_slices      : Number of equal time slices (default 12 = 5 min each)
    max_pct_adv     : Max participation per slice as fraction of ADV (default 0.05)
    """

    def __init__(
        self,
        window_minutes: int = 60,
        num_slices: int = 12,
        max_pct_adv: float = 0.05,
    ) -> None:
        self.window_minutes = window_minutes
        self.num_slices = num_slices
        self.max_pct_adv = max_pct_adv

    def build_plan(
        self,
        symbol: str,
        total_qty: float,
        start_time: datetime,
        adv: float | None = None,
        current_price: float | None = None,
    ) -> TwapPlan:
        """
        Create a TwapPlan with equal-sized time slices.
        Caps each slice at max_pct_adv * adv if adv is provided.
        """
        interval = timedelta(minutes=self.window_minutes / self.num_slices)
        plan = TwapPlan(symbol=symbol, total_qty=total_qty)

        base_slice_qty = total_qty / self.num_slices
        if adv and adv > 0:
            base_slice_qty = min(base_slice_qty, self.max_pct_adv * adv)
        base_slice_qty = max(1.0, round(base_slice_qty, 0))

        remaining = total_qty
        for i in range(self.num_slices):
            if remaining <= 0:
                break
            qty = min(base_slice_qty, remaining)
            limit_price = (
                round(current_price * 1.001, 2)
                if (current_price and i < self.num_slices - 1)
                else None  # last slice: market order to guarantee completion
            )
            plan.slices.append(TwapSlice(
                slice_index=i,
                symbol=symbol,
                qty=qty,
                target_time=start_time + interval * i,
                limit_price=limit_price,
            ))
            remaining -= qty

        logger.info("twap_slicer.plan symbol=%s total_qty=%.0f num_slices=%d interval_min=%.1f",
                    symbol, total_qty, len(plan.slices),
                    self.window_minutes / self.num_slices)
        return plan

    def update_carry(self, plan: TwapPlan, slice_index: int, unfilled: float) -> None:
        """Add unfilled quantity from slice_index to the next slice (carry-forward)."""
        next_idx = slice_index + 1
        if next_idx < len(plan.slices):
            plan.slices[next_idx].qty += unfilled
            plan.slices[next_idx].carried_from_prev += unfilled
            logger.debug("twap_slicer.carry slice=%d→%d qty=%.0f", slice_index, next_idx, unfilled)
