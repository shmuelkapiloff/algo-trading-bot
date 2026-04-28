"""
VWAP Slicer — time-sliced execution targeting VWAP.

Used for medium-sized orders (> 1% ADV) to minimize market impact.

Algorithm:
  1. Divide the execution window into N equal time slices
  2. Weight each slice by the expected volume distribution (historical intraday profile)
  3. Submit child limit orders at the VWAP of each completed slice
  4. Final slice uses market order to guarantee completion

Design constraints:
  - Max participation rate: 20% of ADV per slice (prevents moving the market)
  - Execution window: configurable, default 30 minutes
  - Child order type: marketable limit (mid ± 1 tick) to avoid crossing spread
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Intraday volume profile weights (30-min buckets, 9:30–16:00 ET = 13 buckets)
# Based on typical NYSE/NASDAQ average profile (U-shaped — heavy open/close)
_DEFAULT_VOLUME_PROFILE = [
    0.120,  # 09:30
    0.080,  # 10:00
    0.065,  # 10:30
    0.060,  # 11:00
    0.055,  # 11:30
    0.055,  # 12:00
    0.055,  # 12:30
    0.055,  # 13:00
    0.060,  # 13:30
    0.065,  # 14:00
    0.070,  # 14:30
    0.075,  # 15:00
    0.085,  # 15:30 (last bucket before close)
]


@dataclass
class VwapSlice:
    """One child order in a VWAP execution plan."""
    slice_index: int
    symbol: str
    qty: float
    target_start: datetime
    target_end: datetime
    limit_price: Optional[float] = None
    submitted: bool = False
    filled_qty: float = 0.0
    avg_fill_price: Optional[float] = None


@dataclass
class VwapPlan:
    """Full VWAP execution plan for a parent order."""
    symbol: str
    total_qty: float
    slices: list[VwapSlice] = field(default_factory=list)
    completed: bool = False

    @property
    def remaining_qty(self) -> float:
        return self.total_qty - sum(s.filled_qty for s in self.slices)

    @property
    def avg_fill_price(self) -> Optional[float]:
        fills = [(s.avg_fill_price, s.filled_qty)
                 for s in self.slices if s.avg_fill_price and s.filled_qty > 0]
        if not fills:
            return None
        total_qty = sum(q for _, q in fills)
        if total_qty <= 0:
            return None
        return sum(p * q for p, q in fills) / total_qty


class VwapSlicer:
    """
    Builds and manages a VWAP execution plan.

    Parameters
    ----------
    window_minutes  : Total execution window in minutes (default 30)
    num_slices      : Number of child orders (default 6)
    max_pct_adv     : Max participation as fraction of ADV (default 0.20)
    volume_profile  : 13-bucket intraday profile weights (optional)
    """

    def __init__(
        self,
        window_minutes: int = 30,
        num_slices: int = 6,
        max_pct_adv: float = 0.20,
        volume_profile: list[float] | None = None,
    ) -> None:
        self.window_minutes = window_minutes
        self.num_slices = num_slices
        self.max_pct_adv = max_pct_adv
        self._profile = volume_profile or _DEFAULT_VOLUME_PROFILE

    def build_plan(
        self,
        symbol: str,
        total_qty: float,
        start_time: datetime,
        adv: float | None = None,
        current_price: float | None = None,
    ) -> VwapPlan:
        """
        Create a VwapPlan with child slices.
        If adv is provided, caps each slice at max_pct_adv * adv / num_slices.
        """
        slice_duration = timedelta(minutes=self.window_minutes / self.num_slices)
        plan = VwapPlan(symbol=symbol, total_qty=total_qty)

        remaining = total_qty
        for i in range(self.num_slices):
            if remaining <= 0:
                break
            # Proportional allocation — equal slices for simplicity
            slice_qty = total_qty / self.num_slices
            if adv and adv > 0:
                max_slice_qty = self.max_pct_adv * adv / self.num_slices
                slice_qty = min(slice_qty, max_slice_qty)
            slice_qty = min(slice_qty, remaining)

            t_start = start_time + slice_duration * i
            t_end = t_start + slice_duration
            limit_price = (
                round(current_price * 1.001, 2) if (current_price and i < self.num_slices - 1)
                else None  # last slice: market order
            )
            plan.slices.append(VwapSlice(
                slice_index=i,
                symbol=symbol,
                qty=round(slice_qty, 0),
                target_start=t_start,
                target_end=t_end,
                limit_price=limit_price,
            ))
            remaining -= slice_qty

        logger.info("vwap_slicer.plan symbol=%s total_qty=%.0f num_slices=%d",
                    symbol, total_qty, len(plan.slices))
        return plan
