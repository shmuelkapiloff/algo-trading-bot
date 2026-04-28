"""
Partial Fill Handler — TTL-aware decision matrix for partially filled orders.

Design (from TRADING_BOT_PLAN.md):
  When a fill is partial, the remaining unfilled quantity must be handled:

  Decision matrix
  ---------------
  | Remaining % | Time to market close | Regime  | Decision              |
  |-------------|----------------------|---------|----------------------|
  | > 50%       | > 60 min             | any     | keep — chase fill    |
  | > 50%       | < 60 min             | bull    | cancel remainder     |
  | > 50%       | < 60 min             | bear    | cancel remainder     |
  | < 50%       | any                  | any     | keep small remainder |
  | any         | last 5 min           | any     | ALWAYS cancel        |

  "keep" means: leave the limit order alive
  "cancel" means: cancel remainder, book the partial as a real position
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

MARKET_CLOSE_TIME = time(16, 0)       # 16:00 ET
LAST_MINUTE_CUTOFF = time(15, 55)     # cancel all partials in last 5 min
NEAR_CLOSE_MINUTES = 60


class PartialFillDecision(str, Enum):
    KEEP = "keep"      # leave remainder open
    CANCEL = "cancel"  # cancel remainder, book partial position


@dataclass
class PartialFillContext:
    order_id: str
    symbol: str
    original_qty: float
    filled_qty: float
    regime: Optional[str] = None
    current_time: Optional[datetime] = None

    @property
    def remaining_qty(self) -> float:
        return self.original_qty - self.filled_qty

    @property
    def remaining_pct(self) -> float:
        if self.original_qty <= 0:
            return 0.0
        return self.remaining_qty / self.original_qty

    @property
    def minutes_to_close(self) -> float:
        if self.current_time is None:
            return 999.0
        t = self.current_time.astimezone(timezone.utc)
        # approximate: convert to ET-equivalent for comparison
        close_today = self.current_time.replace(
            hour=MARKET_CLOSE_TIME.hour,
            minute=MARKET_CLOSE_TIME.minute,
            second=0,
            microsecond=0,
        )
        delta = (close_today - self.current_time).total_seconds() / 60
        return max(delta, 0.0)


class PartialFillHandler:
    """
    Evaluates partial fill events and decides whether to keep or cancel
    the unfilled remainder.
    """

    def decide(self, ctx: PartialFillContext) -> PartialFillDecision:
        """
        Returns KEEP or CANCEL based on the TTL-aware decision matrix.
        """
        minutes_left = ctx.minutes_to_close

        # Hard rule: last 5 minutes — always cancel
        if minutes_left <= 5.0:
            logger.info(
                "partial_fill.cancel order_id=%s reason=last_5_min remaining_pct=%.2f",
                ctx.order_id, ctx.remaining_pct,
            )
            return PartialFillDecision.CANCEL

        # Large remainder close to end of day
        if ctx.remaining_pct > 0.50 and minutes_left < NEAR_CLOSE_MINUTES:
            logger.info(
                "partial_fill.cancel order_id=%s reason=near_close remaining_pct=%.2f "
                "minutes_left=%.1f",
                ctx.order_id, ctx.remaining_pct, minutes_left,
            )
            return PartialFillDecision.CANCEL

        # Small remainder — keep regardless of time
        logger.debug(
            "partial_fill.keep order_id=%s remaining_pct=%.2f minutes_left=%.1f",
            ctx.order_id, ctx.remaining_pct, minutes_left,
        )
        return PartialFillDecision.KEEP
