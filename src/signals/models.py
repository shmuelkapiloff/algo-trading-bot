"""
Data contracts between the three core layers.

  Signal Layer  → produces:  SignalIntent
  Portfolio Layer → consumes SignalIntent, produces: OrderIntent
  Execution Layer → consumes OrderIntent, produces: OrderEvent

These are plain dataclasses (no business logic) to keep the
Separation of Concerns contract defined in TRADING_BOT_PLAN.md §6ד clean.
"""

from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class SignalIntent:
    """
    Output of the Strategy layer.

    Fields are deliberately sparse: the Portfolio layer is responsible
    for translating confidence + stop_distance_pct into a concrete qty.
    qty may be populated early (e.g. for gate-level ADV checks) but is
    not required until OrderIntent is constructed.
    """

    # Required at construction time
    symbol: str
    side: OrderSide
    strategy_name: str

    # Generated identity
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Signal quality
    confidence: float = 0.0  # 0.0–1.0; used for EV estimation
    ttl_seconds: int = 86_400  # how long this signal is actionable

    # Human-readable context
    reason: str = ""  # e.g. "RSI < 30, below Bollinger Lower"
    regime: str = ""  # "bull" | "bear" | "sideways"

    # Risk anchors (optional at signal stage; required before order submission)
    stop_distance_pct: Optional[float] = None  # e.g. 0.03 → 3% below entry
    qty: Optional[float] = None  # shares; None until Portfolio sizes it

    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def with_qty(self, qty: float) -> "SignalIntent":
        """Return a copy with qty adjusted (used by gates to reduce size)."""
        return dataclasses.replace(self, qty=qty)


@dataclass
class OrderIntent:
    """
    Output of the Portfolio layer. Fully sized and ready for Execution.

    The Execution layer must not access strategy logic or risk parameters —
    it only reads what is here.
    """

    # Required
    symbol: str
    side: OrderSide
    qty: float

    # Lineage
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_id: str = ""  # traces back to source SignalIntent

    # Execution anchors
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None

    # Metadata
    strategy_name: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
