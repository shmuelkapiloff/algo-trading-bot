"""
Order State Machine — NEW → ACK → PARTIAL → {FILLED / CANCELED / REJECTED / EXPIRED}

Enforces legal state transitions and prevents illegal ones.
Used by the router and reconciliation module to track live order lifecycle.

Phase 5.5 additions
-------------------
- ``transition_ordered()``: strict per-order_id sequence validation.
  Each fill/status event must carry a monotonically-increasing ``seq``
  integer.  Events that arrive with ``seq ≤ last_seq`` are classified as
  *late/duplicate* and are stored in a pending buffer keyed by ``seq``
  rather than discarded.  Once all gaps are closed the pending events are
  replayed automatically via ``_flush_pending()``.
- ``get_pending_count()``: returns total buffered out-of-order events.

State diagram:
    NEW ──► ACK ──► PARTIAL ──► FILLED
      └──► SENT ──► ACK          │
      └──► CANCELED              ▼
      └──► REJECTED          CANCELED (partial → abandoned)
                             EXPIRED
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class OrderState(str, Enum):
    NEW = "NEW"
    SENT = "SENT"  # submitted to broker, awaiting ACK
    ACK = "ACK"  # broker acknowledged
    PARTIAL = "PARTIAL"  # partially filled
    FILLED = "FILLED"  # terminal: fully filled
    CANCELED = "CANCELED"  # terminal: cancelled
    REJECTED = "REJECTED"  # terminal: rejected by broker
    EXPIRED = "EXPIRED"  # terminal: DAY order expired


TERMINAL_STATES = {
    OrderState.FILLED,
    OrderState.CANCELED,
    OrderState.REJECTED,
    OrderState.EXPIRED,
}

_LEGAL_TRANSITIONS: dict[OrderState, set[OrderState]] = {
    OrderState.NEW: {OrderState.SENT, OrderState.CANCELED},
    OrderState.SENT: {OrderState.ACK, OrderState.REJECTED, OrderState.CANCELED},
    OrderState.ACK: {
        OrderState.PARTIAL,
        OrderState.FILLED,
        OrderState.CANCELED,
        OrderState.EXPIRED,
    },
    OrderState.PARTIAL: {OrderState.FILLED, OrderState.CANCELED, OrderState.EXPIRED},
    # Terminal states have no outgoing transitions
    OrderState.FILLED: set(),
    OrderState.CANCELED: set(),
    OrderState.REJECTED: set(),
    OrderState.EXPIRED: set(),
}


@dataclass
class OrderRecord:
    """In-memory record of a single order's lifecycle."""

    order_id: str
    symbol: str
    strategy_name: str
    state: OrderState = OrderState.NEW
    broker_order_id: Optional[str] = None
    submitted_qty: int = 0
    filled_qty: float = 0.0
    fill_price: Optional[float] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_error: Optional[str] = None
    # Phase 5.5 — sequence tracking
    last_seq: int = -1  # last applied event sequence number (-1 = none yet)

    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES


class OrderStateMachine:
    """
    Manages lifecycle of all in-flight orders.

    Not persisted to DB directly — that's OmsLedger's job.
    This is the in-memory view used by the router during the scan session.
    """

    def __init__(self) -> None:
        self._orders: dict[str, OrderRecord] = {}
        # Phase 5.5 — pending buffer: order_id → {seq: kwargs}
        self._pending: dict[str, dict[int, dict[str, Any]]] = {}

    def register(self, record: OrderRecord) -> None:
        """Register a new order record."""
        self._orders[record.order_id] = record
        logger.debug(
            "Order registered: %s  symbol=%s  state=%s",
            record.order_id,
            record.symbol,
            record.state.value,
        )

    def transition(
        self,
        order_id: str,
        target: OrderState,
        broker_order_id: Optional[str] = None,
        filled_qty: Optional[float] = None,
        fill_price: Optional[float] = None,
        error: Optional[str] = None,
    ) -> bool:
        """
        Attempt to transition an order to a new state.
        Returns True if the transition was applied, False if illegal.
        """
        record = self._orders.get(order_id)
        if record is None:
            logger.error("transition: order_id %s not found", order_id)
            return False

        if record.is_terminal():
            logger.debug(
                "transition: %s is already in terminal state %s — ignoring",
                order_id,
                record.state.value,
            )
            return False

        allowed = _LEGAL_TRANSITIONS.get(record.state, set())
        if target not in allowed:
            logger.warning(
                "ILLEGAL transition %s → %s for order %s (allowed: %s)",
                record.state.value,
                target.value,
                order_id,
                {s.value for s in allowed},
            )
            return False

        old_state = record.state
        record.state = target
        record.updated_at = datetime.now(timezone.utc)

        if broker_order_id:
            record.broker_order_id = broker_order_id
        if filled_qty is not None:
            record.filled_qty = filled_qty
        if fill_price is not None:
            record.fill_price = fill_price
        if error:
            record.last_error = error

        logger.info(
            "Order %s: %s → %s  (symbol=%s  filled=%.0f)",
            order_id,
            old_state.value,
            target.value,
            record.symbol,
            record.filled_qty,
        )
        return True

    def get(self, order_id: str) -> Optional[OrderRecord]:
        return self._orders.get(order_id)

    def get_open_orders(self) -> list[OrderRecord]:
        """Return all non-terminal orders."""
        return [r for r in self._orders.values() if not r.is_terminal()]

    def get_orders_for_symbol(self, symbol: str) -> list[OrderRecord]:
        return [r for r in self._orders.values() if r.symbol == symbol]

    def clear_terminal(self) -> int:
        """Remove terminal orders from memory. Returns count removed."""
        terminal = [oid for oid, r in self._orders.items() if r.is_terminal()]
        for oid in terminal:
            del self._orders[oid]
            self._pending.pop(oid, None)
        return len(terminal)

    # ------------------------------------------------------------------
    # Phase 5.5 — strict per-order_id event ordering + late-event buffer
    # ------------------------------------------------------------------

    def transition_ordered(
        self,
        order_id: str,
        target: OrderState,
        seq: int,
        *,
        broker_order_id: Optional[str] = None,
        filled_qty: Optional[float] = None,
        fill_price: Optional[float] = None,
        error: Optional[str] = None,
    ) -> bool:
        """
        Apply a state transition with sequence-number enforcement.

        Rules
        -----
        - ``seq`` must be strictly greater than the record's ``last_seq``.
        - If ``seq <= last_seq``:
            * ``seq == last_seq``   → duplicate; silently ignored.
            * ``seq < last_seq``    → late/out-of-order; buffered in
              ``_pending`` and replayed once the gap closes.
        - After a successful transition, any buffered events whose ``seq``
          is exactly ``last_seq + 1`` are flushed automatically.

        Returns True if the event was applied (immediately or via flush).
        """
        record = self._orders.get(order_id)
        if record is None:
            logger.error("transition_ordered: order_id %s not found", order_id)
            return False

        if seq <= record.last_seq:
            if seq == record.last_seq:
                logger.debug(
                    "transition_ordered: duplicate event seq=%d for order %s — ignored",
                    seq, order_id,
                )
            else:
                logger.warning(
                    "transition_ordered: out-of-order event seq=%d (last=%d) for order %s — buffered",
                    seq, record.last_seq, order_id,
                )
                buf = self._pending.setdefault(order_id, {})
                buf[seq] = dict(
                    target=target,
                    broker_order_id=broker_order_id,
                    filled_qty=filled_qty,
                    fill_price=fill_price,
                    error=error,
                )
            return False

        # Gap check: enforce strict ordering (seq must be last_seq + 1)
        expected = record.last_seq + 1
        if seq != expected:
            logger.warning(
                "transition_ordered: seq gap detected for order %s: expected %d got %d — buffering",
                order_id, expected, seq,
            )
            buf = self._pending.setdefault(order_id, {})
            buf[seq] = dict(
                target=target,
                broker_order_id=broker_order_id,
                filled_qty=filled_qty,
                fill_price=fill_price,
                error=error,
            )
            return False

        applied = self.transition(
            order_id,
            target,
            broker_order_id=broker_order_id,
            filled_qty=filled_qty,
            fill_price=fill_price,
            error=error,
        )
        if applied:
            record.last_seq = seq
            self._flush_pending(order_id)
        return applied

    def _flush_pending(self, order_id: str) -> None:
        """Replay any consecutively buffered events starting at last_seq + 1."""
        buf = self._pending.get(order_id)
        if not buf:
            return
        record = self._orders.get(order_id)
        if record is None:
            return

        while True:
            next_seq = record.last_seq + 1
            kwargs = buf.pop(next_seq, None)
            if kwargs is None:
                break
            target = kwargs.pop("target")
            applied = self.transition(order_id, target, **kwargs)
            if applied:
                record.last_seq = next_seq
            else:
                break  # illegal transition — stop flush

        if not buf:
            del self._pending[order_id]

    def get_pending_count(self) -> int:
        """Return total number of buffered out-of-order events."""
        return sum(len(v) for v in self._pending.values())
