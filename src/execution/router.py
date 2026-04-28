"""
Execution Router — SignalIntent → BrokerOrder (Phase 1).

Single responsibility: translate a sized, risk-approved SignalIntent into
an actual order submitted to the broker, then record the outcome.

Order type selection (from execution_safety config):
  default_order_type: marketable_limit
  buffer_bps:         5 bps above/below current price

Price source for limit calculation:
  Uses signal.stop_distance_pct to derive a sensible limit price.
  In Phase 1 we use the last_price from the most recent bar.
  Phase 2: replace with live Bid/Ask from market data WebSocket.

Retry policy: tenacity 5 attempts with exponential backoff.
Fail-safe: on retries exhausted, record REJECTED in OMS and continue.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from .broker import AbstractBroker, BrokerOrderSide, BrokerOrderStatus
from .order_state_machine import OrderRecord, OrderState, OrderStateMachine
from ..data.models import OrderEventType
from ..data.oms_ledger import OmsLedger
from ..events.bus import EventBus
from ..events import topics
from ..signals.models import OrderSide, SignalIntent

logger = logging.getLogger(__name__)

_TAKE_PROFIT_MULTIPLIER = 2.0  # Risk:Reward = 1:2


class ExecutionRouter:
    """
    Translates approved SignalIntents into broker orders.

    Parameters
    ----------
    broker          : AbstractBroker implementation (AlpacaBroker in Phase 1)
    oms_ledger      : OmsLedger for idempotent event recording
    event_bus       : EventBus for publishing fill/reject events
    state_machine   : OrderStateMachine for in-flight order tracking
    buffer_bps      : Marketable limit buffer (default 5 bps)
    """

    def __init__(
        self,
        broker: AbstractBroker,
        oms_ledger: OmsLedger,
        event_bus: EventBus,
        state_machine: OrderStateMachine,
        buffer_bps: float = 5.0,
    ) -> None:
        self._broker = broker
        self._ledger = oms_ledger
        self._bus = event_bus
        self._sm = state_machine
        self._buffer = buffer_bps / 10_000

    async def execute(
        self,
        signal: SignalIntent,
        last_price: float,
    ) -> Optional[str]:
        """
        Submit an order for the given signal.

        Parameters
        ----------
        signal     : Sized, risk-approved SignalIntent (qty must be set)
        last_price : Last traded price used for limit price calculation

        Returns
        -------
        broker_order_id if successfully submitted, None on failure.
        """
        if signal.qty is None or int(signal.qty) <= 0:
            logger.warning("[router] Skipping %s: qty is None or zero", signal.symbol)
            return None

        order_id = str(uuid.uuid4())
        qty = int(signal.qty)
        broker_side = BrokerOrderSide(signal.side.value)

        # ── Record NEW event in OMS ───────────────────────────────────
        await self._ledger.record_event(
            order_id=order_id,
            symbol=signal.symbol,
            event_type=OrderEventType.NEW,
            payload={
                "side": signal.side.value,
                "qty": qty,
                "strategy_name": signal.strategy_name,
                "confidence": signal.confidence,
                "stop_distance_pct": signal.stop_distance_pct,
            },
            strategy=signal.strategy_name,
        )

        # ── Register in state machine ─────────────────────────────────
        record = OrderRecord(
            order_id=order_id,
            symbol=signal.symbol,
            strategy_name=signal.strategy_name,
            submitted_qty=qty,
        )
        self._sm.register(record)

        # ── Calculate limit price ─────────────────────────────────────
        if broker_side == BrokerOrderSide.BUY:
            limit_price = round(last_price * (1 + self._buffer), 2)
        else:
            limit_price = round(last_price * (1 - self._buffer), 2)

        # ── Submit to broker with retry ───────────────────────────────
        try:
            broker_order = await self._broker.submit_marketable_limit_order(
                symbol=signal.symbol,
                side=broker_side,
                qty=qty,
                limit_price=limit_price,
                client_order_id=order_id,
            )

            # Record SENT event
            await self._ledger.record_event(
                order_id=order_id,
                symbol=signal.symbol,
                event_type=OrderEventType.SENT,
                payload={
                    "broker_order_id": broker_order.broker_order_id,
                    "limit_price": limit_price,
                },
                strategy=signal.strategy_name,
            )

            self._sm.transition(
                order_id,
                OrderState.SENT,
                broker_order_id=broker_order.broker_order_id,
            )

            # Publish submission event
            await self._bus.publish(
                topics.ORDER_SUBMITTED,
                {
                    "order_id": order_id,
                    "broker_order_id": broker_order.broker_order_id,
                    "symbol": signal.symbol,
                    "side": signal.side.value,
                    "qty": qty,
                    "limit_price": limit_price,
                    "strategy_name": signal.strategy_name,
                },
            )

            logger.info(
                "[router] Submitted: %s %s x%d @ %.2f  order_id=%s  broker=%s",
                signal.side.value.upper(),
                signal.symbol,
                qty,
                limit_price,
                order_id,
                broker_order.broker_order_id,
            )
            return broker_order.broker_order_id

        except Exception as exc:
            logger.error("[router] Submit failed for %s: %s", signal.symbol, exc)

            await self._ledger.record_event(
                order_id=order_id,
                symbol=signal.symbol,
                event_type=OrderEventType.REJECTED,
                payload={"error": str(exc)},
                strategy=signal.strategy_name,
            )

            self._sm.transition(order_id, OrderState.REJECTED, error=str(exc))

            await self._bus.publish(
                topics.ORDER_REJECTED,
                {
                    "order_id": order_id,
                    "symbol": signal.symbol,
                    "reason": str(exc),
                    "strategy_name": signal.strategy_name,
                },
            )
            return None

    async def cancel_open_order(self, order_id: str) -> bool:
        """Cancel a previously submitted order by our internal order_id."""
        record = self._sm.get(order_id)
        if record is None or record.broker_order_id is None:
            return False

        success = await self._broker.cancel_order(record.broker_order_id)
        if success:
            self._sm.transition(order_id, OrderState.CANCELED)
            await self._ledger.record_event(
                order_id=order_id,
                symbol=record.symbol,
                event_type=OrderEventType.CANCELED,
                payload={"reason": "manual_cancel"},
                strategy=record.strategy_name,
            )
        return success
