"""
Order Manager — lifecycle management for open orders.

Responsibilities:
  - Submit new orders via the broker adapter
  - Track open orders in memory (order_id → OrderRecord)
  - Retry failed submissions (exponential backoff: 1s, 2s, 4s)
  - Cancel stale orders (TTL exceeded without fill)
  - Route fill/reject events from broker websocket to OrderStateMachine

Backed by the OMS ledger (database.OrderRecord) for persistence.
In-memory state is rebuilt from the DB on startup via reconcile.py.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

_MAX_RETRY_ATTEMPTS = 3
_BASE_RETRY_DELAY = 1.0  # seconds — doubles each attempt


class OrderManagerError(Exception):
    pass


class OrderManager:
    """
    Manages the lifecycle of live orders.

    Parameters
    ----------
    broker : async broker adapter with submit_order() / cancel_order()
    state_machine_factory : callable(order_id) → OrderStateMachine
    on_fill : async callback invoked on FILLED event
    on_reject : async callback invoked on REJECTED event
    """

    def __init__(
        self,
        broker: Any,
        state_machine_factory: Callable[[str], Any] | None = None,
        on_fill: Callable[..., Coroutine] | None = None,
        on_reject: Callable[..., Coroutine] | None = None,
        max_retry_attempts: int = _MAX_RETRY_ATTEMPTS,
    ) -> None:
        self._broker = broker
        self._sm_factory = state_machine_factory
        self._on_fill = on_fill
        self._on_reject = on_reject
        self._max_retries = max_retry_attempts
        self._open_orders: dict[str, dict] = {}  # order_id → metadata

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    async def submit(self, order: dict) -> Optional[str]:
        """
        Submit order to broker with exponential backoff retries.
        Returns broker order_id on success, None on permanent failure.
        """
        for attempt in range(self._max_retries):
            try:
                result = await self._broker.submit_order(order)
                order_id = result.get("order_id") or result.get("id")
                if order_id:
                    self._open_orders[order_id] = {
                        **order,
                        "order_id": order_id,
                        "submitted_at": datetime.now(timezone.utc),
                        "attempt": attempt + 1,
                    }
                    logger.info("order_manager.submit order_id=%s symbol=%s attempt=%d",
                                order_id, order.get("symbol"), attempt + 1)
                    return order_id
            except Exception as exc:  # noqa: BLE001
                delay = _BASE_RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    "order_manager.submit failed attempt=%d symbol=%s error=%s retry_in=%.1fs",
                    attempt + 1, order.get("symbol"), exc, delay,
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(delay)
        logger.error("order_manager.submit permanent_failure symbol=%s", order.get("symbol"))
        return None

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    async def cancel(self, order_id: str) -> bool:
        """Cancel an open order. Returns True on success."""
        try:
            await self._broker.cancel_order(order_id)
            self._open_orders.pop(order_id, None)
            logger.info("order_manager.cancel order_id=%s", order_id)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("order_manager.cancel failed order_id=%s error=%s", order_id, exc)
            return False

    async def cancel_stale(self, max_age_seconds: float = 300.0) -> list[str]:
        """Cancel all orders older than max_age_seconds. Returns list of cancelled IDs."""
        now = datetime.now(timezone.utc)
        stale = [
            oid for oid, meta in self._open_orders.items()
            if (now - meta["submitted_at"]).total_seconds() > max_age_seconds
        ]
        cancelled = []
        for oid in stale:
            if await self.cancel(oid):
                cancelled.append(oid)
        return cancelled

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def handle_fill(self, event: dict) -> None:
        """Called when a fill event arrives from the broker websocket."""
        order_id = event.get("order_id") or event.get("id")
        self._open_orders.pop(order_id, None)
        if self._on_fill:
            await self._on_fill(event)

    async def handle_reject(self, event: dict) -> None:
        """Called when a reject event arrives from the broker websocket."""
        order_id = event.get("order_id") or event.get("id")
        self._open_orders.pop(order_id, None)
        logger.warning("order_manager.rejected order_id=%s reason=%s",
                       order_id, event.get("reason"))
        if self._on_reject:
            await self._on_reject(event)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @property
    def open_order_ids(self) -> list[str]:
        return list(self._open_orders.keys())

    def get_order(self, order_id: str) -> Optional[dict]:
        return self._open_orders.get(order_id)

    def __len__(self) -> int:
        return len(self._open_orders)
