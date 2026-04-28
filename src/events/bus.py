"""
In-process event bus — Phase 1.

Phase 1 uses a single asyncio.Queue per topic. All consumers of a topic
share the same queue (fan-out via registered handlers list).

Phase 2 upgrade path
--------------------
Replace EventBus.publish() with `await redis.xadd(topic, payload)` and
replace the consumer loop with `await redis.xread(...)`. The handler
registration interface does not change.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

Handler = Callable[[str, dict[str, Any]], Awaitable[None]]


class EventBus:
    """
    Lightweight in-process pub/sub.

    Usage
    -----
    bus = EventBus()
    bus.subscribe(topics.ORDER_FILLED, my_handler)
    await bus.publish(topics.ORDER_FILLED, {"order_id": "...", ...})
    await bus.start()   # start consuming (call once at startup)
    await bus.stop()    # graceful drain on shutdown
    """

    def __init__(self, queue_size: int = 512) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue(
            maxsize=queue_size
        )
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Register a coroutine handler for a topic."""
        self._handlers[topic].append(handler)
        logger.debug("EventBus: subscribed %s → %s", topic, handler.__name__)

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        """
        Enqueue an event. Non-blocking — raises QueueFull if bus is saturated.
        Callers should catch asyncio.QueueFull and log/alert.
        """
        await self._queue.put((topic, payload))
        logger.debug("EventBus: published topic=%s", topic)

    def publish_nowait(self, topic: str, payload: dict[str, Any]) -> None:
        """Fire-and-forget publish (drops silently if queue is full)."""
        try:
            self._queue.put_nowait((topic, payload))
        except asyncio.QueueFull:
            logger.warning("EventBus queue full — dropped event: topic=%s", topic)

    async def start(self) -> None:
        """Start the background consumer task."""
        self._task = asyncio.get_event_loop().create_task(
            self._consume_loop(), name="event_bus_consumer"
        )
        logger.info("EventBus started")

    async def stop(self) -> None:
        """
        Signal the consumer to drain and exit.
        Waits up to 5 seconds for the queue to empty.
        """
        self._stop_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("EventBus consumer did not drain in 5s — cancelling")
                self._task.cancel()
        logger.info("EventBus stopped")

    # ------------------------------------------------------------------
    # Consumer loop
    # ------------------------------------------------------------------

    async def _consume_loop(self) -> None:
        """
        Process events from the queue until stop is signalled AND the
        queue is empty (graceful drain).
        """
        while True:
            try:
                topic, payload = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                if self._stop_event.is_set():
                    break
                await asyncio.sleep(0.01)
                continue

            handlers = self._handlers.get(topic, [])
            if not handlers:
                logger.debug("EventBus: no handlers for topic=%s", topic)
                self._queue.task_done()
                continue

            for handler in handlers:
                try:
                    await handler(topic, payload)
                except Exception:
                    logger.exception(
                        "EventBus: handler %s raised for topic=%s",
                        handler.__name__,
                        topic,
                    )
            self._queue.task_done()
