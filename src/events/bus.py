"""
In-process event bus — Phase 1 (asyncio.Queue) + Phase 2 (Redis Streams).

Phase selection
---------------
Set the environment variable ``EVENT_BUS_BACKEND``:
  - ``asyncio_queue``  (default): single-process, in-memory queue
  - ``redis_streams``            : durable, at-least-once, cross-process

Use the factory function ``create_event_bus()`` instead of instantiating
the classes directly so the correct backend is selected at startup.

Phase 2 guarantees (Redis Streams backend)
------------------------------------------
- Delivery:    at-least-once (consumer groups + ACK)
- Ordering:    per-stream (topic), monotonic stream IDs
- DLQ:         events.dlq stream after max_retries failures
- Persistence: survives process restart (Redis AOF/RDB)
- Idempotency: handler must be idempotent (event_id check)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections import defaultdict
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

Handler = Callable[[str, dict[str, Any]], Awaitable[None]]

# Dead-letter queue stream name
_DLQ_STREAM = "events.dlq"
# Consumer group name used by this process
_CONSUMER_GROUP = "algotrader_consumers"
# Consumer name (unique per process)
_CONSUMER_NAME = f"consumer_{os.getpid()}"


# ===========================================================================
# Phase 1 — asyncio.Queue backend (original implementation, unchanged)
# ===========================================================================


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


# ===========================================================================
# Phase 2 — Redis Streams backend
# ===========================================================================


class RedisStreamEventBus:
    """Event bus backed by Redis Streams (Phase 2).

    Provides at-least-once delivery, process-restart durability, and
    dead-letter queue (DLQ) semantics.

    Requires ``redis.asyncio`` (installed as part of ``redis[asyncio]``).

    Parameters
    ----------
    redis_url:
        Redis connection URL (default ``redis://localhost:6379/0``).
    max_retries:
        Number of handler retries before routing to DLQ (default 3).
    consumer_parallelism:
        Number of concurrent consumer coroutines (default 4).
    stream_maxlen:
        MAXLEN for each Redis stream (default 10000, approximate trim).
    poll_interval_ms:
        How long XREADGROUP blocks waiting for new messages (default 500ms).
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        max_retries: int = 3,
        consumer_parallelism: int = 4,
        stream_maxlen: int = 10_000,
        poll_interval_ms: int = 500,
    ) -> None:
        self._redis_url = redis_url
        self._max_retries = max_retries
        self._consumer_parallelism = consumer_parallelism
        self._stream_maxlen = stream_maxlen
        self._poll_interval_ms = poll_interval_ms

        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._redis = None  # set in start()
        self._tasks: list[asyncio.Task] = []
        self._stop_event = asyncio.Event()

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Register a coroutine handler for a topic (stream name)."""
        self._handlers[topic].append(handler)
        logger.debug("RedisStreamEventBus: subscribed %s → %s", topic, handler.__name__)

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        """Publish an event to a Redis Stream.

        Each event gets a unique ``event_id`` for idempotency.
        The payload is JSON-serialised and stored as a stream entry field.
        """
        if self._redis is None:
            raise RuntimeError("RedisStreamEventBus not started — call await bus.start() first")

        event_id = str(uuid.uuid4())
        entry = {
            "event_id": event_id,
            "topic": topic,
            "payload": json.dumps(payload),
        }
        await self._redis.xadd(
            topic,
            entry,
            maxlen=self._stream_maxlen,
            approximate=True,
        )
        logger.debug("RedisStreamEventBus: published topic=%s event_id=%s", topic, event_id)

    def publish_nowait(self, topic: str, payload: dict[str, Any]) -> None:
        """Schedule a publish without awaiting (creates an asyncio Task)."""
        loop = asyncio.get_event_loop()
        loop.create_task(self.publish(topic, payload))

    async def start(self) -> None:
        """Connect to Redis, create consumer groups, start consumer tasks."""
        try:
            import redis.asyncio as aioredis  # type: ignore
        except ImportError:
            raise ImportError(
                "redis[asyncio] is required for the Redis Streams backend. "
                "Install with: pip install 'redis[asyncio]'"
            )

        self._redis = aioredis.from_url(self._redis_url)
        logger.info("RedisStreamEventBus: connected to %s", self._redis_url)

        # Ensure consumer groups exist for all subscribed topics
        for topic in list(self._handlers.keys()):
            await self._ensure_consumer_group(topic)

        # Launch consumer workers
        for i in range(self._consumer_parallelism):
            task = asyncio.get_event_loop().create_task(
                self._consume_loop(worker_id=i),
                name=f"redis_event_bus_consumer_{i}",
            )
            self._tasks.append(task)

        logger.info(
            "RedisStreamEventBus started: %d workers, topics=%s",
            self._consumer_parallelism,
            list(self._handlers.keys()),
        )

    async def stop(self) -> None:
        """Gracefully stop all consumer tasks."""
        self._stop_event.set()
        for task in self._tasks:
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()
        if self._redis:
            await self._redis.aclose()
        logger.info("RedisStreamEventBus stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _ensure_consumer_group(self, stream_name: str) -> None:
        """Create the consumer group if it does not already exist."""
        try:
            await self._redis.xgroup_create(
                stream_name,
                _CONSUMER_GROUP,
                id="0",       # start from beginning
                mkstream=True,  # create stream if it doesn't exist
            )
        except Exception as exc:
            # BUSYGROUP = group already exists; that is fine
            if "BUSYGROUP" not in str(exc):
                logger.warning("xgroup_create(%s): %s", stream_name, exc)

    async def _consume_loop(self, worker_id: int) -> None:
        """Consumer loop: XREADGROUP + ACK + DLQ routing."""
        streams = list(self._handlers.keys())
        if not streams:
            return

        consumer_name = f"{_CONSUMER_NAME}_w{worker_id}"
        stream_ids = {s: ">" for s in streams}  # ">" = only new messages

        while not self._stop_event.is_set():
            try:
                results = await self._redis.xreadgroup(
                    groupname=_CONSUMER_GROUP,
                    consumername=consumer_name,
                    streams=stream_ids,
                    count=10,
                    block=self._poll_interval_ms,
                )
            except Exception as exc:
                logger.warning("[redis_bus] xreadgroup error: %s", exc)
                await asyncio.sleep(1.0)
                continue

            if not results:
                continue

            for stream_name, messages in results:
                # redis.asyncio returns bytes or str depending on decode_responses
                sname = stream_name if isinstance(stream_name, str) else stream_name.decode()
                for msg_id, fields in messages:
                    await self._process_message(sname, msg_id, fields)

    async def _process_message(
        self, stream_name: str, msg_id: Any, fields: dict
    ) -> None:
        """Decode and dispatch one stream message to registered handlers."""
        try:
            # Decode bytes if needed
            def _decode(v: Any) -> str:
                return v.decode() if isinstance(v, bytes) else str(v)

            decoded_fields = {
                (_decode(k) if isinstance(k, bytes) else k): _decode(v)
                for k, v in fields.items()
            }

            topic = decoded_fields.get("topic", stream_name)
            payload_raw = decoded_fields.get("payload", "{}")
            payload: dict[str, Any] = json.loads(payload_raw)

        except Exception as exc:
            logger.error("[redis_bus] Failed to decode message %s: %s", msg_id, exc)
            await self._ack(stream_name, msg_id)
            return

        handlers = self._handlers.get(topic, [])
        success = True
        for handler in handlers:
            for attempt in range(self._max_retries):
                try:
                    await handler(topic, payload)
                    break
                except Exception as exc:
                    logger.warning(
                        "[redis_bus] handler %s failed (attempt %d/%d): %s",
                        handler.__name__,
                        attempt + 1,
                        self._max_retries,
                        exc,
                    )
                    if attempt == self._max_retries - 1:
                        success = False
                        await self._send_to_dlq(stream_name, msg_id, fields, str(exc))

        if success:
            await self._ack(stream_name, msg_id)

    async def _ack(self, stream_name: str, msg_id: Any) -> None:
        """ACK a processed message to remove it from the PEL."""
        try:
            await self._redis.xack(stream_name, _CONSUMER_GROUP, msg_id)
        except Exception as exc:
            logger.warning("[redis_bus] XACK failed for %s: %s", msg_id, exc)

    async def _send_to_dlq(
        self, stream_name: str, msg_id: Any, original_fields: dict, error: str
    ) -> None:
        """Route a failed message to the dead-letter queue stream."""
        try:
            dlq_entry = {
                "original_stream": str(stream_name),
                "original_msg_id": str(msg_id),
                "error": error[:500],
                "original_fields": json.dumps({
                    k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
                    for k, v in original_fields.items()
                }),
            }
            await self._redis.xadd(_DLQ_STREAM, dlq_entry, maxlen=1000, approximate=True)
            logger.error(
                "[redis_bus] Message %s routed to DLQ after %d retries",
                msg_id,
                self._max_retries,
            )
        except Exception as exc:
            logger.error("[redis_bus] Failed to write to DLQ: %s", exc)


# ===========================================================================
# Factory — selects backend based on ENV
# ===========================================================================


def create_event_bus(
    backend: str | None = None,
    redis_url: str | None = None,
) -> "EventBus | RedisStreamEventBus":
    """Create the appropriate event bus backend.

    Parameters
    ----------
    backend:
        ``"asyncio_queue"`` or ``"redis_streams"``.
        Defaults to ``EVENT_BUS_BACKEND`` env var, or ``"asyncio_queue"``.
    redis_url:
        Redis URL (used only for redis_streams backend).
        Defaults to ``REDIS_URL`` env var or ``redis://localhost:6379/0``.
    """
    chosen = backend or os.getenv("EVENT_BUS_BACKEND", "asyncio_queue")
    if chosen == "redis_streams":
        url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        logger.info("EventBus: using Redis Streams backend (%s)", url)
        return RedisStreamEventBus(redis_url=url)
    logger.info("EventBus: using asyncio.Queue backend (Phase 1)")
    return EventBus()

