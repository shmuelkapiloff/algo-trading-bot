"""
Event Loop Watchdog — detects frozen asyncio event loops.

Phase 1 implementation: a background daemon thread sends a heartbeat
coroutine to the main event loop every N seconds. If the loop doesn't
respond within the timeout, it's considered frozen and a HALTED transition
is forced via Redis.

Why a thread? A coroutine scheduled on a frozen loop will never execute.
Only a thread can observe that the loop is unresponsive.

Usage
-----
    watchdog = EventLoopWatchdog(loop, state_store, redis_client)
    watchdog.start()   # launches daemon thread
    # ... at shutdown:
    watchdog.stop()
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL = 10.0  # seconds between heartbeats
_DEFAULT_TIMEOUT = 30.0  # seconds before declaring loop frozen


class EventLoopWatchdog:
    """
    Daemon thread that sends periodic heartbeats to the asyncio event loop.

    If the loop fails to respond within `timeout_seconds`, the watchdog
    forces the trading state to HALTED (fail-safe) and logs a critical alert.

    Parameters
    ----------
    loop             : The running asyncio event loop
    state_store      : RuntimeStateStore (for forced HALTED transition)
    interval_seconds : How often to send a heartbeat (default 10s)
    timeout_seconds  : How long to wait for loop response (default 30s)
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        state_store,  # RuntimeStateStore
        interval_seconds: float = _DEFAULT_INTERVAL,
        timeout_seconds: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._loop = loop
        self._store = state_store
        self._interval = interval_seconds
        self._timeout = timeout_seconds
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the watchdog daemon thread."""
        self._thread = threading.Thread(
            target=self._run,
            name="EventLoopWatchdog",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "EventLoopWatchdog started (interval=%.0fs  timeout=%.0fs)",
            self._interval,
            self._timeout,
        )

    def stop(self) -> None:
        """Signal the watchdog to stop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("EventLoopWatchdog stopped")

    # ------------------------------------------------------------------
    # Internal — runs in daemon thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            alive = self._ping_loop()
            if not alive:
                self._on_loop_frozen()
            self._stop_event.wait(timeout=self._interval)

    def _ping_loop(self) -> bool:
        """
        Submit a no-op coroutine to the event loop and wait up to
        `_timeout` seconds for it to complete.
        Returns True if the loop is alive, False if frozen.
        """
        future: asyncio.Future = asyncio.run_coroutine_threadsafe(
            self._heartbeat_coro(), self._loop
        )
        try:
            future.result(timeout=self._timeout)
            logger.debug("EventLoopWatchdog: heartbeat OK")
            return True
        except Exception as exc:
            logger.critical(
                "EventLoopWatchdog: loop did not respond within %.0fs — loop may be frozen (%s)",
                self._timeout,
                exc,
            )
            return False

    @staticmethod
    async def _heartbeat_coro() -> None:
        """Trivial coroutine — just yields once so we know the loop is alive."""
        await asyncio.sleep(0)

    def _on_loop_frozen(self) -> None:
        """
        Emergency response: force-halt trading state via a new event loop
        (since the main loop is frozen, we create a temporary one for Redis).
        """
        logger.critical(
            "EventLoopWatchdog: LOOP FROZEN — forcing HALTED state via emergency loop"
        )
        try:
            loop = asyncio.new_event_loop()
            from src.runtime_state import TradingState

            loop.run_until_complete(
                self._store.force_transition_internal(
                    target=TradingState.HALTED,
                    reason="watchdog_frozen_loop",
                )
            )
            loop.close()
            logger.critical("EventLoopWatchdog: HALTED state forced successfully")
        except Exception as exc:
            logger.critical("EventLoopWatchdog: failed to force HALTED state: %s", exc)
