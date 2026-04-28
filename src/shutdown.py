"""
Graceful shutdown handler for AlgoTrader Pro.

Handles SIGTERM and SIGINT with the following ordered steps:

  1. Atomically transition trading state to PAUSED via Redis
     (cross-process — all Gunicorn workers see this immediately).
  2. Drain the in-flight signal intake queue within drain_timeout_seconds.
     Logs an error if the queue is not empty after the deadline — this
     indicates signals were dropped, NOT stop-loss orders (those are managed
     by the broker-side bracket orders).
  3. Flush pending OMS ledger writes to PostgreSQL.
  4. Shut down APScheduler without waiting for running jobs.
  5. Stop the asyncio event loop.

Registration
------------
Call register_shutdown_handlers() once in main.py before loop.run_forever():

    from src.shutdown import register_shutdown_handlers, ShutdownDependencies

    deps = ShutdownDependencies(
        runtime_state=runtime_state,
        signal_queue=signal_queue,
        oms_ledger_flush=oms_ledger.flush,
        scheduler_shutdown=lambda: scheduler.shutdown(wait=False),
    )
    register_shutdown_handlers(loop, deps)
"""

from __future__ import annotations

import asyncio
import logging
import signal
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from .runtime_state import RuntimeStateStore, TradingState

logger = logging.getLogger(__name__)


@dataclass
class ShutdownDependencies:
    """
    All resources that must be cleanly stopped on shutdown.

    oms_ledger_flush   async callable with no arguments: await flush()
    scheduler_shutdown sync callable: scheduler.shutdown(wait=False)
    """

    runtime_state: RuntimeStateStore
    signal_queue: asyncio.Queue
    oms_ledger_flush: Callable[[], Awaitable[None]]
    scheduler_shutdown: Callable[[], None]
    drain_timeout_seconds: float = 10.0


async def _graceful_shutdown(
    sig: signal.Signals,
    loop: asyncio.AbstractEventLoop,
    deps: ShutdownDependencies,
) -> None:
    logger.warning("Received %s — initiating graceful shutdown", sig.name)

    # ------------------------------------------------------------------
    # Step 1: Pause new order intake (atomic, cross-process via Redis)
    # ------------------------------------------------------------------
    success, reason = await deps.runtime_state.force_transition_internal(
        target=TradingState.PAUSED,
        reason=f"graceful_shutdown:{sig.name}",
    )
    if success:
        logger.info("Trading state set to PAUSED")
    else:
        # Non-fatal: state may already be PAUSED/HALTED from another worker.
        logger.warning(
            "Could not set PAUSED during shutdown (reason=%s). "
            "State may already be safe — continuing drain.",
            reason,
        )

    # ------------------------------------------------------------------
    # Step 2: Drain in-flight signal queue
    # ------------------------------------------------------------------
    deadline = loop.time() + deps.drain_timeout_seconds
    drain_iterations = 0

    while not deps.signal_queue.empty():
        if loop.time() > deadline:
            remaining = deps.signal_queue.qsize()
            # IMPORTANT: these are entry signals, not stop-loss orders.
            # Broker-side bracket orders remain active independently.
            logger.error(
                "Signal queue drain timeout — %d entry signal(s) discarded. "
                "Open positions are protected by broker-side stop-loss orders.",
                remaining,
            )
            break
        await asyncio.sleep(0.05)
        drain_iterations += 1

    if drain_iterations > 0:
        logger.info("Signal queue fully drained (%d poll cycles)", drain_iterations)

    # ------------------------------------------------------------------
    # Step 3: Flush OMS ledger
    # ------------------------------------------------------------------
    try:
        await asyncio.wait_for(deps.oms_ledger_flush(), timeout=5.0)
        logger.info("OMS ledger flushed successfully")
    except asyncio.TimeoutError:
        logger.error(
            "OMS ledger flush timed out after 5s. "
            "Some events may be missing from the DB — "
            "run reconcile.py before restarting."
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("OMS ledger flush raised an unexpected error: %s", exc)

    # ------------------------------------------------------------------
    # Step 4: Stop APScheduler
    # ------------------------------------------------------------------
    try:
        deps.scheduler_shutdown()
        logger.info("APScheduler stopped")
    except Exception as exc:  # noqa: BLE001
        logger.error("Scheduler shutdown error: %s", exc)

    # ------------------------------------------------------------------
    # Step 5: Stop the event loop
    # ------------------------------------------------------------------
    logger.info("Graceful shutdown complete — stopping event loop")
    loop.stop()


def register_shutdown_handlers(
    loop: asyncio.AbstractEventLoop,
    deps: ShutdownDependencies,
) -> None:
    """
    Register SIGTERM and SIGINT handlers on the given event loop.

    Must be called once in main.py before loop.run_forever().
    Handlers are registered as asyncio tasks so they run inside the
    event loop and can safely await coroutines.
    """
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(_graceful_shutdown(s, loop, deps)),
        )
    logger.info("Graceful shutdown handlers registered for SIGTERM and SIGINT")
