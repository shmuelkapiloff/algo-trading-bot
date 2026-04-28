"""
OMS Event Ledger — Phase 1 (Lean).

Append-only write path for order lifecycle events. All writes are idempotent:
inserting a duplicate (order_id, event_type) pair is silently ignored via
ON CONFLICT DO NOTHING. This means every caller can safely retry without
checking for prior state.

Architecture
------------
- OmsLedger owns *one* async_sessionmaker and creates a session per write.
  Do not share sessions across calls.
- record_event() is the only public write method. It inserts one row and
  returns the event_id that was persisted (or the pre-existing one on
  conflict).
- flush() drains any in-flight async context and is called by the graceful
  shutdown handler (src/shutdown.py) before the process exits.
- get_order_events() is the read path used by the reconcile job and dashboard.

Usage
-----
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from src.data.oms_ledger import OmsLedger
    from src.data.models import Base

    engine = create_async_engine(DATABASE_URL)
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    ledger = OmsLedger(async_session)

    event_id = await ledger.record_event(
        order_id="ord-123",
        symbol="AAPL",
        event_type=OrderEventType.FILLED,
        payload={"fill_qty": 10, "avg_px": 182.50},
        strategy="momentum",
    )
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .models import OrderEventType, TradeEvent

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class OmsLedger:
    """
    Append-only OMS event ledger.

    Thread-safe: each method creates its own session.
    All writes are idempotent (ON CONFLICT DO NOTHING on the
    UNIQUE(order_id, event_type) constraint).
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def record_event(
        self,
        order_id: str,
        symbol: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        strategy: str | None = None,
        occurred_at: datetime | None = None,
        event_id: str | None = None,
    ) -> str:
        """
        Persist one order lifecycle event.

        Returns the event_id that was written (or that already existed for
        this (order_id, event_type) pair — idempotent on conflict).

        Parameters
        ----------
        order_id    : Broker or locally-generated order identifier.
        symbol      : Ticker (e.g. "AAPL").
        event_type  : One of OrderEventType.* constants.
        payload     : Broker-specific fields dict (fill_qty, avg_px, …).
        strategy    : Strategy name (optional; used for Phase 2 attribution).
        occurred_at : When the event was observed. Defaults to now(UTC).
        event_id    : Supply a pre-generated UUID for caller-controlled
                      idempotency keys. Defaults to a new UUID v4.

        Raises
        ------
        ValueError  : If event_type is not a recognised OrderEventType value.
        """
        if event_type not in OrderEventType.ALL:
            raise ValueError(
                f"Unknown event_type {event_type!r}. "
                f"Must be one of: {sorted(OrderEventType.ALL)}"
            )

        eid = event_id or str(uuid.uuid4())
        now = occurred_at or _utcnow()
        data = payload or {}

        async with self._session_factory() as session:
            async with session.begin():
                # Detect dialect to choose the correct INSERT … ON CONFLICT syntax.
                dialect = session.bind.dialect.name  # type: ignore[union-attr]

                if dialect == "postgresql":
                    stmt = (
                        pg_insert(TradeEvent)
                        .values(
                            event_id=eid,
                            order_id=order_id,
                            symbol=symbol,
                            event_type=event_type,
                            payload=data,
                            occurred_at=now,
                            created_at=_utcnow(),
                            strategy=strategy,
                        )
                        .on_conflict_do_nothing(
                            index_elements=["order_id", "event_type"]
                        )
                        .returning(TradeEvent.event_id)
                    )
                    result = await session.execute(stmt)
                    row = result.scalar_one_or_none()
                    if row is None:
                        # Conflict: row already existed — fetch the existing event_id
                        existing = await session.execute(
                            select(TradeEvent.event_id).where(
                                TradeEvent.order_id == order_id,
                                TradeEvent.event_type == event_type,
                            )
                        )
                        eid = existing.scalar_one()
                        logger.debug(
                            "OMS dedup: (order_id=%s, event_type=%s) already exists, "
                            "returning existing event_id=%s",
                            order_id,
                            event_type,
                            eid,
                        )
                    else:
                        eid = row

                else:
                    # SQLite: INSERT OR IGNORE (aiosqlite dev path)
                    stmt = (  # type: ignore[assignment]
                        sqlite_insert(TradeEvent)
                        .values(
                            event_id=eid,
                            order_id=order_id,
                            symbol=symbol,
                            event_type=event_type,
                            payload=data,
                            occurred_at=now,
                            created_at=_utcnow(),
                            strategy=strategy,
                        )
                        .prefix_with("OR IGNORE")
                    )
                    await session.execute(stmt)

                    # Check if our row landed or a prior one exists
                    existing = await session.execute(
                        select(TradeEvent.event_id).where(
                            TradeEvent.order_id == order_id,
                            TradeEvent.event_type == event_type,
                        )
                    )
                    eid = existing.scalar_one()

        logger.debug(
            "OMS record_event: order_id=%s event_type=%s symbol=%s event_id=%s",
            order_id,
            event_type,
            symbol,
            eid,
        )
        return eid

    # ------------------------------------------------------------------
    # Read (reconcile + dashboard)
    # ------------------------------------------------------------------

    async def get_order_events(self, order_id: str) -> Sequence[TradeEvent]:
        """
        Return all events for a given order_id in chronological order.

        Used by the reconcile job and the Streamlit dashboard (read-only path).
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(TradeEvent)
                .where(TradeEvent.order_id == order_id)
                .order_by(TradeEvent.occurred_at)
            )
            return result.scalars().all()

    async def get_latest_event_type(self, order_id: str) -> str | None:
        """
        Return the event_type of the most recent event for an order.

        Returns None if the order_id has no events (unknown order).
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(TradeEvent.event_type)
                .where(TradeEvent.order_id == order_id)
                .order_by(TradeEvent.occurred_at.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def flush(self) -> None:
        """
        No-op for the session-per-call pattern (sessions auto-commit on
        context-manager exit). Provided so shutdown.py can call
        `await ledger.flush()` without needing to know the implementation.

        If you switch to a connection-pool-level write buffer in the future,
        implement the drain logic here.
        """
        logger.info(
            "OmsLedger.flush() called — no pending writes (session-per-call pattern)"
        )
