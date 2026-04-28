"""
Database models for AlgoTrader Pro — Phase 1 (Lean).

Single-table OMS event ledger using SQLAlchemy 2.x async-native API.
Compatible with both SQLite (dev) and PostgreSQL + TimescaleDB (prod).

Design choices
--------------
- One table for all order lifecycle events (NEW / ACK / PARTIAL / FILLED /
  REJECTED / CANCELED). No separate Command vs. Event envelope — that
  complexity is deferred to Phase 2 (Redis Streams).
- UNIQUE(order_id, event_type) enforces dedup at the DB layer. Any retry that
  tries to insert the same (order_id, event_type) pair is a silent no-op via
  ON CONFLICT DO NOTHING. No Python-level dedup window needed.
- event_id UUID is the idempotency key for upstream callers.
- payload JSONB stores broker-specific fields (fill_qty, avg_px, reject_reason)
  without schema churn.

Connection setup (main.py)
--------------------------
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from src.data.models import Base, DATABASE_URL_ENV

    engine = create_async_engine(os.environ[DATABASE_URL_ENV], echo=False)
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    # Create tables (dev / first-run only)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

SQLAlchemy URL formats
----------------------
  Dev  (SQLite):    sqlite+aiosqlite:///trading.db
  Prod (Postgres):  postgresql+asyncpg://user:pass@host:5432/algotrader
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Index, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON

# Environment variable name read by main.py
DATABASE_URL_ENV = "DATABASE_URL"


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------


class Base(AsyncAttrs, DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# OrderEventType — the exhaustive set of legal event types
# ---------------------------------------------------------------------------


class OrderEventType:
    NEW = "NEW"  # order constructed locally, not yet sent to broker
    SENT = "SENT"  # submitted to broker API
    ACK = "ACK"  # broker acknowledged receipt
    PARTIAL = "PARTIAL"  # partially filled
    FILLED = "FILLED"  # fully filled
    CANCELED = "CANCELED"  # canceled (our request or broker-initiated)
    REJECTED = "REJECTED"  # broker rejected the order
    EXPIRED = "EXPIRED"  # order expired (e.g. DAY order after close)

    ALL: frozenset[str] = frozenset(
        {NEW, SENT, ACK, PARTIAL, FILLED, CANCELED, REJECTED, EXPIRED}
    )


# ---------------------------------------------------------------------------
# TradeEvent table
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


class TradeEvent(Base):
    """
    Append-only OMS event ledger row.

    One row per (order_id, event_type) transition. The UNIQUE constraint
    makes every write idempotent: ON CONFLICT DO NOTHING means retries
    are always safe.

    Columns
    -------
    event_id    : UUID string — idempotency key for the caller.
    order_id    : UUID string — broker or locally-generated order identifier.
    symbol      : Ticker (e.g. "AAPL"). Stored for fast dashboard queries
                  without JSON parsing.
    event_type  : One of OrderEventType.*
    payload     : Broker-specific fields (fill_qty, avg_px, reject_reason, …)
                  Stored as JSONB on Postgres, JSON on SQLite.
    occurred_at : Event timestamp (UTC). Set by the application, not the DB,
                  so it represents when the event was observed, not when it
                  was persisted.
    created_at  : DB insert timestamp (UTC). Useful for lag monitoring.
    strategy    : Strategy name that generated the original signal. Optional;
                  used for per-strategy P&L attribution in Phase 2.
    """

    __tablename__ = "trade_events"

    event_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=_new_uuid,
        comment="Idempotency key (UUID v4)",
    )
    order_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        comment="Broker or locally-generated order ID",
    )
    symbol: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        comment="Ticker symbol, e.g. AAPL",
    )
    event_type: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="NEW | SENT | ACK | PARTIAL | FILLED | CANCELED | REJECTED | EXPIRED",
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        # JSONB on Postgres (indexed, fast); JSON on SQLite (dev compat)
        JSONB().with_variant(JSON(), "sqlite"),
        nullable=False,
        default=dict,
        comment="Broker-specific event fields",
    )
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        comment="When the event was observed (application time, UTC)",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        comment="DB insert timestamp (UTC) — for lag monitoring",
    )
    strategy: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        default=None,
        comment="Strategy that generated the signal (Phase 2 attribution)",
    )

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    __table_args__ = (
        UniqueConstraint(
            "order_id",
            "event_type",
            name="uq_trade_events_order_event",
        ),
        # Composite index on symbol + occurred_at for time-range dashboard queries
        Index("ix_trade_events_symbol_time", "symbol", "occurred_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<TradeEvent order_id={self.order_id!r} "
            f"event_type={self.event_type!r} "
            f"symbol={self.symbol!r}>"
        )
