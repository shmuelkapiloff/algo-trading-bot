"""
Database models and async engine setup.

Design rules (from TRADING_BOT_PLAN.md):
- Dev: sqlite+aiosqlite:///trading.db  — async-native, never blocking sqlite://
- Prod: postgresql+asyncpg://user:pass@host/db
- Always use create_async_engine(); never create_engine() (sync)
- Schema managed via Alembic; do NOT call create_all() in production
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///trading.db")

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    # pool_pre_ping keeps long-lived connections alive
    pool_pre_ping=True,
)

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class PriceBar(Base):
    """OHLCV bars — split-adjusted. Primary time-series table."""

    __tablename__ = "price_bars"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(16), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    open_adj = Column(Float, nullable=False)
    high_adj = Column(Float, nullable=False)
    low_adj = Column(Float, nullable=False)
    close_adj = Column(Float, nullable=False)
    close_raw = Column(Float, nullable=True)  # audit — pre-adjustment
    volume = Column(BigInteger, nullable=False)
    source = Column(String(32), nullable=False, default="alpaca")

    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_price_bars_symbol_ts"),
        Index("ix_price_bars_symbol_ts", "symbol", "timestamp"),
    )


class Position(Base):
    """Open and closed positions."""

    __tablename__ = "positions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(16), nullable=False, index=True)
    side = Column(String(8), nullable=False)  # "long" | "short"
    qty = Column(Float, nullable=False)
    avg_entry_price = Column(Float, nullable=False)
    stop_loss_price = Column(Float, nullable=True)
    take_profit_price = Column(Float, nullable=True)
    strategy = Column(String(64), nullable=True)
    opened_at = Column(DateTime(timezone=True), nullable=False)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    realized_pnl = Column(Float, nullable=True)
    is_open = Column(Boolean, nullable=False, default=True, index=True)


class OrderRecord(Base):
    """Append-only OMS ledger — one row per order event."""

    __tablename__ = "order_records"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    order_id = Column(String(64), nullable=False, index=True)
    client_order_id = Column(String(64), nullable=True)
    symbol = Column(String(16), nullable=False)
    side = Column(String(8), nullable=False)
    qty = Column(Float, nullable=False)
    filled_qty = Column(Float, nullable=False, default=0.0)
    avg_fill_price = Column(Float, nullable=True)
    state = Column(String(32), nullable=False)
    strategy = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)
    idempotency_key = Column(String(128), nullable=True, unique=True)
    seq = Column(BigInteger, nullable=True)  # monotonic sequence per order_id

    __table_args__ = (Index("ix_order_records_order_id", "order_id"),)


class TradeRecord(Base):
    """Completed fills — used for P&L, TCA, and MAE/MFE tracking."""

    __tablename__ = "trade_records"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    order_id = Column(String(64), nullable=False, index=True)
    symbol = Column(String(16), nullable=False, index=True)
    side = Column(String(8), nullable=False)
    qty = Column(Float, nullable=False)
    fill_price = Column(Float, nullable=False)
    commission = Column(Float, nullable=False, default=0.0)
    slippage_bps = Column(Float, nullable=True)
    strategy = Column(String(64), nullable=True)
    filled_at = Column(DateTime(timezone=True), nullable=False)
    realized_pnl = Column(Float, nullable=True)


class Sp500PitSnapshot(Base):
    """Point-in-time S&P 500 constituent table (prevents look-ahead in backtest)."""

    __tablename__ = "sp500_constituents_pit"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    as_of_date = Column(String(10), nullable=False)   # YYYY-MM-DD
    symbol = Column(String(16), nullable=False)
    added_date = Column(String(10), nullable=True)
    removed_date = Column(String(10), nullable=True)  # NULL = still member
    source = Column(String(32), nullable=False, default="manual")

    __table_args__ = (
        UniqueConstraint("as_of_date", "symbol", name="uq_sp500_pit"),
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


async def get_session() -> AsyncSession:  # type: ignore[return]
    """Dependency injection helper — yields a session."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    """Create all tables (dev only — in prod use Alembic migrations)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
