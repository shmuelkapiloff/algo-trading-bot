"""
Alembic environment configuration.

Supports both offline (SQL script generation) and online (live DB) modes.
Detects async SQLAlchemy URLs and uses run_sync() wrapper for async engines.
"""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

# Alembic Config object
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import the metadata from our models so Alembic can detect schema changes
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.models import Base  # noqa: E402

target_metadata = Base.metadata

# Database URL: prefer environment variable over alembic.ini
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    config.get_main_option("sqlalchemy.url", "sqlite+aiosqlite:///trading.db"),
)


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode — generate SQL script without
    connecting to a live database. Used for review and CI pipelines.
    """
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode against a live database.
    Uses async engine (asyncpg / aiosqlite) with run_sync wrapper.
    """
    connectable = create_async_engine(DATABASE_URL, poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
