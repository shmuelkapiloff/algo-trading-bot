"""Initial schema — OMS trade events table.

Revision ID: 001_initial_schema
Revises: —
Create Date: 2024-01-01 00:00:00.000000 UTC
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "trade_events",
        sa.Column("id", sa.Integer(), nullable=False, autoincrement=True),
        sa.Column("order_id", sa.String(36), nullable=False),
        sa.Column("symbol", sa.String(10), nullable=False),
        sa.Column("event_type", sa.String(20), nullable=False),
        sa.Column("strategy", sa.String(64), nullable=True),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("payload", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("order_id", "event_type", name="uq_order_event"),
    )
    op.create_index("ix_trade_events_symbol", "trade_events", ["symbol"])
    op.create_index("ix_trade_events_occurred_at", "trade_events", ["occurred_at"])
    op.create_index("ix_trade_events_order_id", "trade_events", ["order_id"])


def downgrade() -> None:
    op.drop_table("trade_events")
