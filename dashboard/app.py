"""
AlgoTrader Pro — Streamlit Dashboard (read-only monitoring).

IMPORTANT: This process is completely isolated from the trading engine.
It reads from PostgreSQL/SQLite (via SQLAlchemy sync) and Redis ONLY.
It never imports from src/ directly and never writes to any shared state.

Architecture (locked decision #10):
  - Separate OS process (launched by docker-compose or directly)
  - No shared memory, no direct function calls to trading engine
  - Data access: DB read-only queries + Redis GET only
  - Cannot submit orders or modify trading state

Launch:
  streamlit run dashboard/app.py --server.port 8501

Environment variables required:
  DATABASE_URL : e.g. sqlite:///trading.db  (sync URL, NOT aiosqlite)
  REDIS_URL    : e.g. redis://localhost:6379/0
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta

import pandas as pd
import redis
import streamlit as st
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AlgoTrader Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Connection helpers (cached per session)
# ---------------------------------------------------------------------------


@st.cache_resource
def _get_db_engine():
    """Create a synchronous SQLAlchemy engine for dashboard read queries."""
    url = os.getenv("DATABASE_URL", "sqlite:///trading.db")
    # Strip async driver prefix if accidentally set (aiosqlite → sqlite, asyncpg → postgresql)
    url = url.replace("sqlite+aiosqlite", "sqlite").replace(
        "postgresql+asyncpg", "postgresql"
    )
    return create_engine(
        url, connect_args={"check_same_thread": False} if "sqlite" in url else {}
    )


@st.cache_resource
def _get_redis():
    """Create a synchronous Redis client for dashboard reads."""
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.from_url(url, decode_responses=True)


def _redis_get(key: str) -> str | None:
    try:
        return _get_redis().get(key)
    except Exception:
        return None


def _redis_keys(pattern: str) -> list[str]:
    try:
        return _get_redis().keys(pattern)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_positions() -> pd.DataFrame:
    """Load open positions from Redis (algotrader:positions:*)."""
    keys = _redis_keys("algotrader:positions:*")
    rows = []
    for k in keys:
        raw = _redis_get(k)
        if raw:
            try:
                pos = json.loads(raw)
                rows.append(pos)
            except json.JSONDecodeError:
                pass
    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "side",
                "qty",
                "avg_entry_price",
                "stop_distance_pct",
                "strategy_name",
                "opened_at",
            ]
        )
    df = pd.DataFrame(rows)
    return df


def load_trading_state() -> str:
    return _redis_get("algotrader:trading_state") or "unknown"


def load_equity() -> float:
    raw = _redis_get("algotrader:equity")
    try:
        return float(raw) if raw else 0.0
    except (ValueError, TypeError):
        return 0.0


def load_pdt_count() -> int:
    raw = _redis_get("algotrader:pdt_daytrades")
    try:
        return int(raw) if raw else 0
    except (ValueError, TypeError):
        return 0


def load_recent_trades(limit: int = 50) -> pd.DataFrame:
    """Load recent trade events from the OMS ledger."""
    engine = _get_db_engine()
    query = text(
        """
        SELECT order_id, symbol, event_type, strategy, occurred_at, payload
        FROM trade_events
        ORDER BY occurred_at DESC
        LIMIT :limit
    """
    )
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"limit": limit})
            rows = [dict(r._mapping) for r in result]
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            return df
    except Exception as e:
        st.warning(f"Could not load trade events: {e}")
        return pd.DataFrame()


def load_daily_pnl() -> pd.DataFrame:
    """Aggregate realized P&L by day from FILLED events."""
    engine = _get_db_engine()
    query = text(
        """
        SELECT
            DATE(occurred_at) as trade_date,
            COUNT(*) as fills,
            SUM(CAST(JSON_EXTRACT(payload, '$.fill_price') AS REAL) *
                CAST(JSON_EXTRACT(payload, '$.filled_qty') AS REAL)) as gross_notional
        FROM trade_events
        WHERE event_type = 'filled'
        GROUP BY DATE(occurred_at)
        ORDER BY trade_date DESC
        LIMIT 30
    """
    )
    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            rows = [dict(r._mapping) for r in result]
            return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("AlgoTrader Pro")
st.sidebar.caption("Read-only monitoring dashboard")

auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
if auto_refresh:
    import time

    time.sleep(0.1)
    st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh now"):
    st.rerun()

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title("📈 AlgoTrader Pro — Live Monitor")

# ── Row 1: Key metrics ────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)

equity = load_equity()
state = load_trading_state()
pdt_count = load_pdt_count()
positions_df = load_positions()
n_positions = len(positions_df)

STATE_COLORS = {
    "active": "🟢",
    "paused": "🟡",
    "close_only": "🟠",
    "halted": "🔴",
    "unknown": "⚪",
}

with col1:
    st.metric("Portfolio Equity", f"${equity:,.2f}")
with col2:
    icon = STATE_COLORS.get(state.lower(), "⚪")
    st.metric("Trading State", f"{icon} {state.upper()}")
with col3:
    st.metric("Open Positions", n_positions)
with col4:
    st.metric("PDT Day Trades (5d)", f"{pdt_count} / 3")

st.markdown("---")

# ── Row 2: Positions table ────────────────────────────────────────────────

st.subheader("Open Positions")

if positions_df.empty:
    st.info("No open positions.")
else:
    # Compute unrealized P&L column if possible (requires current price from DB)
    display_cols = [
        "symbol",
        "side",
        "qty",
        "avg_entry_price",
        "stop_distance_pct",
        "strategy_name",
        "opened_at",
    ]
    display_cols = [c for c in display_cols if c in positions_df.columns]
    st.dataframe(
        positions_df[display_cols].rename(
            columns={
                "symbol": "Symbol",
                "side": "Side",
                "qty": "Qty",
                "avg_entry_price": "Avg Entry",
                "stop_distance_pct": "Stop %",
                "strategy_name": "Strategy",
                "opened_at": "Opened At",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("---")

# ── Row 3: Daily P&L + Recent trades ─────────────────────────────────────

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Daily Activity (last 30 days)")
    pnl_df = load_daily_pnl()
    if pnl_df.empty:
        st.info("No trade history yet.")
    else:
        st.dataframe(pnl_df, use_container_width=True, hide_index=True)

with col_right:
    st.subheader("Recent Trade Events")
    trades_df = load_recent_trades(50)
    if trades_df.empty:
        st.info("No trade events recorded yet.")
    else:
        display_trade_cols = [
            c
            for c in ["occurred_at", "symbol", "event_type", "strategy"]
            if c in trades_df.columns
        ]
        st.dataframe(
            trades_df[display_trade_cols],
            use_container_width=True,
            hide_index=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} | "
    "Read-only — cannot submit orders from this dashboard."
)
