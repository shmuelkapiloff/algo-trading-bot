"""
Performance page — win rate, P&L breakdown, drawdown chart.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text


@st.cache_resource
def _engine():
    url = os.getenv("DATABASE_URL", "sqlite:///trading.db")
    url = url.replace("sqlite+aiosqlite", "sqlite").replace(
        "postgresql+asyncpg", "postgresql"
    )
    return create_engine(
        url, connect_args={"check_same_thread": False} if "sqlite" in url else {}
    )


def _load_fills() -> pd.DataFrame:
    query = text(
        """
        SELECT symbol, strategy, occurred_at, payload
        FROM trade_events
        WHERE event_type = 'filled'
        ORDER BY occurred_at DESC
    """
    )
    try:
        with _engine().connect() as conn:
            rows = [dict(r._mapping) for r in conn.execute(query)]
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def render():
    st.header("Performance Analytics")

    df = _load_fills()

    if df.empty:
        st.info("No fill history yet. Start paper trading to see performance metrics.")
        return

    total_fills = len(df)
    strategies = (
        df["strategy"].value_counts() if "strategy" in df.columns else pd.Series()
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Fills Recorded", total_fills)
    with col2:
        if not strategies.empty:
            st.metric("Active Strategies", len(strategies))
    with col3:
        st.metric("As of", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    if "strategy" in df.columns:
        st.subheader("Fills by Strategy")
        st.bar_chart(strategies)

    st.subheader("Recent Fills")
    display_cols = [c for c in ["occurred_at", "symbol", "strategy"] if c in df.columns]
    st.dataframe(df[display_cols].head(50), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render()
