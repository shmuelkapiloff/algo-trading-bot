"""
Overview page — portfolio-level summary with equity curve.
This page is imported by dashboard/app.py (multi-page Streamlit).
"""

from __future__ import annotations

import os

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


def render():
    st.header("Portfolio Overview")

    # Load cumulative notional by day
    query = text(
        """
        SELECT
            DATE(occurred_at) as trade_date,
            COUNT(*) as n_fills
        FROM trade_events
        WHERE event_type = 'filled'
        GROUP BY DATE(occurred_at)
        ORDER BY trade_date ASC
    """
    )
    try:
        with _engine().connect() as conn:
            rows = [dict(r._mapping) for r in conn.execute(query)]
    except Exception:
        rows = []

    if rows:
        df = pd.DataFrame(rows)
        st.line_chart(df.set_index("trade_date")["n_fills"], use_container_width=True)
    else:
        st.info(
            "No trade data available yet. Start paper trading to see the equity curve."
        )


if __name__ == "__main__":
    render()
