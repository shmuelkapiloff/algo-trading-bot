"""
Positions page — detailed view of all open and recently closed positions.
"""

from __future__ import annotations

import json

import streamlit as st

from dashboard.app import load_positions, load_recent_trades


def render():
    st.header("Positions")

    st.subheader("Open Positions")
    df = load_positions()
    if df.empty:
        st.info("No open positions.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Recent Fills")
    trades = load_recent_trades(20)
    if not trades.empty:
        fills = trades[trades.get("event_type", pd.Series(dtype=str)) == "filled"]
        if not fills.empty:
            st.dataframe(fills, use_container_width=True, hide_index=True)
        else:
            st.info("No recent fills.")
    else:
        st.info("No trade history.")


try:
    import pandas as pd
except ImportError:
    pass

if __name__ == "__main__":
    render()
