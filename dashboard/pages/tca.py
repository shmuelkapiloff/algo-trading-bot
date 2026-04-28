"""TCA Dashboard Page — Transaction Cost Analysis.

Displays:
  - Fill rate % (filled qty / submitted qty)
  - Average slippage in bps vs VWAP benchmark
  - % orders filled within 30 seconds
  - Throttle status (green/yellow/red based on TCA circuit breaker)
  - Recent trades table with per-trade slippage
  - Slippage trend chart over last 30 days

Data source: ``tca_records`` table populated by src/monitoring/tca.py.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# TCA alert thresholds (mirror src/monitoring/tca.py config)
_WARN_SLIPPAGE_BPS = 10.0
_PAUSE_SLIPPAGE_BPS = 25.0
_WARN_FILL_RATE = 0.85
_WARN_30S_FILL_PCT = 0.70


@st.cache_resource
def _engine():
    url = os.getenv("DATABASE_URL", "sqlite:///trading.db")
    url = url.replace("sqlite+aiosqlite", "sqlite").replace(
        "postgresql+asyncpg", "postgresql"
    )
    return create_engine(
        url, connect_args={"check_same_thread": False} if "sqlite" in url else {}
    )


def _load_tca_data(days: int = 30) -> pd.DataFrame:
    """Load recent TCA records from the database."""
    since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    query = text(
        """
        SELECT
            symbol,
            order_id,
            side,
            strategy_name,
            fill_rate,
            slippage_bps,
            fill_latency_ms,
            filled_qty,
            submitted_qty,
            recorded_at
        FROM tca_records
        WHERE recorded_at >= :since
        ORDER BY recorded_at DESC
        LIMIT 500
        """
    )
    try:
        with _engine().connect() as conn:
            rows = [dict(r._mapping) for r in conn.execute(query, {"since": since})]
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _load_throttle_status() -> dict:
    """Load TCA circuit breaker status from Redis (or return defaults)."""
    try:
        import redis as _redis  # type: ignore

        r = _redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        throttle_active = r.get("tca:throttle_active")
        pause_active = r.get("tca:pause_active")
        return {
            "throttle": throttle_active == b"1",
            "pause": pause_active == b"1",
        }
    except Exception:
        return {"throttle": False, "pause": False}


def render():
    st.header("Transaction Cost Analysis (TCA)")

    # ── Controls ──────────────────────────────────────────────────────
    col_days, col_refresh = st.columns([3, 1])
    with col_days:
        days = st.slider("Lookback (days)", min_value=1, max_value=90, value=30, step=1)
    with col_refresh:
        if st.button("Refresh"):
            st.cache_data.clear()

    # ── Load data ─────────────────────────────────────────────────────
    df = _load_tca_data(days=days)
    throttle = _load_throttle_status()

    # ── Circuit breaker status banner ─────────────────────────────────
    if throttle.get("pause"):
        st.error(
            "🔴 **TCA CIRCUIT BREAKER ACTIVE** — New orders are PAUSED. "
            "Avg slippage exceeded pause threshold. Manual review required."
        )
    elif throttle.get("throttle"):
        st.warning(
            "🟡 **TCA THROTTLE ACTIVE** — Position sizes reduced by 50%. "
            "Slippage above warn threshold."
        )
    else:
        st.success("🟢 TCA: Normal — All metrics within thresholds.")

    if df.empty:
        st.info("No TCA records found for the selected period.")
        _render_empty_metrics()
        return

    # ── Key metrics ───────────────────────────────────────────────────
    avg_slippage = df["slippage_bps"].mean() if "slippage_bps" in df else 0.0
    avg_fill_rate = df["fill_rate"].mean() if "fill_rate" in df else 1.0

    within_30s = 0.0
    if "fill_latency_ms" in df:
        within_30s = (df["fill_latency_ms"] <= 30_000).mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        color = "normal"
        delta = None
        if avg_slippage > _PAUSE_SLIPPAGE_BPS:
            color = "inverse"
        elif avg_slippage > _WARN_SLIPPAGE_BPS:
            delta = "⚠ Above threshold"
        st.metric(
            "Avg Slippage (bps)",
            f"{avg_slippage:.1f}",
            delta=delta,
            delta_color=color,
        )

    with col2:
        fill_delta = None
        if avg_fill_rate < _WARN_FILL_RATE:
            fill_delta = "⚠ Below threshold"
        st.metric(
            "Fill Rate",
            f"{avg_fill_rate:.1%}",
            delta=fill_delta,
            delta_color="inverse" if avg_fill_rate < _WARN_FILL_RATE else "normal",
        )

    with col3:
        s30_delta = None
        if within_30s < _WARN_30S_FILL_PCT:
            s30_delta = "⚠ Below 70%"
        st.metric(
            "Filled Within 30s",
            f"{within_30s:.1%}",
            delta=s30_delta,
            delta_color="inverse" if within_30s < _WARN_30S_FILL_PCT else "normal",
        )

    with col4:
        st.metric("Total Orders", f"{len(df):,}")

    st.divider()

    # ── Slippage trend chart ──────────────────────────────────────────
    st.subheader("Slippage Trend")
    if "recorded_at" in df.columns and "slippage_bps" in df.columns:
        chart_df = (
            df.copy()
            .assign(recorded_at=pd.to_datetime(df["recorded_at"]))
            .set_index("recorded_at")
            .sort_index()
            .resample("D")["slippage_bps"]
            .mean()
            .dropna()
            .to_frame()
        )
        if not chart_df.empty:
            st.line_chart(chart_df, use_container_width=True)

    # ── Per-strategy breakdown ─────────────────────────────────────────
    if "strategy_name" in df.columns:
        st.subheader("By Strategy")
        strategy_df = (
            df.groupby("strategy_name")
            .agg(
                orders=("order_id", "count"),
                avg_slippage_bps=("slippage_bps", "mean"),
                avg_fill_rate=("fill_rate", "mean"),
            )
            .reset_index()
            .sort_values("avg_slippage_bps", ascending=False)
        )
        st.dataframe(strategy_df, use_container_width=True)

    # ── Recent orders table ───────────────────────────────────────────
    st.subheader(f"Recent Orders (last {days} days)")
    display_cols = [
        c for c in ["recorded_at", "symbol", "side", "strategy_name",
                     "slippage_bps", "fill_rate", "fill_latency_ms", "filled_qty"]
        if c in df.columns
    ]
    st.dataframe(
        df[display_cols].head(100),
        use_container_width=True,
        hide_index=True,
    )


def _render_empty_metrics():
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Slippage (bps)", "—")
    col2.metric("Fill Rate", "—")
    col3.metric("Filled Within 30s", "—")
    col4.metric("Total Orders", "0")
