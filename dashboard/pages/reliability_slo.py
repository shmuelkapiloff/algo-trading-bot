"""Reliability SLO Dashboard Page.

Tracks Phase 5.6 reliability SLOs:
  - broker latency p95/p99
  - fill rate
  - reconciliation mismatch count
  - broker failover mode
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

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


def _redis_get(key: str) -> str | None:
    try:
        import redis as _redis  # type: ignore

        r = _redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True
        )
        return r.get(key)
    except Exception:
        return None


def _load_tca_snapshot(days: int = 30) -> pd.DataFrame:
    since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    query = text(
        """
        SELECT
            recorded_at,
            fill_rate,
            fill_latency_ms,
            slippage_bps
        FROM tca_records
        WHERE recorded_at >= :since
        ORDER BY recorded_at DESC
        LIMIT 3000
        """
    )
    try:
        with _engine().connect() as conn:
            rows = [dict(r._mapping) for r in conn.execute(query, {"since": since})]
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _load_reconciliation_stats(days: int = 30) -> tuple[int, float]:
    """Return (mismatch_count, avg_reconciliation_seconds)."""
    since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Best-effort query against OMS event ledger payload fields.
    query = text(
        """
        SELECT
            COUNT(*) AS mismatches,
            AVG(
                CAST(JSON_EXTRACT(payload, '$.reconciliation_seconds') AS REAL)
            ) AS avg_recon_s
        FROM trade_events
        WHERE occurred_at >= :since
          AND (
              event_type = 'reconciliation_mismatch'
              OR JSON_EXTRACT(payload, '$.reconciliation_mismatch') = 1
          )
        """
    )
    try:
        with _engine().connect() as conn:
            row = conn.execute(query, {"since": since}).fetchone()
        if row is None:
            return 0, 0.0
        mismatches = int(row._mapping.get("mismatches") or 0)
        avg_recon_s = float(row._mapping.get("avg_recon_s") or 0.0)
        return mismatches, avg_recon_s
    except Exception:
        return 0, 0.0


def render():
    st.header("Reliability SLO")

    days = st.slider("Lookback (days)", min_value=1, max_value=90, value=30)

    broker_health_raw = _redis_get("algotrader:broker_health")
    broker_mode = _redis_get("algotrader:broker_mode") or "unknown"

    health = {}
    if broker_health_raw:
        try:
            health = json.loads(broker_health_raw)
        except Exception:
            health = {}

    tca = _load_tca_snapshot(days=days)
    mismatches, avg_recon_s = _load_reconciliation_stats(days=days)

    p95_latency_ms = 0.0
    p99_latency_ms = 0.0
    avg_fill_rate = 0.0

    if not tca.empty:
        if "fill_latency_ms" in tca.columns:
            p95_latency_ms = float(tca["fill_latency_ms"].quantile(0.95))
            p99_latency_ms = float(tca["fill_latency_ms"].quantile(0.99))
        if "fill_rate" in tca.columns:
            avg_fill_rate = float(tca["fill_rate"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Broker Mode", broker_mode.upper())
    c2.metric("Latency p95 (ms)", f"{p95_latency_ms:,.0f}")
    c3.metric("Latency p99 (ms)", f"{p99_latency_ms:,.0f}")
    c4.metric("Avg Fill Rate", f"{avg_fill_rate:.1%}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Recon Mismatches", f"{mismatches}")
    c6.metric("Avg Recon Time (s)", f"{avg_recon_s:.2f}")
    c7.metric(
        "Last Broker Check (s)",
        f"{float(health.get('latency_s', 0.0)):.2f}",
        delta=("UNHEALTHY" if not health.get("ok", True) else "healthy"),
        delta_color="inverse" if not health.get("ok", True) else "normal",
    )

    st.caption("SLO targets: p95 < 2000ms, p99 < 10000ms, fill rate > 85%, recon mismatch ~0.")

    if not tca.empty and "recorded_at" in tca.columns:
        chart_df = (
            tca.copy()
            .assign(recorded_at=pd.to_datetime(tca["recorded_at"]))
            .set_index("recorded_at")
            .sort_index()
        )
        st.subheader("Latency Trend")
        if "fill_latency_ms" in chart_df.columns:
            st.line_chart(chart_df[["fill_latency_ms"]], use_container_width=True)
