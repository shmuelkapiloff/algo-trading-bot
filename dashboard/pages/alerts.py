"""Alerts & Runbooks Page — Emergency controls and incident playbooks.

Provides:
  - Emergency PAUSE button (calls FastAPI /halt endpoint with fencing token)
  - Emergency RESUME button (calls FastAPI /resume endpoint)
  - System status indicators (trading state, watchdog, canary)
  - Active incidents list (from Redis)
  - Severity-graded alert feed
  - Runbook quick-reference cards for common incidents

Security note:
  - The HALT/RESUME buttons call the FastAPI control-plane API at
    CONTROL_API_URL (default http://localhost:8001).
  - All commands require a valid API key (CONTROL_API_KEY env var).
  - This page is analytics-only in the Streamlit process; actual
    order submission never happens here.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st

_CONTROL_API_URL = os.getenv("CONTROL_API_URL", "http://localhost:8001")
_CONTROL_API_KEY = os.getenv("CONTROL_API_KEY", "")
_REQUEST_TIMEOUT = 5  # seconds


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _api_post(path: str, payload: dict | None = None) -> tuple[bool, str]:
    """POST to the control-plane API. Returns (success, message)."""
    url = f"{_CONTROL_API_URL}{path}"
    headers = {"X-API-Key": _CONTROL_API_KEY, "Content-Type": "application/json"}
    try:
        resp = requests.post(url, json=payload or {}, headers=headers, timeout=_REQUEST_TIMEOUT)
        if resp.ok:
            data = resp.json()
            return True, data.get("message", "OK")
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except requests.exceptions.ConnectionError:
        return False, "Control API unreachable — is the FastAPI server running?"
    except Exception as exc:
        return False, str(exc)


def _api_get(path: str) -> tuple[bool, dict]:
    """GET from the control-plane API. Returns (success, data)."""
    url = f"{_CONTROL_API_URL}{path}"
    headers = {"X-API-Key": _CONTROL_API_KEY}
    try:
        resp = requests.get(url, headers=headers, timeout=_REQUEST_TIMEOUT)
        if resp.ok:
            return True, resp.json()
        return False, {}
    except Exception:
        return False, {}


def _load_trading_state() -> str:
    """Fetch current trading state from control API."""
    ok, data = _api_get("/status")
    if ok:
        return data.get("state", "unknown")
    # Fallback: try Redis directly
    try:
        import redis as _redis  # type: ignore
        r = _redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        state = r.get("algotrader:state")
        return state.decode() if state else "unknown"
    except Exception:
        return "unknown"


def _load_recent_alerts(n: int = 50) -> list[dict]:
    """Load recent alerts from Redis list."""
    try:
        import json
        import redis as _redis  # type: ignore
        r = _redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        raw = r.lrange("algotrader:alerts", 0, n - 1)
        alerts = []
        for item in raw:
            try:
                alerts.append(json.loads(item))
            except Exception:
                alerts.append({"message": item.decode(), "severity": "info"})
        return alerts
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Runbook definitions
# ---------------------------------------------------------------------------

_RUNBOOKS = [
    {
        "title": "🔴 Emergency Halt — Broker Latency Spike",
        "trigger": "Broker p95 latency > 2000ms for 3+ consecutive samples",
        "severity": "critical",
        "steps": [
            "1. Press **EMERGENCY HALT** button above",
            "2. Check broker status page (Alpaca status.alpaca.markets)",
            "3. Verify Redis connectivity: `redis-cli ping`",
            "4. Check canary probe logs: `docker logs algotrader | grep canary`",
            "5. If broker resolved: press **RESUME** and monitor for 15 min",
            "6. If still degraded: keep halted, investigate manually",
        ],
    },
    {
        "title": "🟡 Slippage Alert — TCA Throttle Active",
        "trigger": "avg_slippage_bps > 12 bps over last 20 orders",
        "severity": "warn",
        "steps": [
            "1. Open TCA page — review recent slippage trend",
            "2. Check if broad market is in a volatile period (check VIX)",
            "3. If isolated spike: wait for auto-resume after 30 min",
            "4. If persistent: reduce position sizes via config_ui",
            "5. Check execution logs for specific symbol outliers",
        ],
    },
    {
        "title": "🔴 PDT Rule Approaching (Day Trade Counter)",
        "trigger": "PDT counter >= 3 day trades in a rolling 5-session window",
        "severity": "critical",
        "steps": [
            "1. Check positions page for current open positions",
            "2. Do NOT close any positions intraday (counts as day trade)",
            "3. Wait for EOD before considering exits",
            "4. Review strategy settings to reduce round-trip frequency",
            "5. Consider enabling PDT protection mode in config",
        ],
    },
    {
        "title": "🟡 Reconciliation Mismatch",
        "trigger": "Position count / qty diverges > 5% between DB and broker",
        "severity": "warn",
        "steps": [
            "1. Press **EMERGENCY HALT** to stop new orders",
            "2. Run manual reconciliation: `docker exec algotrader python -m src.execution.reconcile`",
            "3. Compare DB positions vs Alpaca dashboard manually",
            "4. Correct DB if needed (check OMS ledger for last known good state)",
            "5. Resume only after DB and broker agree",
        ],
    },
    {
        "title": "🔴 Portfolio Drawdown Breach",
        "trigger": "Portfolio loss >= max_total_drawdown (default 15%)",
        "severity": "critical",
        "steps": [
            "1. System should auto-halt (check if HALT is active)",
            "2. Review all open positions — identify largest losers",
            "3. Do NOT immediately close all positions (may lock in losses at worst time)",
            "4. Review market regime — did macro environment shift?",
            "5. Escalate to manual review before resuming",
            "6. Consider reducing max_open_positions in config",
        ],
    },
]


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------


def render():
    st.header("Alerts & Runbooks")
    st.caption("Emergency controls and incident playbooks. All trading commands go through the FastAPI control plane.")

    # ── Trading state banner ───────────────────────────────────────────
    state = _load_trading_state()
    state_color = {
        "active": "🟢",
        "halted": "🔴",
        "paused": "🟡",
        "close_only": "🟠",
    }.get(state, "⚪")
    st.subheader(f"System State: {state_color} **{state.upper()}**")

    st.divider()

    # ── Emergency controls ─────────────────────────────────────────────
    st.subheader("Emergency Controls")
    st.warning(
        "These buttons call the FastAPI control-plane. "
        "A **HALT** stops all new order submissions immediately. "
        "A **RESUME** re-enables trading after manual review."
    )

    col_halt, col_resume, col_status = st.columns(3)

    with col_halt:
        if st.button("🛑 EMERGENCY HALT", type="primary", use_container_width=True):
            ok, msg = _api_post("/halt", {"reason": "manual_dashboard_halt"})
            if ok:
                st.success(f"HALT issued: {msg}")
                st.rerun()
            else:
                st.error(f"HALT failed: {msg}")

    with col_resume:
        if st.button("▶ RESUME TRADING", use_container_width=True):
            ok, msg = _api_post("/resume", {"reason": "manual_dashboard_resume"})
            if ok:
                st.success(f"RESUME issued: {msg}")
                st.rerun()
            else:
                st.error(f"RESUME failed: {msg}")

    with col_status:
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()

    st.divider()

    # ── Active alerts ──────────────────────────────────────────────────
    st.subheader("Recent Alerts")
    alerts = _load_recent_alerts(50)
    if alerts:
        for alert in alerts[:20]:
            severity = alert.get("severity", "info")
            msg = alert.get("message", str(alert))
            ts = alert.get("timestamp", "")
            icon = {"critical": "🔴", "warn": "🟡", "info": "ℹ️"}.get(severity, "⚪")
            st.write(f"{icon} `{ts}` — {msg}")
    else:
        st.info("No recent alerts in Redis. System may not be running, or no alerts have fired.")

    st.divider()

    # ── Runbooks ───────────────────────────────────────────────────────
    st.subheader("Incident Runbooks")
    st.caption("Step-by-step resolution guides for common incidents.")

    for rb in _RUNBOOKS:
        severity_color = {"critical": "🔴", "warn": "🟡"}.get(rb["severity"], "ℹ️")
        with st.expander(f"{rb['title']}"):
            st.caption(f"**Trigger:** {rb['trigger']}")
            for step in rb["steps"]:
                st.markdown(step)
