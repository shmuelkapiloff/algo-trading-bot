"""Canary Gate Evaluator for Phase 6 readiness.

Evaluates promotion readiness for:
- Canary 1 (10%)
- Canary 2 (50%)
- Full (100%)

The script is intentionally read-only and computes a gate decision from
existing metrics in the database plus optional CLI overrides.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

from sqlalchemy import create_engine, text


@dataclass
class CanarySnapshot:
    trading_days: int
    reconciliation_mismatches: int
    sharpe_30d: float | None
    max_drawdown_pct: float | None
    slo_green: bool


@dataclass
class GateDecision:
    canary1_ready: bool
    canary2_ready: bool
    full_ready: bool
    reasons: list[str]


def evaluate_gate(snapshot: CanarySnapshot) -> GateDecision:
    reasons: list[str] = []

    canary1_ready = (
        snapshot.trading_days >= 10
        and snapshot.reconciliation_mismatches == 0
    )
    if snapshot.trading_days < 10:
        reasons.append(
            f"Canary1: trading_days={snapshot.trading_days} < required 10"
        )
    if snapshot.reconciliation_mismatches != 0:
        reasons.append(
            "Canary1: reconciliation mismatches must be 0"
        )

    # "15 ימים נוספים" after Canary1 => total 25 days minimum
    canary2_ready = (
        canary1_ready
        and snapshot.trading_days >= 25
        and (snapshot.sharpe_30d is not None and snapshot.sharpe_30d > 0.5)
    )
    if canary1_ready and snapshot.trading_days < 25:
        reasons.append(
            f"Canary2: trading_days={snapshot.trading_days} < required 25"
        )
    if snapshot.sharpe_30d is None:
        reasons.append("Canary2: sharpe_30d is missing")
    elif snapshot.sharpe_30d <= 0.5:
        reasons.append(
            f"Canary2: sharpe_30d={snapshot.sharpe_30d:.3f} <= 0.5"
        )

    # "20 ימים נוספים" after Canary2 => total 45 days minimum
    full_ready = (
        canary2_ready
        and snapshot.trading_days >= 45
        and (snapshot.max_drawdown_pct is not None and snapshot.max_drawdown_pct < 8.0)
        and snapshot.slo_green
    )
    if canary2_ready and snapshot.trading_days < 45:
        reasons.append(
            f"Full: trading_days={snapshot.trading_days} < required 45"
        )
    if snapshot.max_drawdown_pct is None:
        reasons.append("Full: max_drawdown_pct is missing")
    elif snapshot.max_drawdown_pct >= 8.0:
        reasons.append(
            f"Full: max_drawdown_pct={snapshot.max_drawdown_pct:.2f} >= 8.0"
        )
    if not snapshot.slo_green:
        reasons.append("Full: SLO gate is not green")

    return GateDecision(
        canary1_ready=canary1_ready,
        canary2_ready=canary2_ready,
        full_ready=full_ready,
        reasons=reasons,
    )


def _engine():
    url = os.getenv("DATABASE_URL", "sqlite:///trading.db")
    url = url.replace("sqlite+aiosqlite", "sqlite").replace(
        "postgresql+asyncpg", "postgresql"
    )
    return create_engine(
        url, connect_args={"check_same_thread": False} if "sqlite" in url else {}
    )


def _load_snapshot(
    sharpe_30d: float | None,
    max_drawdown_pct: float | None,
) -> CanarySnapshot:
    trading_days = 0
    reconciliation_mismatches = 0
    slo_green = False

    eng = _engine()

    with eng.connect() as conn:
        day_q = text(
            """
            SELECT COUNT(DISTINCT DATE(recorded_at)) AS trading_days
            FROM tca_records
            """
        )
        row = conn.execute(day_q).fetchone()
        if row is not None:
            trading_days = int(row._mapping.get("trading_days") or 0)

        recon_q = text(
            """
            SELECT COUNT(*) AS mismatches
            FROM trade_events
            WHERE event_type = 'reconciliation_mismatch'
               OR JSON_EXTRACT(payload, '$.reconciliation_mismatch') = 1
            """
        )
        row = conn.execute(recon_q).fetchone()
        if row is not None:
            reconciliation_mismatches = int(row._mapping.get("mismatches") or 0)

        slo_q = text(
            """
            SELECT
                AVG(fill_rate) AS avg_fill_rate,
                AVG(CASE WHEN fill_latency_ms IS NOT NULL THEN fill_latency_ms END) AS avg_latency_ms
            FROM tca_records
            """
        )
        row = conn.execute(slo_q).fetchone()
        if row is not None:
            avg_fill_rate = float(row._mapping.get("avg_fill_rate") or 0.0)
            avg_latency_ms = float(row._mapping.get("avg_latency_ms") or 0.0)
            slo_green = avg_fill_rate >= 0.85 and avg_latency_ms < 2000.0

    return CanarySnapshot(
        trading_days=trading_days,
        reconciliation_mismatches=reconciliation_mismatches,
        sharpe_30d=sharpe_30d,
        max_drawdown_pct=max_drawdown_pct,
        slo_green=slo_green,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Phase 6 canary gate readiness")
    parser.add_argument(
        "--sharpe-30d",
        type=float,
        default=None,
        help="Optional rolling 30D Sharpe (required for Canary2/Full gate)",
    )
    parser.add_argument(
        "--max-dd-pct",
        type=float,
        default=None,
        help="Optional max drawdown percent (required for Full gate)",
    )
    args = parser.parse_args()

    snapshot = _load_snapshot(
        sharpe_30d=args.sharpe_30d,
        max_drawdown_pct=args.max_dd_pct,
    )
    decision = evaluate_gate(snapshot)

    payload = {
        "snapshot": {
            "trading_days": snapshot.trading_days,
            "reconciliation_mismatches": snapshot.reconciliation_mismatches,
            "sharpe_30d": snapshot.sharpe_30d,
            "max_drawdown_pct": snapshot.max_drawdown_pct,
            "slo_green": snapshot.slo_green,
        },
        "decision": {
            "canary1_ready": decision.canary1_ready,
            "canary2_ready": decision.canary2_ready,
            "full_ready": decision.full_ready,
            "reasons": decision.reasons,
        },
    }

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
