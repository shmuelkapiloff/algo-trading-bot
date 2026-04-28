from __future__ import annotations

from scripts.canary_gate import CanarySnapshot, evaluate_gate


def test_canary1_ready_only():
    snap = CanarySnapshot(
        trading_days=10,
        reconciliation_mismatches=0,
        sharpe_30d=None,
        max_drawdown_pct=None,
        slo_green=False,
    )
    d = evaluate_gate(snap)
    assert d.canary1_ready is True
    assert d.canary2_ready is False
    assert d.full_ready is False


def test_canary2_ready_when_sharpe_and_days_ok():
    snap = CanarySnapshot(
        trading_days=25,
        reconciliation_mismatches=0,
        sharpe_30d=0.7,
        max_drawdown_pct=None,
        slo_green=False,
    )
    d = evaluate_gate(snap)
    assert d.canary1_ready is True
    assert d.canary2_ready is True
    assert d.full_ready is False


def test_full_ready_requires_all_conditions():
    snap = CanarySnapshot(
        trading_days=45,
        reconciliation_mismatches=0,
        sharpe_30d=0.9,
        max_drawdown_pct=7.5,
        slo_green=True,
    )
    d = evaluate_gate(snap)
    assert d.canary1_ready is True
    assert d.canary2_ready is True
    assert d.full_ready is True


def test_full_not_ready_if_dd_too_high():
    snap = CanarySnapshot(
        trading_days=60,
        reconciliation_mismatches=0,
        sharpe_30d=1.0,
        max_drawdown_pct=9.2,
        slo_green=True,
    )
    d = evaluate_gate(snap)
    assert d.full_ready is False
    assert any("max_drawdown_pct" in r for r in d.reasons)


def test_canary1_not_ready_with_mismatches():
    snap = CanarySnapshot(
        trading_days=50,
        reconciliation_mismatches=2,
        sharpe_30d=1.2,
        max_drawdown_pct=4.0,
        slo_green=True,
    )
    d = evaluate_gate(snap)
    assert d.canary1_ready is False
    assert d.canary2_ready is False
    assert d.full_ready is False
