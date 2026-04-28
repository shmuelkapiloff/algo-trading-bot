from __future__ import annotations

import pytest

from backtesting.walk_forward import build_windows
from scripts.capital_ramp_drill import simulate_ramp
from scripts.historical_replay import run_scenario
from src.data.borrow_availability import BorrowAvailabilityService
from src.data.sp500_provider import _parse_constituents_csv
from src.security.audit_trail import SignedAuditTrail


class _FakeBroker:
    async def list_assets(self):
        return [
            {"symbol": "AAPL", "shortable": True},
            {"symbol": "TSLA", "shortable": False},
            {"symbol": "MSFT", "shortable": True},
        ]


@pytest.mark.asyncio
async def test_borrow_availability_filters_unshortable():
    svc = BorrowAvailabilityService(_FakeBroker())
    blocked = await svc.get_unavailable_symbols(["AAPL", "TSLA"])
    assert blocked == {"TSLA"}


def test_sp500_csv_parser_requires_reasonable_size():
    raw = "symbol,name,sector\nAAPL,Apple,Tech\n"
    with pytest.raises(ValueError):
        _parse_constituents_csv(raw)


def test_walk_forward_windows():
    dates = [f"2020-01-{i:02d}" for i in range(1, 31)]
    windows = build_windows(dates, train_size=10, test_size=5)
    assert len(windows) == 4
    assert windows[0].train_start == "2020-01-01"
    assert windows[0].test_start == "2020-01-11"


def test_historical_replay_scenarios_pass():
    for name in ("flash_crash_2010", "march_2020"):
        _, validation = run_scenario(name)
        assert validation.all_pass is True


def test_audit_trail_signed_and_verifiable():
    trail = SignedAuditTrail(secret=b"1234567890abcdef")
    entry = trail.record(actor="ops", action="halt", payload={"reason": "drill"})
    assert trail.verify(entry) is True


def test_capital_ramp_drill_progression():
    early = simulate_ramp(current_days=8)
    mid = simulate_ramp(current_days=30)
    assert early[0]["eligible"] is False
    assert mid[0]["eligible"] is True
    assert mid[1]["eligible"] is True
    assert mid[2]["eligible"] is False
