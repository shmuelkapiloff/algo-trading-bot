"""Historical replay scenarios for stress periods.

Scenarios:
- flash_crash_2010
- march_2020

This is a deterministic scenario runner that emits synthetic OMS events and
validates terminal states via ReplayHarness.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from trading_bot.src.backtesting.deterministic_replay import ReplayEvent, ReplayHarness
    from trading_bot.src.execution.order_state_machine import OrderState
except ModuleNotFoundError:
    from src.backtesting.deterministic_replay import ReplayEvent, ReplayHarness
    from src.execution.order_state_machine import OrderState


def _scenario_events(name: str) -> list[ReplayEvent]:
    if name == "flash_crash_2010":
        return [
            ReplayEvent(0, "FC1", "register", {"symbol": "SPY"}),
            ReplayEvent(1, "FC1", "transition", {"target": "SENT"}),
            ReplayEvent(2, "FC1", "transition", {"target": "ACK"}),
            ReplayEvent(3, "FC1", "transition", {"target": "FILLED", "filled_qty": 100}),
        ]
    if name == "march_2020":
        return [
            ReplayEvent(0, "CV1", "register", {"symbol": "QQQ"}),
            ReplayEvent(1, "CV1", "transition", {"target": "SENT"}),
            ReplayEvent(2, "CV1", "transition", {"target": "ACK"}),
            ReplayEvent(3, "CV1", "transition", {"target": "PARTIAL", "filled_qty": 40}),
            ReplayEvent(4, "CV1", "transition", {"target": "FILLED", "filled_qty": 100}),
        ]
    raise ValueError(f"unknown scenario: {name}")


def run_scenario(name: str):
    events = _scenario_events(name)
    harness = ReplayHarness(seed=2020)
    result = harness.run(events)

    order_id = events[0].order_id
    vr = harness.validate_ledger(result, {order_id: OrderState.FILLED})
    return result, vr


if __name__ == "__main__":
    for scenario in ["flash_crash_2010", "march_2020"]:
        result, vr = run_scenario(scenario)
        print(scenario, result.ledger_digest, "PASS" if vr.all_pass else "FAIL")
