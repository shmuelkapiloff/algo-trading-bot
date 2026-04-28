"""Canary rollout + capital ramp drill simulator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RampStage:
    name: str
    allocation_pct: int
    min_trading_days: int


STAGES = [
    RampStage("canary1", 10, 10),
    RampStage("canary2", 50, 25),
    RampStage("full", 100, 45),
]


def simulate_ramp(current_days: int) -> list[dict]:
    result: list[dict] = []
    for s in STAGES:
        result.append(
            {
                "stage": s.name,
                "allocation_pct": s.allocation_pct,
                "eligible": current_days >= s.min_trading_days,
                "required_days": s.min_trading_days,
            }
        )
    return result


if __name__ == "__main__":
    import json

    print(json.dumps(simulate_ramp(current_days=30), indent=2))
