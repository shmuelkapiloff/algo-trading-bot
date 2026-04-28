"""Incident Drill Simulator (Phase 5.6).

Simulates a broker outage and verifies:
  1. auto-failover after N consecutive failures
  2. transition to CLOSE_ONLY
  3. return to PRIMARY after healthy checks

Run:
  python scripts/incident_drill.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass

# Ensure project root is importable when executing this script directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from trading_bot.src.execution.broker_health import (
        BrokerHealthMonitor,
        BrokerMode,
        HealthCheckResult,
    )
except ModuleNotFoundError:
    from src.execution.broker_health import (
        BrokerHealthMonitor,
        BrokerMode,
        HealthCheckResult,
    )


class _MemoryRedis:
    def __init__(self):
        self._data: dict[str, str] = {}

    async def set(self, key: str, value, ex=None):
        self._data[key] = value
        return True

    async def get(self, key: str):
        return self._data.get(key)


class _FakeBroker:
    def __init__(self, name="broker"):
        self.name = name

    async def get_account(self):
        return {"id": "ok"}

    async def list_positions(self):
        return []


class _FakeStateStore:
    def __init__(self):
        self.calls = []

    async def force_transition_internal(self, target, reason: str):
        self.calls.append((target, reason))
        return True, "ok"


class _FakeBus:
    def __init__(self):
        self.events = []

    async def publish(self, topic, payload):
        self.events.append((topic, payload))


class _FakeAlerts:
    def __init__(self):
        self.messages = []

    async def send_alert(self, level: str, message: str):
        self.messages.append((level, message))


@dataclass
class DrillResult:
    failover_triggered: bool
    close_only_transitioned: bool
    returned_to_primary: bool


async def run_incident_drill() -> DrillResult:
    # Drill context: ephemeral keys are enough for local simulation.
    try:
        from src.security import fencing_tokens as _ft_src

        _ft_src.generate_ephemeral_keys()
    except Exception:
        pass
    try:
        from trading_bot.src.security import fencing_tokens as _ft_pkg

        _ft_pkg.generate_ephemeral_keys()
    except Exception:
        pass

    primary = _FakeBroker("primary")
    secondary = _FakeBroker("secondary")
    bus = _FakeBus()
    state = _FakeStateStore()
    alerts = _FakeAlerts()
    redis_client = _MemoryRedis()

    mon = BrokerHealthMonitor(
        primary=primary,
        secondary=secondary,
        event_bus=bus,
        state_store=state,
        redis_client=redis_client,
        alert_dispatcher=alerts,
        oms_ledger=object(),
        check_interval_s=5.0,
        max_consecutive_failures=3,
        latency_threshold_s=10.0,
    )

    # Simulate outage: 3 unhealthy checks
    for _ in range(3):
        await mon._handle_result(
            HealthCheckResult(
                broker_name="primary",
                ok=False,
                latency_s=0.1,
                error="simulated_outage",
            )
        )

    failover_triggered = mon.mode == BrokerMode.FAILOVER
    close_only_transitioned = len(state.calls) > 0

    # Simulate recovery
    await mon._handle_result(
        HealthCheckResult(
            broker_name="primary",
            ok=True,
            latency_s=0.05,
            error=None,
        )
    )

    returned_to_primary = mon.mode == BrokerMode.PRIMARY

    return DrillResult(
        failover_triggered=failover_triggered,
        close_only_transitioned=close_only_transitioned,
        returned_to_primary=returned_to_primary,
    )


def main() -> int:
    result = asyncio.run(run_incident_drill())
    print("Incident Drill Result")
    print(f"  failover_triggered:     {result.failover_triggered}")
    print(f"  close_only_transitioned:{result.close_only_transitioned}")
    print(f"  returned_to_primary:    {result.returned_to_primary}")

    ok = (
        result.failover_triggered
        and result.close_only_transitioned
        and result.returned_to_primary
    )
    if ok:
        print("PASS: incident drill completed")
        return 0

    print("FAIL: incident drill assertions failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
