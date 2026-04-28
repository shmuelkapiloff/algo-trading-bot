"""
Tests for Phase 5.5 implementations:
  - OMS per-order_id event ordering + late-event reconciliation
  - Circuit breaker cascade auto-activation
  - Deterministic replay harness
  - Control-plane HA leader election (unit-level — Redis mocked)
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading_bot.src.execution.order_state_machine import (
    OrderRecord,
    OrderState,
    OrderStateMachine,
)
from trading_bot.src.backtesting.deterministic_replay import (
    ReplayEvent,
    ReplayHarness,
)


# ===========================================================================
# OMS: per-order_id event ordering
# ===========================================================================


class TestOrderedTransitions:
    def _make_oms_with_order(self, order_id="ORD1") -> tuple[OrderStateMachine, str]:
        oms = OrderStateMachine()
        oms.register(
            OrderRecord(order_id=order_id, symbol="AAPL", strategy_name="test")
        )
        return oms, order_id

    def test_in_order_sequence_applies(self):
        oms, oid = self._make_oms_with_order()
        applied = oms.transition_ordered(oid, OrderState.SENT, seq=0)
        assert applied is True
        assert oms.get(oid).state == OrderState.SENT

    def test_next_seq_applies(self):
        oms, oid = self._make_oms_with_order()
        oms.transition_ordered(oid, OrderState.SENT, seq=0)
        applied = oms.transition_ordered(oid, OrderState.ACK, seq=1)
        assert applied is True
        assert oms.get(oid).state == OrderState.ACK

    def test_duplicate_seq_ignored(self):
        oms, oid = self._make_oms_with_order()
        oms.transition_ordered(oid, OrderState.SENT, seq=0)
        applied = oms.transition_ordered(oid, OrderState.SENT, seq=0)
        assert applied is False
        assert oms.get(oid).last_seq == 0

    def test_gap_buffers_event(self):
        oms, oid = self._make_oms_with_order()
        # seq=0 applied, then seq=2 arrives before seq=1
        oms.transition_ordered(oid, OrderState.SENT, seq=0)
        applied = oms.transition_ordered(oid, OrderState.FILLED, seq=2)
        assert applied is False
        assert oms.get_pending_count() == 1
        # State still SENT — seq=2 is buffered
        assert oms.get(oid).state == OrderState.SENT

    def test_gap_flushed_when_missing_seq_arrives(self):
        oms, oid = self._make_oms_with_order()
        oms.transition_ordered(oid, OrderState.SENT, seq=0)
        # Send seq=2 first (buffered)
        oms.transition_ordered(oid, OrderState.FILLED, seq=2)
        assert oms.get_pending_count() == 1
        # Now send seq=1 (ACK) — should apply seq=1 then flush seq=2
        oms.transition_ordered(oid, OrderState.ACK, seq=1)
        # Both should have been applied
        assert oms.get(oid).state == OrderState.FILLED
        assert oms.get_pending_count() == 0

    def test_late_event_below_last_seq_buffered(self):
        oms, oid = self._make_oms_with_order()
        oms.transition_ordered(oid, OrderState.SENT, seq=0)
        oms.transition_ordered(oid, OrderState.ACK, seq=1)
        # seq=0 arrives again after seq=1 — very late
        applied = oms.transition_ordered(oid, OrderState.SENT, seq=0)
        assert applied is False
        # ACK state unchanged
        assert oms.get(oid).state == OrderState.ACK

    def test_clear_terminal_removes_pending(self):
        oms, oid = self._make_oms_with_order()
        oms.transition_ordered(oid, OrderState.SENT, seq=0)
        oms.transition_ordered(oid, OrderState.FILLED, seq=2)  # buffered
        # Force terminal directly via regular transition
        oms.transition(oid, OrderState.ACK)
        oms.transition(oid, OrderState.FILLED)
        count = oms.clear_terminal()
        assert count == 1
        assert oms.get_pending_count() == 0

    def test_unknown_order_returns_false(self):
        oms = OrderStateMachine()
        applied = oms.transition_ordered("NONEXISTENT", OrderState.SENT, seq=0)
        assert applied is False


# ===========================================================================
# Circuit breaker cascade auto-activation
# ===========================================================================


class TestCascadeCircuitBreaker:
    def _make_controller(self):
        from trading_bot.src.risk.incident_controller import IncidentController
        return IncidentController(state_store=None, alert_dispatcher=None, event_bus=None)

    @pytest.mark.asyncio
    async def test_single_critical_no_escalation(self):
        from trading_bot.src.risk.incident_controller import IncidentType, Severity
        ctrl = self._make_controller()
        await ctrl.report(IncidentType.BROKER_LATENCY_SPIKE, Severity.CRITICAL)
        assert ctrl.get_active_severity() == Severity.CRITICAL
        # No cascade yet (only 1 incident)
        open_incidents = ctrl.get_active_incidents()
        emergency_count = sum(1 for i in open_incidents if i.severity == Severity.EMERGENCY)
        assert emergency_count == 0

    @pytest.mark.asyncio
    async def test_three_criticals_trigger_emergency(self):
        from trading_bot.src.risk.incident_controller import IncidentType, Severity
        ctrl = self._make_controller()
        for _ in range(3):
            await ctrl.report(IncidentType.BROKER_LATENCY_SPIKE, Severity.CRITICAL)
        assert ctrl.get_active_severity() == Severity.EMERGENCY

    @pytest.mark.asyncio
    async def test_cascade_clears_timestamp_buffer(self):
        from trading_bot.src.risk.incident_controller import IncidentType, Severity
        ctrl = self._make_controller()
        for _ in range(3):
            await ctrl.report(IncidentType.BROKER_LATENCY_SPIKE, Severity.CRITICAL)
        # After cascade fires, buffer should be cleared
        assert ctrl._cascade_timestamps == []

    @pytest.mark.asyncio
    async def test_stale_criticals_dont_cascade(self):
        """Criticals outside the window should not count."""
        from trading_bot.src.risk.incident_controller import (
            IncidentController,
            IncidentType,
            Severity,
            _CASCADE_WINDOW_SECONDS,
        )
        ctrl = IncidentController()
        # Inject 2 old timestamps outside the window
        ctrl._cascade_timestamps = [time.monotonic() - _CASCADE_WINDOW_SECONDS - 10] * 2
        # One new CRITICAL — total in window = 1, below threshold
        await ctrl.report(IncidentType.RECONCILIATION_MISMATCH, Severity.CRITICAL)
        assert ctrl.get_active_severity() == Severity.CRITICAL
        open_incidents = ctrl.get_active_incidents()
        emergency_count = sum(1 for i in open_incidents if i.severity == Severity.EMERGENCY)
        assert emergency_count == 0


# ===========================================================================
# Deterministic replay harness
# ===========================================================================


class TestDeterministicReplay:
    def _base_events(self):
        return [
            ReplayEvent(seq=0, order_id="ORD1", event_type="register",
                        payload={"symbol": "AAPL", "strategy_name": "test"}),
            ReplayEvent(seq=1, order_id="ORD1", event_type="transition",
                        payload={"target": "SENT"}),
            ReplayEvent(seq=2, order_id="ORD1", event_type="transition",
                        payload={"target": "ACK"}),
            ReplayEvent(seq=3, order_id="ORD1", event_type="transition",
                        payload={"target": "FILLED", "filled_qty": 100}),
        ]

    def test_basic_replay(self):
        harness = ReplayHarness(seed=42)
        result = harness.run(self._base_events())
        assert result.events_processed == 4
        assert result.orders["ORD1"].final_state == OrderState.FILLED

    def test_validation_passes(self):
        harness = ReplayHarness(seed=42)
        result = harness.run(self._base_events())
        vr = harness.validate_ledger(result, {"ORD1": OrderState.FILLED})
        assert vr.all_pass is True
        assert vr.failures == []

    def test_validation_fails_on_wrong_expected(self):
        harness = ReplayHarness(seed=42)
        result = harness.run(self._base_events())
        vr = harness.validate_ledger(result, {"ORD1": OrderState.CANCELED})
        assert vr.all_pass is False
        assert vr.failures[0].order_id == "ORD1"

    def test_same_seed_same_digest(self):
        events = self._base_events()
        r1 = ReplayHarness(seed=7).run(events)
        r2 = ReplayHarness(seed=7).run(events)
        assert r1.ledger_digest == r2.ledger_digest

    def test_out_of_order_events_reconciled(self):
        events = [
            ReplayEvent(seq=0, order_id="ORD2", event_type="register",
                        payload={"symbol": "TSLA"}),
            # Deliver seq=2 before seq=1
            ReplayEvent(seq=2, order_id="ORD2", event_type="transition",
                        payload={"target": "ACK"}),
            ReplayEvent(seq=1, order_id="ORD2", event_type="transition",
                        payload={"target": "SENT"}),
            ReplayEvent(seq=3, order_id="ORD2", event_type="transition",
                        payload={"target": "FILLED", "filled_qty": 50}),
        ]
        harness = ReplayHarness(seed=0)
        result = harness.run(events)
        assert result.orders["ORD2"].final_state == OrderState.FILLED

    def test_noop_events_skipped(self):
        events = self._base_events() + [
            ReplayEvent(seq=99, order_id="ORD1", event_type="noop", payload={})
        ]
        result = ReplayHarness(seed=0).run(events)
        assert result.events_skipped == 1
        assert result.events_processed == 4

    def test_invalid_state_skipped(self):
        events = [
            ReplayEvent(seq=0, order_id="ORD3", event_type="register",
                        payload={"symbol": "NVDA"}),
            ReplayEvent(seq=1, order_id="ORD3", event_type="transition",
                        payload={"target": "BOGUS_STATE"}),
        ]
        result = ReplayHarness(seed=0).run(events)
        assert result.events_skipped == 1

    def test_validation_missing_order(self):
        harness = ReplayHarness(seed=0)
        result = harness.run(self._base_events())
        vr = harness.validate_ledger(result, {"NONEXISTENT": OrderState.FILLED})
        assert vr.all_pass is False
        assert vr.failures[0].actual is None


# ===========================================================================
# Event Bus Phase 2: Redis Streams backend exists
# ===========================================================================


class TestRedisStreamsBusExists:
    """Smoke-test that RedisStreamEventBus is importable and has the right interface."""

    def test_import(self):
        from trading_bot.src.events.bus import RedisStreamEventBus, create_event_bus
        assert RedisStreamEventBus is not None

    def test_create_event_bus_asyncio_backend(self):
        from trading_bot.src.events.bus import EventBus, create_event_bus
        bus = create_event_bus(backend="asyncio_queue")
        assert isinstance(bus, EventBus)

    def test_subscribe_and_publish_interface(self):
        from trading_bot.src.events.bus import RedisStreamEventBus
        bus = RedisStreamEventBus.__new__(RedisStreamEventBus)
        bus._handlers = {}
        from collections import defaultdict
        bus._handlers = defaultdict(list)

        async def handler(topic, payload):
            pass

        bus.subscribe("test.topic", handler)
        assert handler in bus._handlers["test.topic"]
