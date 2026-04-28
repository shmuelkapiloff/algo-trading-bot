"""Tests for src/risk/incident_controller.py."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.risk.incident_controller import (
    IncidentController,
    IncidentType,
    IncidentStatus,
    Severity,
)


@pytest.fixture
def controller():
    return IncidentController(
        state_store=None,
        alert_dispatcher=None,
        event_bus=None,
    )


@pytest.fixture
def controller_with_mocks():
    state_store = MagicMock()
    state_store.pause = AsyncMock()
    state_store.emergency_stop = AsyncMock()
    state_store.resume = AsyncMock()

    alerts = MagicMock()
    alerts.send = AsyncMock()
    alerts.send_urgent = AsyncMock()

    bus = MagicMock()
    bus.publish = AsyncMock()

    return (
        IncidentController(state_store, alerts, bus),
        state_store,
        alerts,
        bus,
    )


class TestInitialState:
    def test_initial_multiplier_is_one(self, controller):
        assert controller.get_position_size_multiplier() == 1.0

    def test_initial_not_paused(self, controller):
        assert controller.is_paused() is False

    def test_initial_no_active_incidents(self, controller):
        assert controller.get_active_incidents() == []

    def test_initial_severity_is_none(self, controller):
        assert controller.get_active_severity() is None


class TestWarningEscalation:
    @pytest.mark.asyncio
    async def test_warning_throttles_to_50pct(self, controller):
        await controller.report(IncidentType.LIQUIDITY_STRESS, Severity.WARNING)
        assert controller.get_position_size_multiplier() == 0.5

    @pytest.mark.asyncio
    async def test_warning_does_not_pause(self, controller):
        await controller.report(IncidentType.TCA_SLIPPAGE_BREACH, Severity.WARNING)
        assert controller.is_paused() is False

    @pytest.mark.asyncio
    async def test_warning_creates_open_incident(self, controller):
        await controller.report(IncidentType.BROKER_LATENCY_SPIKE, Severity.WARNING)
        incidents = controller.get_active_incidents()
        assert len(incidents) == 1
        assert incidents[0].status == IncidentStatus.OPEN

    @pytest.mark.asyncio
    async def test_warning_sends_alert(self, controller_with_mocks):
        ctrl, _, alerts, _ = controller_with_mocks
        await ctrl.report(IncidentType.LIQUIDITY_STRESS, Severity.WARNING)
        alerts.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_warning_publishes_event(self, controller_with_mocks):
        ctrl, _, _, bus = controller_with_mocks
        await ctrl.report(IncidentType.LIQUIDITY_STRESS, Severity.WARNING)
        bus.publish.assert_called_once()
        topic = bus.publish.call_args[0][0]
        assert "warning" in topic


class TestCriticalEscalation:
    @pytest.mark.asyncio
    async def test_critical_zeroes_multiplier(self, controller):
        await controller.report(IncidentType.RECONCILIATION_MISMATCH, Severity.CRITICAL)
        assert controller.get_position_size_multiplier() == 0.0

    @pytest.mark.asyncio
    async def test_critical_pauses(self, controller):
        await controller.report(IncidentType.DRAWDOWN_BREACH, Severity.CRITICAL)
        assert controller.is_paused() is True

    @pytest.mark.asyncio
    async def test_critical_calls_state_store_pause(self, controller_with_mocks):
        ctrl, state_store, _, _ = controller_with_mocks
        await ctrl.report(IncidentType.DRAWDOWN_BREACH, Severity.CRITICAL)
        state_store.pause.assert_called_once()

    @pytest.mark.asyncio
    async def test_critical_sends_urgent_alert(self, controller_with_mocks):
        ctrl, _, alerts, _ = controller_with_mocks
        await ctrl.report(IncidentType.CORRELATION_CRISIS, Severity.CRITICAL)
        alerts.send_urgent.assert_called_once()


class TestEmergencyEscalation:
    @pytest.mark.asyncio
    async def test_emergency_zeroes_multiplier(self, controller):
        await controller.report(IncidentType.DRAWDOWN_BREACH, Severity.EMERGENCY)
        assert controller.get_position_size_multiplier() == 0.0

    @pytest.mark.asyncio
    async def test_emergency_pauses(self, controller):
        await controller.report(IncidentType.DRAWDOWN_BREACH, Severity.EMERGENCY)
        assert controller.is_paused() is True

    @pytest.mark.asyncio
    async def test_emergency_calls_emergency_stop(self, controller_with_mocks):
        ctrl, state_store, _, _ = controller_with_mocks
        await ctrl.report(IncidentType.DRAWDOWN_BREACH, Severity.EMERGENCY)
        state_store.emergency_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_emergency_publishes_emergency_event(self, controller_with_mocks):
        ctrl, _, _, bus = controller_with_mocks
        await ctrl.report(IncidentType.MANUAL, Severity.EMERGENCY)
        topic = bus.publish.call_args[0][0]
        assert "emergency" in topic


class TestSeverityRatchet:
    @pytest.mark.asyncio
    async def test_critical_overrides_warning_multiplier(self, controller):
        await controller.report(IncidentType.LIQUIDITY_STRESS, Severity.WARNING)
        assert controller.get_position_size_multiplier() == 0.5
        await controller.report(IncidentType.DRAWDOWN_BREACH, Severity.CRITICAL)
        assert controller.get_position_size_multiplier() == 0.0

    @pytest.mark.asyncio
    async def test_active_severity_ratchets_up(self, controller):
        await controller.report(IncidentType.LIQUIDITY_STRESS, Severity.WARNING)
        assert controller.get_active_severity() == Severity.WARNING
        await controller.report(IncidentType.DRAWDOWN_BREACH, Severity.CRITICAL)
        assert controller.get_active_severity() == Severity.CRITICAL


class TestResolveIncident:
    @pytest.mark.asyncio
    async def test_resolve_clears_open_incident(self, controller):
        incident = await controller.report(
            IncidentType.LIQUIDITY_STRESS, Severity.WARNING
        )
        await controller.resolve(incident.incident_id, notes="resolved by smoke test")
        assert controller.get_active_incidents() == []

    @pytest.mark.asyncio
    async def test_resolve_restores_multiplier_when_no_more_incidents(self, controller):
        incident = await controller.report(
            IncidentType.LIQUIDITY_STRESS, Severity.WARNING
        )
        await controller.resolve(incident.incident_id)
        assert controller.get_position_size_multiplier() == 1.0

    @pytest.mark.asyncio
    async def test_resolve_keeps_higher_severity_if_other_open(self, controller):
        inc_warn = await controller.report(IncidentType.LIQUIDITY_STRESS, Severity.WARNING)
        await controller.report(IncidentType.DRAWDOWN_BREACH, Severity.CRITICAL)
        await controller.resolve(inc_warn.incident_id)
        # Critical still active
        assert controller.get_active_severity() == Severity.CRITICAL
        assert controller.get_position_size_multiplier() == 0.0

    @pytest.mark.asyncio
    async def test_resolve_unknown_id_does_not_raise(self, controller):
        # Should not raise
        await controller.resolve("non-existent-id", "test")


class TestSummary:
    @pytest.mark.asyncio
    async def test_summary_reflects_active_state(self, controller):
        await controller.report(IncidentType.CORRELATION_CRISIS, Severity.CRITICAL)
        summary = controller.summary()
        assert summary["active_severity"] == "critical"
        assert summary["is_paused"] is True
        assert summary["open_incident_count"] == 1
        assert summary["position_size_multiplier"] == 0.0

    def test_summary_clean_initial(self, controller):
        summary = controller.summary()
        assert summary["active_severity"] is None
        assert summary["is_paused"] is False
        assert summary["open_incident_count"] == 0
