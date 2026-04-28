"""
Incident Controller — Safe-Mode Orchestration.

Implements escalation chains and fencing token-based mitigation for:
  - Broker latency spikes
  - Reconciliation mismatches
  - Portfolio drawdown breaches
  - Correlation crisis (> 0.75 avg pairwise correlation)
  - Liquidity stress (> 50 bps spread across universe)

Severity levels (from TRADING_BOT_PLAN.md §6יה)
------------------------------------------------
  WARNING   → throttle 50%, notify, continue
  CRITICAL  → pause new opens, alert escalation
  EMERGENCY → circuit breaker, reduce-only mode

Usage
-----
    controller = IncidentController(
        state_store=state_store,
        alert_dispatcher=alerts,
        event_bus=bus,
    )

    # Call from monitoring loops:
    await controller.report(
        incident_type=IncidentType.BROKER_LATENCY_SPIKE,
        severity=Severity.CRITICAL,
        details={"p95_ms": 2300, "threshold_ms": 2000},
    )

    # Check current throttle multiplier before placing orders:
    multiplier = controller.get_position_size_multiplier()  # e.g. 0.5 during WARNING
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase 5.5 — Cascade circuit-breaker constants
# ---------------------------------------------------------------------------
# If _CASCADE_THRESHOLD CRITICAL incidents accumulate within
# _CASCADE_WINDOW_SECONDS the controller auto-escalates to EMERGENCY.
_CASCADE_THRESHOLD: int = 3
_CASCADE_WINDOW_SECONDS: float = 60.0


class Severity(str, Enum):
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class IncidentType(str, Enum):
    BROKER_LATENCY_SPIKE = "broker_latency_spike"
    RECONCILIATION_MISMATCH = "reconciliation_mismatch"
    DRAWDOWN_BREACH = "drawdown_breach"
    CORRELATION_CRISIS = "correlation_crisis"
    LIQUIDITY_STRESS = "liquidity_stress"
    TCA_SLIPPAGE_BREACH = "tca_slippage_breach"
    MANUAL = "manual"


class IncidentStatus(str, Enum):
    OPEN = "open"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"


@dataclass
class Incident:
    incident_id: str
    incident_type: IncidentType
    severity: Severity
    status: IncidentStatus
    details: Dict[str, Any]
    opened_at: datetime
    resolved_at: Optional[datetime] = None
    mitigation_notes: str = ""
    fencing_token_id: Optional[str] = None


class IncidentController:
    """
    Escalation controller for runtime safety events.

    Parameters
    ----------
    state_store:
        RuntimeStateStore for state transitions (pause/circuit-breaker).
    alert_dispatcher:
        AlertDispatcher for Telegram/log notifications.
    event_bus:
        EventBus for publishing incident events to other subsystems.

    All parameters are optional for testability; pass None to disable that channel.
    """

    def __init__(
        self,
        state_store=None,
        alert_dispatcher=None,
        event_bus=None,
    ) -> None:
        self._state_store = state_store
        self._alerts = alert_dispatcher
        self._bus = event_bus

        self._incidents: Dict[str, Incident] = {}
        self._active_severity: Optional[Severity] = None
        self._position_size_multiplier: float = 1.0
        # Phase 5.5 — cascade detection: wall-clock timestamps of recent CRITICALs
        self._cascade_timestamps: List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def report(
        self,
        incident_type: IncidentType,
        severity: Severity,
        details: Optional[Dict[str, Any]] = None,
    ) -> Incident:
        """
        Report a new safety incident and trigger the appropriate escalation.

        Returns the created Incident record.
        """
        incident = Incident(
            incident_id=str(uuid.uuid4()),
            incident_type=incident_type,
            severity=severity,
            status=IncidentStatus.OPEN,
            details=details or {},
            opened_at=datetime.now(timezone.utc),
        )
        self._incidents[incident.incident_id] = incident

        logger.warning(
            "[incident] %s incident: type=%s details=%s",
            severity.value.upper(), incident_type.value, details
        )

        # Escalate based on severity
        if severity == Severity.WARNING:
            await self._handle_warning(incident)
        elif severity == Severity.CRITICAL:
            await self._handle_critical(incident)
        elif severity == Severity.EMERGENCY:
            await self._handle_emergency(incident)

        # Update active severity (ratchet up, not down)
        self._update_active_severity(severity)

        return incident

    async def resolve(self, incident_id: str, notes: str = "") -> None:
        """
        Mark an incident as resolved and restore normal operation
        if no other open incidents remain at the same or higher severity.
        """
        incident = self._incidents.get(incident_id)
        if incident is None:
            logger.warning("[incident] resolve: incident %s not found", incident_id)
            return

        incident.status = IncidentStatus.RESOLVED
        incident.resolved_at = datetime.now(timezone.utc)
        incident.mitigation_notes = notes

        logger.info(
            "[incident] Resolved %s (type=%s)", incident_id, incident.incident_type.value
        )

        # Recalculate active severity from remaining open incidents
        self._recalculate_active_severity()

    def get_position_size_multiplier(self) -> float:
        """
        Return the current position size multiplier.

          1.0 = normal operation
          0.5 = WARNING throttle
          0.0 = CRITICAL/EMERGENCY — no new positions
        """
        return self._position_size_multiplier

    def get_active_incidents(self) -> List[Incident]:
        """Return all currently open incidents."""
        return [i for i in self._incidents.values() if i.status == IncidentStatus.OPEN]

    def get_active_severity(self) -> Optional[Severity]:
        """Return the highest severity among open incidents."""
        return self._active_severity

    def is_paused(self) -> bool:
        """True if new order opens are blocked (CRITICAL or EMERGENCY)."""
        return self._active_severity in (Severity.CRITICAL, Severity.EMERGENCY)

    def summary(self) -> Dict[str, Any]:
        """Return a dict summary for health endpoints and dashboard."""
        return {
            "active_severity": self._active_severity.value if self._active_severity else None,
            "position_size_multiplier": self._position_size_multiplier,
            "is_paused": self.is_paused(),
            "open_incident_count": len(self.get_active_incidents()),
            "open_incidents": [
                {
                    "id": i.incident_id,
                    "type": i.incident_type.value,
                    "severity": i.severity.value,
                    "opened_at": i.opened_at.isoformat(),
                }
                for i in self.get_active_incidents()
            ],
        }

    # ------------------------------------------------------------------
    # Escalation handlers
    # ------------------------------------------------------------------

    async def _handle_warning(self, incident: Incident) -> None:
        """WARNING: throttle position sizes to 50%, alert, continue."""
        self._position_size_multiplier = min(self._position_size_multiplier, 0.5)
        logger.warning(
            "[incident] WARNING activated: position sizes throttled to 50%% (type=%s)",
            incident.incident_type.value,
        )
        await self._send_alert(
            f"⚠️ WARNING incident: {incident.incident_type.value}\n"
            f"Details: {incident.details}\n"
            "Position sizes throttled to 50%.",
            urgent=False,
        )
        await self._publish_event("incident.warning", incident)

    async def _handle_critical(self, incident: Incident) -> None:
        """CRITICAL: pause new opens (reduce-only), alert escalation."""
        self._position_size_multiplier = 0.0

        # Pause via RuntimeStateStore
        if self._state_store:
            try:
                await self._state_store.pause(
                    reason=f"incident_controller:{incident.incident_type.value}"
                )
            except Exception as exc:
                logger.error("[incident] state_store.pause() failed: %s", exc)

        logger.error(
            "[incident] CRITICAL activated: new orders paused (type=%s)",
            incident.incident_type.value,
        )
        await self._send_alert(
            f"🚨 CRITICAL incident: {incident.incident_type.value}\n"
            f"Details: {incident.details}\n"
            "New order opens PAUSED. Manual review required.",
            urgent=True,
        )
        await self._publish_event("incident.critical", incident)

        # Phase 5.5 — cascade detection: auto-escalate to EMERGENCY if
        # _CASCADE_THRESHOLD CRITICALs accumulate within _CASCADE_WINDOW_SECONDS.
        now = time.monotonic()
        self._cascade_timestamps.append(now)
        # Prune stale timestamps outside the window
        cutoff = now - _CASCADE_WINDOW_SECONDS
        self._cascade_timestamps = [t for t in self._cascade_timestamps if t >= cutoff]
        if len(self._cascade_timestamps) >= _CASCADE_THRESHOLD:
            logger.critical(
                "[incident] CASCADE DETECTED: %d CRITICAL incidents in %.0fs — "
                "auto-activating circuit breaker (EMERGENCY)",
                len(self._cascade_timestamps),
                _CASCADE_WINDOW_SECONDS,
            )
            self._cascade_timestamps.clear()
            cascade_incident = Incident(
                incident_id=str(uuid.uuid4()),
                incident_type=IncidentType.MANUAL,
                severity=Severity.EMERGENCY,
                status=IncidentStatus.OPEN,
                details={
                    "reason": "cascade_auto_activation",
                    "trigger_incident": incident.incident_id,
                },
                opened_at=datetime.now(timezone.utc),
            )
            self._incidents[cascade_incident.incident_id] = cascade_incident
            await self._handle_emergency(cascade_incident)
            self._update_active_severity(Severity.EMERGENCY)

    async def _handle_emergency(self, incident: Incident) -> None:
        """EMERGENCY: circuit breaker, reduce-only mode, maximum alert."""
        self._position_size_multiplier = 0.0

        # Activate emergency via RuntimeStateStore
        if self._state_store:
            try:
                # Transition to emergency (reduce-only) state
                await self._state_store.emergency_stop(
                    reason=f"incident_controller:{incident.incident_type.value}"
                )
            except AttributeError:
                # Fallback: use pause if emergency_stop not implemented
                try:
                    await self._state_store.pause(
                        reason=f"EMERGENCY:{incident.incident_type.value}"
                    )
                except Exception as exc:
                    logger.error("[incident] state_store transition failed: %s", exc)
            except Exception as exc:
                logger.error("[incident] emergency_stop failed: %s", exc)

        logger.critical(
            "[incident] EMERGENCY: circuit breaker activated (type=%s details=%s)",
            incident.incident_type.value, incident.details,
        )
        await self._send_alert(
            f"🔴 EMERGENCY: {incident.incident_type.value}\n"
            f"Details: {incident.details}\n"
            "CIRCUIT BREAKER ACTIVATED — reduce-only mode.\n"
            "Manual intervention required immediately.",
            urgent=True,
        )
        await self._publish_event("incident.emergency", incident)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _send_alert(self, message: str, urgent: bool = False) -> None:
        if self._alerts is None:
            logger.info("[incident] alert (no dispatcher): %s", message[:100])
            return
        try:
            if urgent and hasattr(self._alerts, "send_urgent"):
                await self._alerts.send_urgent(message)
            else:
                await self._alerts.send(message)
        except Exception as exc:
            logger.error("[incident] alert dispatch failed: %s", exc)

    async def _publish_event(self, topic: str, incident: Incident) -> None:
        if self._bus is None:
            return
        try:
            await self._bus.publish(
                topic,
                {
                    "incident_id": incident.incident_id,
                    "incident_type": incident.incident_type.value,
                    "severity": incident.severity.value,
                    "details": incident.details,
                    "opened_at": incident.opened_at.isoformat(),
                },
            )
        except Exception as exc:
            logger.error("[incident] event publish failed: %s", exc)

    def _update_active_severity(self, new_severity: Severity) -> None:
        _rank = {Severity.WARNING: 1, Severity.CRITICAL: 2, Severity.EMERGENCY: 3}
        if (
            self._active_severity is None
            or _rank[new_severity] > _rank[self._active_severity]
        ):
            self._active_severity = new_severity

    def _recalculate_active_severity(self) -> None:
        open_incidents = self.get_active_incidents()
        if not open_incidents:
            self._active_severity = None
            self._position_size_multiplier = 1.0
            logger.info("[incident] All incidents resolved — normal operation restored.")
            return

        _rank = {Severity.WARNING: 1, Severity.CRITICAL: 2, Severity.EMERGENCY: 3}
        highest = max(open_incidents, key=lambda i: _rank[i.severity])
        self._active_severity = highest.severity

        if highest.severity == Severity.WARNING:
            self._position_size_multiplier = 0.5
        else:
            self._position_size_multiplier = 0.0
