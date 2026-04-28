"""
SLO Error Budget Burn-Rate Tracker.

Tracks whether the error budget for each SLO is being consumed
faster than the allowed burn rate, and emits alerts when breached.

SLOs tracked (configurable):
  - latency_p95       : p95 order-to-fill latency ≤ 500 ms
  - fill_rate         : ≥ 95% of orders filled within TTL
  - reconciliation    : Zero unreconciled positions for > 5 minutes
  - data_freshness    : Bars arrive within 90 seconds of market close

Burn-rate model (Google SRE):
  - 30-day rolling window
  - Budget = (1 - target) × window
  - Alert when 1h burn rate × 14.4 > budget remaining
    (= consuming 2× daily budget in 1 hour)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SloDefinition:
    name: str
    target: float      # e.g. 0.95 for 95%
    window_seconds: int = 30 * 86400  # 30 days


@dataclass
class SloEvent:
    timestamp: float   # unix time
    success: bool      # True = good event, False = bad event


@dataclass
class BurnRateAlert:
    slo_name: str
    burn_rate: float
    budget_remaining_pct: float
    threshold: float


DEFAULT_SLOS = [
    SloDefinition("latency_p95", target=0.95),
    SloDefinition("fill_rate", target=0.95),
    SloDefinition("reconciliation", target=0.999),
    SloDefinition("data_freshness", target=0.95),
]


class SloBudgetTracker:
    """
    Tracks SLO error budget burn rates.

    Usage::

        tracker = SloBudgetTracker()
        tracker.record("fill_rate", success=True)
        tracker.record("fill_rate", success=False)
        alerts = tracker.check_burn_rates()
    """

    # Alert when 1h burn rate × 14.4 exceeds this multiple of the daily budget
    FAST_BURN_THRESHOLD = 2.0
    # Also alert on slow burn: 6h burn rate × 2.4 > 1 (consumes full budget in window)
    SLOW_BURN_THRESHOLD = 1.0

    def __init__(self, slos: list[SloDefinition] | None = None) -> None:
        self._slos: dict[str, SloDefinition] = {
            s.name: s for s in (slos or DEFAULT_SLOS)
        }
        # Rolling deque of events per SLO (bounded to avoid unbounded memory)
        self._events: dict[str, deque[SloEvent]] = {
            name: deque(maxlen=100_000) for name in self._slos
        }

    # ------------------------------------------------------------------
    # Record
    # ------------------------------------------------------------------

    def record(self, slo_name: str, success: bool) -> None:
        if slo_name not in self._events:
            logger.warning("slo_budget.record unknown_slo=%s", slo_name)
            return
        self._events[slo_name].append(SloEvent(timestamp=time.time(), success=success))

    def record_latency(self, latency_ms: float, threshold_ms: float = 500.0) -> None:
        self.record("latency_p95", success=latency_ms <= threshold_ms)

    def record_fill(self, filled: bool) -> None:
        self.record("fill_rate", success=filled)

    def record_reconciliation(self, clean: bool) -> None:
        self.record("reconciliation", success=clean)

    def record_data_freshness(self, fresh: bool) -> None:
        self.record("data_freshness", success=fresh)

    # ------------------------------------------------------------------
    # Burn-rate calculation
    # ------------------------------------------------------------------

    def _error_rate(self, slo_name: str, window_seconds: float) -> float:
        """Fraction of bad events in the last window_seconds."""
        now = time.time()
        events = [e for e in self._events[slo_name] if now - e.timestamp <= window_seconds]
        if not events:
            return 0.0
        bad = sum(1 for e in events if not e.success)
        return bad / len(events)

    def burn_rate(self, slo_name: str, window_seconds: float = 3600.0) -> float:
        """
        Burn rate = actual error rate / allowed error rate.
        > 1 means budget being consumed; > 14.4 is a fast burn.
        """
        slo = self._slos.get(slo_name)
        if slo is None:
            return 0.0
        allowed_error_rate = 1.0 - slo.target
        if allowed_error_rate <= 0:
            return float("inf")
        actual = self._error_rate(slo_name, window_seconds)
        return actual / allowed_error_rate

    def budget_remaining_pct(self, slo_name: str) -> float:
        """Remaining error budget as a fraction (1.0 = full, 0.0 = exhausted)."""
        slo = self._slos.get(slo_name)
        if slo is None:
            return 1.0
        allowed_error_rate = 1.0 - slo.target
        if allowed_error_rate <= 0:
            return 0.0
        actual = self._error_rate(slo_name, slo.window_seconds)
        consumed = actual / allowed_error_rate
        return max(0.0, 1.0 - consumed)

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def check_burn_rates(self) -> list[BurnRateAlert]:
        """
        Returns list of active BurnRateAlert objects.
        Empty list = all SLOs healthy.
        """
        alerts: list[BurnRateAlert] = []
        for name in self._slos:
            # Fast burn: 1-hour window
            br_1h = self.burn_rate(name, window_seconds=3600)
            projected_1h = br_1h * (1 / 24 / 30)  # fraction of monthly budget per hour
            if br_1h > 14.4 * self.FAST_BURN_THRESHOLD:
                alerts.append(BurnRateAlert(
                    slo_name=name,
                    burn_rate=br_1h,
                    budget_remaining_pct=self.budget_remaining_pct(name),
                    threshold=14.4 * self.FAST_BURN_THRESHOLD,
                ))
                logger.warning("slo_budget.fast_burn slo=%s burn_rate=%.1f", name, br_1h)
                continue

            # Slow burn: 6-hour window
            br_6h = self.burn_rate(name, window_seconds=6 * 3600)
            if br_6h > 6.0:
                alerts.append(BurnRateAlert(
                    slo_name=name,
                    burn_rate=br_6h,
                    budget_remaining_pct=self.budget_remaining_pct(name),
                    threshold=6.0,
                ))
                logger.warning("slo_budget.slow_burn slo=%s burn_rate=%.1f", name, br_6h)

        return alerts

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, dict]:
        return {
            name: {
                "burn_rate_1h": round(self.burn_rate(name, 3600), 2),
                "burn_rate_6h": round(self.burn_rate(name, 6 * 3600), 2),
                "budget_remaining_pct": round(self.budget_remaining_pct(name) * 100, 1),
                "total_events": len(self._events[name]),
            }
            for name in self._slos
        }
