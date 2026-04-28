"""
Deterministic Replay Harness — Phase 5.5.

Provides seed-based, fully deterministic simulation of the OMS event stream
with ledger validation.  Used to:

  1. Reproduce any production sequence of events offline.
  2. Verify that OMS state after replay matches expected terminal states.
  3. Run regression checks — same seed must always produce identical ledger.

Design
------
- ``ReplayEvent`` is the unit of replay: a (seq, event_type, payload) tuple.
- ``ReplayHarness`` drives events through an ``OrderStateMachine`` instance
  in strict sequence order.
- After replay, ``validate_ledger()`` compares final states against
  ``expected_states: dict[order_id → OrderState]`` and returns a
  ``ReplayValidationResult`` with per-order pass/fail details.
- ``seed`` is stored in every ``ReplayResult`` so runs are reproducible.

Usage
-----
    events = [
        ReplayEvent(seq=0, order_id="A", event_type="register", payload={...}),
        ReplayEvent(seq=1, order_id="A", event_type="transition",
                    payload={"target": "SENT"}),
        ...
    ]

    harness = ReplayHarness(seed=42)
    result = harness.run(events)

    validation = harness.validate_ledger(
        result, expected={"A": OrderState.FILLED}
    )
    assert validation.all_pass, validation.failures
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from trading_bot.src.execution.order_state_machine import (
    OrderRecord,
    OrderState,
    OrderStateMachine,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ReplayEvent:
    """A single event in the replay stream."""

    seq: int
    order_id: str
    # event_type: "register" | "transition" | "noop"
    event_type: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_type not in {"register", "transition", "noop"}:
            raise ValueError(f"Unknown event_type: {self.event_type!r}")


@dataclass
class ReplayOrderSummary:
    order_id: str
    final_state: Optional[OrderState]
    last_seq: int
    pending_count: int


@dataclass
class ReplayResult:
    """Output of a single ``ReplayHarness.run()`` call."""

    seed: int
    events_processed: int
    events_skipped: int  # noop or unrecognised
    orders: Dict[str, ReplayOrderSummary]
    # SHA-256 of canonical ledger JSON — used for regression comparison
    ledger_digest: str
    elapsed_s: float


@dataclass
class ValidationFailure:
    order_id: str
    expected: OrderState
    actual: Optional[OrderState]


@dataclass
class ReplayValidationResult:
    all_pass: bool
    failures: List[ValidationFailure] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class ReplayHarness:
    """
    Seed-based deterministic OMS replay harness.

    Parameters
    ----------
    seed:
        Random seed stored in the result for reproducibility.
        The harness itself is deterministic — the seed is just metadata
        unless the caller uses ``harness.rng`` for jitter injection.
    oms:
        Optionally inject a pre-configured ``OrderStateMachine``.
        If omitted a fresh instance is created per ``run()`` call.
    """

    def __init__(self, seed: int = 0, oms: Optional[OrderStateMachine] = None) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self._oms = oms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, events: List[ReplayEvent]) -> ReplayResult:
        """
        Replay all events in strict ``seq`` order.

        Events are sorted by ``(order_id, seq)`` before processing so
        callers need not pre-sort.  Out-of-order events are handled by
        ``transition_ordered()`` (buffered and flushed automatically).
        """
        oms = self._oms if self._oms is not None else OrderStateMachine()

        t0 = time.monotonic()
        processed = skipped = 0

        # Sort by seq then order_id for determinism
        sorted_events = sorted(events, key=lambda e: (e.seq, e.order_id))

        for ev in sorted_events:
            if ev.event_type == "noop":
                skipped += 1
                continue

            if ev.event_type == "register":
                record = OrderRecord(
                    order_id=ev.order_id,
                    symbol=ev.payload.get("symbol", "UNKNOWN"),
                    strategy_name=ev.payload.get("strategy_name", "replay"),
                    submitted_qty=int(ev.payload.get("submitted_qty", 0)),
                    last_seq=ev.seq,  # register event owns this sequence slot
                )
                oms.register(record)
                processed += 1

            elif ev.event_type == "transition":
                raw_target = ev.payload.get("target", "")
                try:
                    target = OrderState(raw_target)
                except ValueError:
                    logger.warning(
                        "ReplayHarness: unknown OrderState %r for order %s seq %d — skipped",
                        raw_target, ev.order_id, ev.seq,
                    )
                    skipped += 1
                    continue

                oms.transition_ordered(
                    ev.order_id,
                    target,
                    seq=ev.seq,
                    broker_order_id=ev.payload.get("broker_order_id"),
                    filled_qty=ev.payload.get("filled_qty"),
                    fill_price=ev.payload.get("fill_price"),
                    error=ev.payload.get("error"),
                )
                processed += 1

        # Build result
        orders: Dict[str, ReplayOrderSummary] = {}
        for order_id, rec in oms._orders.items():  # noqa: SLF001
            orders[order_id] = ReplayOrderSummary(
                order_id=order_id,
                final_state=rec.state,
                last_seq=rec.last_seq,
                pending_count=len(oms._pending.get(order_id, {})),  # noqa: SLF001
            )

        digest = self._compute_digest(orders)

        return ReplayResult(
            seed=self.seed,
            events_processed=processed,
            events_skipped=skipped,
            orders=orders,
            ledger_digest=digest,
            elapsed_s=time.monotonic() - t0,
        )

    def validate_ledger(
        self,
        result: ReplayResult,
        expected: Dict[str, OrderState],
    ) -> ReplayValidationResult:
        """
        Compare replay result against expected terminal states.

        Returns ``ReplayValidationResult`` with ``all_pass=True`` if every
        order in ``expected`` matches its final state in the replay.
        """
        failures: List[ValidationFailure] = []
        for order_id, exp_state in expected.items():
            summary = result.orders.get(order_id)
            actual = summary.final_state if summary else None
            if actual != exp_state:
                failures.append(
                    ValidationFailure(
                        order_id=order_id,
                        expected=exp_state,
                        actual=actual,
                    )
                )
        return ReplayValidationResult(all_pass=not failures, failures=failures)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_digest(orders: Dict[str, ReplayOrderSummary]) -> str:
        """SHA-256 of a canonical JSON representation of the final ledger."""
        canonical = {
            oid: {
                "state": s.final_state.value if s.final_state else None,
                "last_seq": s.last_seq,
                "pending_count": s.pending_count,
            }
            for oid, s in sorted(orders.items())
        }
        return hashlib.sha256(
            json.dumps(canonical, sort_keys=True).encode()
        ).hexdigest()
