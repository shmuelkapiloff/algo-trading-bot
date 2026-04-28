"""
Abstract base for Pre-Trade Risk Gates.

Each gate encapsulates exactly one admission policy (SRP).
The PreTradeGateway composes an ordered list of gates (OCP):
adding a new policy means creating a new class and injecting it —
no existing gate or the gateway itself is modified.

Gate contract
-------------
  Input:  (signal: SignalIntent, portfolio_state: dict)
  Output: GateResult(approved, reason, modified_signal)

A gate may modify the signal (e.g. reduce qty during correlation crisis)
by returning GateResult with modified_signal set. The PreTradeGateway
forwards the modified copy to all subsequent gates.

Dependencies
------------
Each gate receives its collaborators via __init__ (constructor injection).
This makes every gate independently unit-testable with simple stubs:

    gate = TailRiskGate(risk_engine=MockRiskEngine())
    result = await gate.evaluate(signal, portfolio_state={})
    assert result.approved
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GateResult:
    """
    Result of a single gate evaluation.

    approved        True if this gate passes the signal.
    reason          Machine-readable code. Approval: "approved" or
                    "warning:<detail>". Rejection: "rejection_code:detail".
    modified_signal If a gate adjusts the signal (e.g. halves qty),
                    it returns the modified SignalIntent here.
                    None means no modification — the original signal is forwarded.
    """

    approved: bool
    reason: str
    modified_signal: Optional[object] = field(default=None)

    @classmethod
    def approve(
        cls,
        reason: str = "approved",
        modified_signal: Optional[object] = None,
    ) -> "GateResult":
        return cls(approved=True, reason=reason, modified_signal=modified_signal)

    @classmethod
    def reject(cls, reason: str) -> "GateResult":
        return cls(approved=False, reason=reason)


class RiskGate(ABC):
    """
    Single admission policy for the Pre-Trade Risk Gateway.

    Stateless per-evaluation. Shared state (portfolio positions,
    market data, TCA metrics) is injected through the constructor
    and accessed via the injected collaborator — not via globals.
    """

    @property
    def gate_name(self) -> str:
        """Used for logging and metrics labels. Override if desired."""
        return self.__class__.__name__

    @abstractmethod
    async def evaluate(
        self,
        signal: object,
        portfolio_state: dict,
    ) -> GateResult:
        """
        Evaluate a single admission criterion.

        Args:
            signal:           SignalIntent (may already be modified by earlier gates).
            portfolio_state:  Snapshot of current portfolio context. Expected keys:
                                "positions"         list of current open positions
                                "total_value"       float, portfolio NAV in USD
                                "risk_budget_used"  float, fraction of global risk consumed
                                "regime"            str, current market regime

        Returns:
            GateResult. If approved=False, the gateway stops and returns this result.
            If modified_signal is set, it replaces the input for subsequent gates.
        """
        ...
