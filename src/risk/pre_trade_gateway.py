"""
Pre-Trade Risk Gateway — Five-gate admission control.

Evaluates an ordered sequence of RiskGate policies against a SignalIntent.
Gates are injected at construction time (Dependency Injection). Adding a
new policy requires only creating a new RiskGate subclass and appending it
to the injected list — no modification to this class (OCP).

Gate execution contract
-----------------------
  - Gates execute in order; the first rejection ends evaluation.
  - If a gate returns modified_signal, the modified copy is forwarded
    to all subsequent gates. The final approved result carries the
    fully-modified signal.
  - Latency of every gate evaluation is recorded for observability.

Canonical gate order (from TRADING_BOT_PLAN.md §6יג):
    1. SignalViabilityGate     — EV > 1.5× cost
    2. PortfolioRiskGate       — sector / correlation / concentration
    3. LiquidityGate           — spread / depth / ADV
    4. TailRiskGate            — ES / VaR / crisis correlation
    5. ExecutionReadinessGate  — broker latency / TCA health

Wire-up example (main.py)
--------------------------
    from src.risk.gates import (
        SignalViabilityGate, PortfolioRiskGate, LiquidityGate,
        TailRiskGate, ExecutionReadinessGate,
    )
    from src.risk.pre_trade_gateway import PreTradeGateway

    gateway = PreTradeGateway(gates=[
        SignalViabilityGate(cost_estimator, performance_metrics),
        PortfolioRiskGate(portfolio_provider, max_sector_exposure=regime_sector_limit),
        LiquidityGate(market_data),
        TailRiskGate(risk_engine),
        ExecutionReadinessGate(tca_metrics),
        # Adding a 6th gate: zero changes to this file
    ])

    result = await gateway.admit_order(signal, portfolio_state)
    if not result.approved:
        log.info("Signal rejected: %s", result.reason)
        return

    # result.modified_signal is the (possibly-modified) signal to hand
    # to the Portfolio layer for final sizing.
    final_signal = result.modified_signal or signal
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

from .gates.base_gate import GateResult, RiskGate

logger = logging.getLogger(__name__)


class PreTradeGateway:
    """
    Ordered admission pipeline. Composes any number of RiskGate instances.

    Thread-safe: each call to admit_order() is independent and carries
    no mutable state in this class.
    """

    def __init__(self, gates: Sequence[RiskGate]) -> None:
        if not gates:
            raise ValueError(
                "PreTradeGateway requires at least one gate. "
                "Provide an ordered list of RiskGate instances."
            )
        self._gates = list(gates)
        logger.info(
            "PreTradeGateway initialised with %d gate(s): [%s]",
            len(self._gates),
            ", ".join(g.gate_name for g in self._gates),
        )

    async def admit_order(
        self,
        signal,
        portfolio_state: dict,
    ) -> GateResult:
        """
        Evaluate the signal through all gates in order.

        Returns the final GateResult:
          - approved=False  → first rejection encountered (short-circuits)
          - approved=True   → all gates passed; modified_signal is the
                              (potentially-modified) signal to forward
                              to the Portfolio layer.

        All gate evaluations are logged at DEBUG level with per-gate
        latency in microseconds for profiling.
        """
        current_signal = signal
        pipeline_start_ns = time.perf_counter_ns()

        for gate in self._gates:
            gate_start_ns = time.perf_counter_ns()
            result = await gate.evaluate(current_signal, portfolio_state)
            gate_latency_us = (time.perf_counter_ns() - gate_start_ns) / 1_000

            logger.debug(
                "Gate [%-30s]  symbol=%-6s  approved=%-5s  "
                "reason=%-50s  latency=%.1f µs",
                gate.gate_name,
                getattr(current_signal, "symbol", "?"),
                result.approved,
                result.reason,
                gate_latency_us,
            )

            if not result.approved:
                total_latency_us = (time.perf_counter_ns() - pipeline_start_ns) / 1_000
                logger.info(
                    "PreTradeGateway REJECTED symbol=%s  gate=%s  reason=%s  "
                    "total_latency=%.1f µs",
                    getattr(current_signal, "symbol", "?"),
                    gate.gate_name,
                    result.reason,
                    total_latency_us,
                )
                return result

            # Propagate any signal modification (e.g. reduced qty from TailRiskGate)
            if result.modified_signal is not None:
                current_signal = result.modified_signal

        total_latency_us = (time.perf_counter_ns() - pipeline_start_ns) / 1_000
        logger.debug(
            "PreTradeGateway APPROVED symbol=%s  total_latency=%.1f µs",
            getattr(current_signal, "symbol", "?"),
            total_latency_us,
        )
        # Return approved with the final (possibly-modified) signal
        return GateResult.approve(modified_signal=current_signal)
