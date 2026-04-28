"""
Tests for PreTradeGateway and all five RiskGate implementations.

Covers:
  - Gateway short-circuits on first rejection
  - Gateway forwards modified_signal between gates
  - Gateway approves when all gates pass
  - SignalViabilityGate rejects when EV < threshold
  - PortfolioRiskGate rejects on sector / correlation / concentration
  - LiquidityGate rejects on wide spread / ADV breach / thin depth
  - TailRiskGate rejects on ES / VaR breach; modifies qty in crisis
  - ExecutionReadinessGate rejects on latency / slippage / fill-rate breach

All collaborators are replaced with simple stubs (no external I/O).
Run: pytest tests/test_pre_trade_gateway.py -v
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock

import pytest

from trading_bot.src.signals.models import OrderSide, SignalIntent
from trading_bot.src.risk.gates.base_gate import GateResult, RiskGate
from trading_bot.src.risk.gates.signal_viability_gate import SignalViabilityGate
from trading_bot.src.risk.gates.portfolio_risk_gate import PortfolioRiskGate
from trading_bot.src.risk.gates.liquidity_gate import LiquidityGate
from trading_bot.src.risk.gates.tail_risk_gate import TailRiskGate
from trading_bot.src.risk.gates.execution_readiness_gate import ExecutionReadinessGate
from trading_bot.src.risk.pre_trade_gateway import PreTradeGateway


# ---------------------------------------------------------------------------
# Stub factories
# ---------------------------------------------------------------------------


def make_signal(
    symbol: str = "AAPL",
    confidence: float = 0.70,
    strategy_name: str = "momentum",
    qty: Optional[float] = 100.0,
) -> SignalIntent:
    return SignalIntent(
        symbol=symbol,
        side=OrderSide.BUY,
        strategy_name=strategy_name,
        confidence=confidence,
        qty=qty,
    )


class StubCostEstimator:
    def __init__(self, spread_bps: float = 5.0, slippage_bps: float = 3.0):
        self._spread = spread_bps
        self._slippage = slippage_bps

    def get_spread_bps(self, symbol: str) -> float:
        return self._spread

    def get_avg_slippage_bps(self) -> float:
        return self._slippage


class StubPerformanceMetrics:
    def __init__(self, expected_win: float = 0.020):
        self._win = expected_win

    def get_expected_win_per_signal(self, strategy_name: str) -> float:
        return self._win


class StubPortfolioProvider:
    def __init__(
        self,
        sector: str = "Technology",
        sector_exposure: float = 0.10,
        max_corr: float = 0.30,
        position_weight: float = 0.00,
    ):
        self._sector = sector
        self._sector_exposure = sector_exposure
        self._max_corr = max_corr
        self._position_weight = position_weight

    def get_sector(self, symbol: str) -> str:
        return self._sector

    def get_sector_exposure(self, sector: str) -> float:
        return self._sector_exposure

    def get_max_pairwise_correlation(self, symbol: str) -> float:
        return self._max_corr

    def get_position_weight(self, symbol: str) -> float:
        return self._position_weight


class StubMarketData:
    def __init__(
        self,
        spread_bps: float = 8.0,
        adv: float = 1_000_000.0,
        depth_usd: float = 500_000.0,
    ):
        self._spread = spread_bps
        self._adv = adv
        self._depth = depth_usd

    def get_spread_bps(self, symbol: str) -> float:
        return self._spread

    def get_adv(self, symbol: str) -> float:
        return self._adv

    def get_book_depth_usd(self, symbol: str) -> float:
        return self._depth


class StubRiskEngine:
    def __init__(
        self,
        es: float = 0.02,
        var: float = 0.01,
        avg_corr: float = 0.30,
    ):
        self._es = es
        self._var = var
        self._avg_corr = avg_corr

    def estimate_es_after_trade(self, signal, portfolio_state: dict) -> float:
        return self._es

    def estimate_var_after_trade(self, signal, portfolio_state: dict) -> float:
        return self._var

    def get_average_portfolio_correlation(self) -> float:
        return self._avg_corr


class StubTCAMetrics:
    def __init__(
        self,
        latency_p95_ms: float = 300.0,
        slippage_bps: float = 5.0,
        fill_rate: float = 0.95,
    ):
        self._latency = latency_p95_ms
        self._slippage = slippage_bps
        self._fill_rate = fill_rate

    def get_broker_latency_p95_ms(self) -> float:
        return self._latency

    def get_avg_slippage_bps(self) -> float:
        return self._slippage

    def get_fill_rate_p95(self) -> float:
        return self._fill_rate


# ---------------------------------------------------------------------------
# Helper: build a fully-passing gateway
# ---------------------------------------------------------------------------


def build_passing_gateway() -> PreTradeGateway:
    return PreTradeGateway(
        gates=[
            SignalViabilityGate(StubCostEstimator(), StubPerformanceMetrics()),
            PortfolioRiskGate(StubPortfolioProvider()),
            LiquidityGate(StubMarketData()),
            TailRiskGate(StubRiskEngine()),
            ExecutionReadinessGate(StubTCAMetrics()),
        ]
    )


# ---------------------------------------------------------------------------
# PreTradeGateway — structural tests
# ---------------------------------------------------------------------------


def test_gateway_requires_at_least_one_gate():
    with pytest.raises(ValueError, match="at least one gate"):
        PreTradeGateway(gates=[])


@pytest.mark.asyncio
async def test_gateway_approves_when_all_gates_pass():
    gw = build_passing_gateway()
    result = await gw.admit_order(make_signal(), portfolio_state={})
    assert result.approved is True


@pytest.mark.asyncio
async def test_gateway_short_circuits_on_first_rejection():
    """Second gate always rejects; we verify only the first rejection is returned."""

    class AlwaysReject(RiskGate):
        async def evaluate(self, signal, portfolio_state):
            return GateResult.reject("sentinel_rejection")

    class NeverReached(RiskGate):
        async def evaluate(self, signal, portfolio_state):
            raise AssertionError("This gate should never be reached")

    gw = PreTradeGateway(gates=[AlwaysReject(), NeverReached()])
    result = await gw.admit_order(make_signal(), portfolio_state={})
    assert result.approved is False
    assert result.reason == "sentinel_rejection"


@pytest.mark.asyncio
async def test_gateway_forwards_modified_signal_to_next_gate():
    """A gate that modifies qty; subsequent gate receives the reduced qty."""
    received_qty: list[Optional[float]] = []

    class HalveQty(RiskGate):
        async def evaluate(self, signal, portfolio_state):
            modified = signal.with_qty(signal.qty * 0.5)
            return GateResult.approve(modified_signal=modified)

    class RecordQty(RiskGate):
        async def evaluate(self, signal, portfolio_state):
            received_qty.append(signal.qty)
            return GateResult.approve()

    gw = PreTradeGateway(gates=[HalveQty(), RecordQty()])
    await gw.admit_order(make_signal(qty=200.0), portfolio_state={})
    assert received_qty == [100.0]


@pytest.mark.asyncio
async def test_gateway_approved_result_carries_modified_signal():
    """Final approved result has modified_signal set to the final version."""

    class DoubleQty(RiskGate):
        async def evaluate(self, signal, portfolio_state):
            return GateResult.approve(modified_signal=signal.with_qty(signal.qty * 2))

    gw = PreTradeGateway(gates=[DoubleQty()])
    result = await gw.admit_order(make_signal(qty=50.0), portfolio_state={})
    assert result.approved is True
    assert result.modified_signal.qty == 100.0


# ---------------------------------------------------------------------------
# SignalViabilityGate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signal_viability_rejects_when_ev_below_threshold():
    # spread=10 bps, slippage=5 bps → total_cost_rt = 0.0030
    # threshold = 0.0030 × 1.5 = 0.0045
    # confidence=0.50 × expected_win=0.001 = 0.0005 → REJECT
    gate = SignalViabilityGate(
        StubCostEstimator(spread_bps=10.0, slippage_bps=5.0),
        StubPerformanceMetrics(expected_win=0.001),
    )
    result = await gate.evaluate(make_signal(confidence=0.50), {})
    assert result.approved is False
    assert "signal_ev_negative_after_costs" in result.reason


@pytest.mark.asyncio
async def test_signal_viability_approves_when_ev_above_threshold():
    # spread=5 bps, slippage=3 bps → total_cost_rt = 0.0016
    # threshold = 0.0016 × 1.5 = 0.0024
    # confidence=0.70 × expected_win=0.020 = 0.014 → APPROVE
    gate = SignalViabilityGate(
        StubCostEstimator(spread_bps=5.0, slippage_bps=3.0),
        StubPerformanceMetrics(expected_win=0.020),
    )
    result = await gate.evaluate(make_signal(confidence=0.70), {})
    assert result.approved is True


# ---------------------------------------------------------------------------
# PortfolioRiskGate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_portfolio_gate_rejects_on_sector_overweight():
    gate = PortfolioRiskGate(
        StubPortfolioProvider(sector_exposure=0.31),
        max_sector_exposure=0.30,
    )
    result = await gate.evaluate(make_signal(), {})
    assert result.approved is False
    assert "sector_overweight" in result.reason


@pytest.mark.asyncio
async def test_portfolio_gate_rejects_on_high_correlation():
    gate = PortfolioRiskGate(
        StubPortfolioProvider(max_corr=0.75),
        max_correlation=0.70,
    )
    result = await gate.evaluate(make_signal(), {})
    assert result.approved is False
    assert "correlation_cluster" in result.reason


@pytest.mark.asyncio
async def test_portfolio_gate_rejects_on_position_concentration():
    gate = PortfolioRiskGate(
        StubPortfolioProvider(position_weight=0.04),
        absolute_max_position_pct=0.03,
    )
    result = await gate.evaluate(make_signal(), {})
    assert result.approved is False
    assert "concentration_breach" in result.reason


# ---------------------------------------------------------------------------
# LiquidityGate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_liquidity_gate_rejects_on_wide_spread():
    gate = LiquidityGate(StubMarketData(spread_bps=20.0), max_spread_bps=15.0)
    result = await gate.evaluate(make_signal(), {})
    assert result.approved is False
    assert "spread_too_wide" in result.reason


@pytest.mark.asyncio
async def test_liquidity_gate_rejects_on_adv_breach():
    # qty=100, adv=1000 → 10% > max 2%
    gate = LiquidityGate(
        StubMarketData(adv=1000.0),
        max_order_adv_pct=0.02,
    )
    result = await gate.evaluate(make_signal(qty=100.0), {})
    assert result.approved is False
    assert "adv_exceeded" in result.reason


@pytest.mark.asyncio
async def test_liquidity_gate_skips_adv_check_when_qty_none():
    # qty is None → ADV check skipped; only spread and depth evaluated
    gate = LiquidityGate(
        StubMarketData(spread_bps=5.0, adv=100.0, depth_usd=100_000.0),
        max_order_adv_pct=0.02,
    )
    result = await gate.evaluate(make_signal(qty=None), {})
    assert result.approved is True


@pytest.mark.asyncio
async def test_liquidity_gate_rejects_on_thin_depth():
    gate = LiquidityGate(
        StubMarketData(depth_usd=10_000.0),
        min_book_depth_usd=50_000.0,
    )
    result = await gate.evaluate(make_signal(), {})
    assert result.approved is False
    assert "depth_insufficient" in result.reason


# ---------------------------------------------------------------------------
# TailRiskGate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tail_risk_gate_rejects_on_es_breach():
    gate = TailRiskGate(StubRiskEngine(es=0.06), es_max_pct_portfolio=0.05)
    result = await gate.evaluate(make_signal(), {})
    assert result.approved is False
    assert "es_breach" in result.reason


@pytest.mark.asyncio
async def test_tail_risk_gate_rejects_on_var_breach():
    gate = TailRiskGate(StubRiskEngine(var=0.04), var_max_pct_portfolio=0.03)
    result = await gate.evaluate(make_signal(), {})
    assert result.approved is False
    assert "var_breach" in result.reason


@pytest.mark.asyncio
async def test_tail_risk_gate_reduces_qty_in_crisis():
    gate = TailRiskGate(
        StubRiskEngine(avg_corr=0.80),
        correlation_crisis_threshold=0.75,
        crisis_size_reduction=0.50,
    )
    result = await gate.evaluate(make_signal(qty=200.0), {})
    assert result.approved is True
    assert "crisis_correlation" in result.reason
    assert result.modified_signal is not None
    assert result.modified_signal.qty == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_tail_risk_gate_approves_normally_below_crisis_threshold():
    gate = TailRiskGate(
        StubRiskEngine(es=0.02, var=0.01, avg_corr=0.50),
        correlation_crisis_threshold=0.75,
    )
    result = await gate.evaluate(make_signal(), {})
    assert result.approved is True
    assert result.modified_signal is None


# ---------------------------------------------------------------------------
# ExecutionReadinessGate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_gate_rejects_on_high_latency():
    gate = ExecutionReadinessGate(
        StubTCAMetrics(latency_p95_ms=2000.0),
        broker_latency_p95_max_ms=1500.0,
    )
    result = await gate.evaluate(make_signal(), {})
    assert result.approved is False
    assert "latency_breach" in result.reason


@pytest.mark.asyncio
async def test_execution_gate_rejects_on_high_slippage():
    gate = ExecutionReadinessGate(
        StubTCAMetrics(slippage_bps=30.0),
        slippage_pause_bps=25.0,
    )
    result = await gate.evaluate(make_signal(), {})
    assert result.approved is False
    assert "slippage_breach" in result.reason


@pytest.mark.asyncio
async def test_execution_gate_rejects_on_low_fill_rate():
    gate = ExecutionReadinessGate(
        StubTCAMetrics(fill_rate=0.50),
        min_fill_rate_p95=0.60,
    )
    result = await gate.evaluate(make_signal(), {})
    assert result.approved is False
    assert "fill_rate_breach" in result.reason


@pytest.mark.asyncio
async def test_execution_gate_approves_when_all_metrics_healthy():
    gate = ExecutionReadinessGate(
        StubTCAMetrics(latency_p95_ms=300.0, slippage_bps=5.0, fill_rate=0.95)
    )
    result = await gate.evaluate(make_signal(), {})
    assert result.approved is True
