"""
Position Sizing Pipeline — Phase 1 (Fixed-Fractional, Lean).

Refactors the flat calculate_position_size() function into a 5-step
pipeline of independent SizingStep classes, each testable in isolation.

Pipeline steps (in order):
  1. FixedFractionalStep    — base size from risk% / stop%
  2. VolScaleStep           — scale by target_vol / realized_vol (optional)
  3. LiquidityCapStep       — cap to adv_fraction × ADV
  4. GlobalRiskBudgetStep   — scale down if budget exhausted
  5. HardCapStep            — absolute position size cap

Each step receives a SizingContext and float (current size), returns float.

Usage
-----
    pipeline = PositionSizingPipeline.default(settings.risk)
    shares = pipeline.size(signal, ctx, last_price)
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from ..signals.models import SignalIntent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared context
# ---------------------------------------------------------------------------


@dataclass
class SizingContext:
    """Immutable snapshot of portfolio state used by all pipeline steps."""

    portfolio_value: float  # current equity (USD)
    current_open_risk: float  # sum of open positions' risk as fraction
    realized_vol: float = 0.0  # annualised realized vol (fraction, e.g. 0.18)
    adv_dollars: float = 0.0  # average daily volume in USD for the symbol
    rolling_sharpe_60d: float = 0.0  # 60-day rolling Sharpe (for future Kelly)


# ---------------------------------------------------------------------------
# Step ABC
# ---------------------------------------------------------------------------


class SizingStep(ABC):
    """A single, independently-testable step in the sizing pipeline."""

    @abstractmethod
    def apply(
        self,
        size_dollars: float,
        signal: SignalIntent,
        ctx: SizingContext,
    ) -> float:
        """Return the (possibly adjusted) dollar size."""


# ---------------------------------------------------------------------------
# Concrete steps
# ---------------------------------------------------------------------------


class FixedFractionalStep(SizingStep):
    """
    Base size = (portfolio_value × max_risk_per_trade) / stop_distance_pct.

    This is the Phase 1 core sizing formula. Result is capped at
    absolute_max_position_pct × portfolio_value.
    """

    def __init__(
        self,
        max_risk_per_trade: float,
        absolute_max_position_pct: float,
        stop_loss_floor_pct: float = 0.005,
    ) -> None:
        self._max_risk = max_risk_per_trade
        self._abs_max_pct = absolute_max_position_pct
        self._floor = stop_loss_floor_pct

    def apply(
        self, size_dollars: float, signal: SignalIntent, ctx: SizingContext
    ) -> float:
        if ctx.portfolio_value <= 0:
            return 0.0
        stop_pct = max(signal.stop_distance_pct or self._floor, self._floor)
        base = (ctx.portfolio_value * self._max_risk) / stop_pct
        cap = ctx.portfolio_value * self._abs_max_pct
        return min(base, cap)


class VolScaleStep(SizingStep):
    """
    Scale size by (target_vol / realized_vol), clamped to [min_scale, max_scale].

    Skipped (returns size unchanged) when realized_vol == 0.
    """

    def __init__(
        self,
        target_vol: float = 0.15,
        min_scale: float = 0.5,
        max_scale: float = 2.0,
    ) -> None:
        self._target = target_vol
        self._min = min_scale
        self._max = max_scale

    def apply(
        self, size_dollars: float, signal: SignalIntent, ctx: SizingContext
    ) -> float:
        if ctx.realized_vol <= 0:
            return size_dollars
        scale = min(max(self._target / ctx.realized_vol, self._min), self._max)
        return size_dollars * scale


class LiquidityCapStep(SizingStep):
    """
    Cap position to adv_fraction × daily ADV in dollars.

    If ctx.adv_dollars == 0 (not provided), this step is a no-op.
    """

    def __init__(self, adv_fraction: float = 0.05) -> None:
        self._adv_frac = adv_fraction

    def apply(
        self, size_dollars: float, signal: SignalIntent, ctx: SizingContext
    ) -> float:
        if ctx.adv_dollars <= 0:
            return size_dollars
        return min(size_dollars, ctx.adv_dollars * self._adv_frac)


class GlobalRiskBudgetStep(SizingStep):
    """
    If (current_open_risk + size_risk) > max_global_open_risk, scale down.
    """

    def __init__(self, max_global_open_risk: float) -> None:
        self._max = max_global_open_risk

    def apply(
        self, size_dollars: float, signal: SignalIntent, ctx: SizingContext
    ) -> float:
        if ctx.portfolio_value <= 0:
            return 0.0
        stop_pct = max(signal.stop_distance_pct or 0.005, 0.005)
        size_risk = (size_dollars * stop_pct) / ctx.portfolio_value
        remaining = self._max - ctx.current_open_risk
        if remaining <= 0:
            logger.info(
                "[sizing/budget] %s rejected: global risk budget exhausted "
                "(open=%.3f%% >= max=%.3f%%)",
                signal.symbol,
                ctx.current_open_risk * 100,
                self._max * 100,
            )
            return 0.0
        if size_risk > remaining:
            scale = remaining / size_risk
            logger.debug(
                "[sizing/budget] %s scaled to %.1f%% to fit risk budget",
                signal.symbol,
                scale * 100,
            )
            return size_dollars * scale
        return size_dollars


class HardCapStep(SizingStep):
    """
    Absolute hard cap: size <= portfolio_value × hard_cap_pct.
    """

    def __init__(self, hard_cap_pct: float) -> None:
        self._cap = hard_cap_pct

    def apply(
        self, size_dollars: float, signal: SignalIntent, ctx: SizingContext
    ) -> float:
        return min(size_dollars, ctx.portfolio_value * self._cap)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PositionSizingPipeline:
    """
    Runs SignalIntent through an ordered list of SizingStep instances,
    then converts the final dollar size to whole share count.
    """

    def __init__(self, steps: List[SizingStep]) -> None:
        self._steps = steps

    @classmethod
    def default(
        cls,
        max_risk_per_trade: float = 0.01,
        absolute_max_position_pct: float = 0.03,
        max_global_open_risk: float = 0.02,
        stop_loss_floor_pct: float = 0.005,
        target_vol: float = 0.15,
        adv_fraction: float = 0.05,
    ) -> "PositionSizingPipeline":
        """Factory with standard Phase 1 configuration."""
        return cls(
            steps=[
                FixedFractionalStep(
                    max_risk_per_trade=max_risk_per_trade,
                    absolute_max_position_pct=absolute_max_position_pct,
                    stop_loss_floor_pct=stop_loss_floor_pct,
                ),
                VolScaleStep(target_vol=target_vol),
                LiquidityCapStep(adv_fraction=adv_fraction),
                GlobalRiskBudgetStep(max_global_open_risk=max_global_open_risk),
                HardCapStep(hard_cap_pct=absolute_max_position_pct),
            ]
        )

    def size(
        self,
        signal: SignalIntent,
        ctx: SizingContext,
        last_price: float,
    ) -> int:
        """
        Run the pipeline and return whole share count.

        Returns 0 if any step reduces size to <= 0 or last_price is invalid.
        """
        if last_price <= 0:
            logger.warning(
                "[sizing] %s: invalid last_price=%.4f", signal.symbol, last_price
            )
            return 0

        size = 0.0
        for step in self._steps:
            size = step.apply(size, signal, ctx)
            if size <= 0:
                logger.debug(
                    "[sizing] %s: rejected at step %s",
                    signal.symbol,
                    type(step).__name__,
                )
                return 0

        shares = math.floor(size / last_price)
        if shares <= 0:
            return 0

        logger.info(
            "[sizing] %s: %d shares @ $%.2f  (size_dollars=%.2f)",
            signal.symbol,
            shares,
            last_price,
            size,
        )
        return shares
