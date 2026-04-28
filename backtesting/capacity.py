"""
Capacity Limits — per-symbol position size caps based on ADV, spread, and book depth.

Prevents the backtest (and live strategy) from assuming it can execute orders
larger than the market can absorb without significant market impact.

Rules (from TRADING_BOT_PLAN.md):
  - Max order size: 1% of ADV for liquid (Tier 1), 0.5% for less liquid (Tier 2)
  - Max position size: 5% of ADV (regardless of tier)
  - Spread > 50 bps: position size halved
  - Book depth check: estimated market impact must be < 10 bps for the order to proceed

Tiers (based on ADV):
  Tier 1: ADV > $5M   — 1.0% single-order cap
  Tier 2: ADV $1M-$5M — 0.5% single-order cap
  Tier 3: ADV < $1M   — 0.25% single-order cap (illiquid; rarely traded)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class LiquidityTier(str, Enum):
    TIER_1 = "tier_1"   # ADV > $5M
    TIER_2 = "tier_2"   # ADV $1M–$5M
    TIER_3 = "tier_3"   # ADV < $1M


# Max single-order size as fraction of ADV
_SINGLE_ORDER_CAP: dict[LiquidityTier, float] = {
    LiquidityTier.TIER_1: 0.010,
    LiquidityTier.TIER_2: 0.005,
    LiquidityTier.TIER_3: 0.0025,
}

# Max position size as fraction of ADV
_POSITION_CAP_PCT_ADV = 0.05


@dataclass
class CapacityResult:
    symbol: str
    tier: LiquidityTier
    adv_dollars: float
    max_order_dollars: float
    max_position_dollars: float
    spread_adjusted: bool       # True if spread > 50 bps forced a haircut
    market_impact_bps: float    # estimated one-way market impact
    allowed: bool               # False if order exceeds capacity


class CapacityChecker:
    """
    Checks whether a proposed order fits within market capacity constraints.

    Parameters
    ----------
    spread_haircut_threshold_bps : Halve size if spread > this (default 50 bps)
    impact_limit_bps             : Reject if estimated impact > this (default 10 bps)
    """

    def __init__(
        self,
        spread_haircut_threshold_bps: float = 50.0,
        impact_limit_bps: float = 10.0,
    ) -> None:
        self.spread_haircut_threshold = spread_haircut_threshold_bps
        self.impact_limit_bps = impact_limit_bps

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def classify_tier(adv_dollars: float) -> LiquidityTier:
        if adv_dollars >= 5_000_000:
            return LiquidityTier.TIER_1
        if adv_dollars >= 1_000_000:
            return LiquidityTier.TIER_2
        return LiquidityTier.TIER_3

    @staticmethod
    def estimate_market_impact_bps(
        order_dollars: float,
        adv_dollars: float,
        spread_bps: float = 10.0,
    ) -> float:
        """
        Simplified square-root market impact model:
          impact = spread * sqrt(order_size / ADV) * 0.5
        Returns impact in bps.
        """
        if adv_dollars <= 0:
            return 9999.0
        participation = order_dollars / adv_dollars
        return spread_bps * (participation ** 0.5) * 0.5

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------

    def check(
        self,
        symbol: str,
        order_dollars: float,
        adv_dollars: float,
        spread_bps: float = 10.0,
    ) -> CapacityResult:
        tier = self.classify_tier(adv_dollars)
        single_cap_pct = _SINGLE_ORDER_CAP[tier]
        max_single = adv_dollars * single_cap_pct
        max_position = adv_dollars * _POSITION_CAP_PCT_ADV

        spread_adjusted = False
        if spread_bps > self.spread_haircut_threshold:
            max_single *= 0.5
            max_position *= 0.5
            spread_adjusted = True

        impact_bps = self.estimate_market_impact_bps(order_dollars, adv_dollars, spread_bps)
        allowed = (
            order_dollars <= max_single
            and impact_bps <= self.impact_limit_bps
        )

        if not allowed:
            logger.debug(
                "capacity.rejected symbol=%s order=$%.0f max=$%.0f impact=%.1fbps",
                symbol, order_dollars, max_single, impact_bps,
            )

        return CapacityResult(
            symbol=symbol,
            tier=tier,
            adv_dollars=adv_dollars,
            max_order_dollars=max_single,
            max_position_dollars=max_position,
            spread_adjusted=spread_adjusted,
            market_impact_bps=impact_bps,
            allowed=allowed,
        )

    def cap_order_size(
        self,
        symbol: str,
        requested_dollars: float,
        adv_dollars: float,
        spread_bps: float = 10.0,
    ) -> float:
        """Return the maximum allowed order size in dollars (capped if needed)."""
        result = self.check(symbol, requested_dollars, adv_dollars, spread_bps)
        return min(requested_dollars, result.max_order_dollars)
