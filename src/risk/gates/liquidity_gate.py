"""
Gate 3 — Liquidity Constraints

Enforces three liquidity conditions before an order is submitted:

  1. Bid/ask spread ≤ max_spread_bps (default 15 bps).
     Wider spreads erode EV and indicate thin markets.

  2. Order size ≤ max_order_adv_pct × ADV (default 2%).
     Larger orders move the market and inflate slippage.
     Only checked when signal.qty is already set (post-sizing signals).

  3. Book depth ≥ min_book_depth_usd (default $50,000).
     Insufficient depth means the order cannot be filled without
     significant price impact.

From TRADING_BOT_PLAN.md §6ג:
    max_allowed_slippage_bps: 20
    max_order_adv_pct: 0.02
    Liquidity tiers in universe_exclusions: median spread <= 15 bps

Dependencies (constructor-injected):
    market_data_provider    Live spread, ADV, and book depth queries.
"""

from __future__ import annotations

import logging
from typing import Protocol

from .base_gate import GateResult, RiskGate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Injected collaborator protocol
# ---------------------------------------------------------------------------


class MarketDataProvider(Protocol):
    def get_spread_bps(self, symbol: str) -> float:
        """Current bid/ask spread in basis points."""
        ...

    def get_adv(self, symbol: str) -> float:
        """Average Daily Volume in shares (20-day rolling)."""
        ...

    def get_book_depth_usd(self, symbol: str) -> float:
        """Aggregated available depth at best bid+ask in USD."""
        ...


# ---------------------------------------------------------------------------
# Gate implementation
# ---------------------------------------------------------------------------


class LiquidityGate(RiskGate):
    """
    Blocks orders that would face illiquid or thin-market conditions.

    If signal.qty is None (signal not yet sized by the Portfolio layer),
    the ADV percentage check is skipped — only spread and depth are enforced.
    """

    def __init__(
        self,
        market_data: MarketDataProvider,
        max_spread_bps: float = 15.0,
        max_order_adv_pct: float = 0.02,
        min_book_depth_usd: float = 50_000.0,
    ) -> None:
        self._market_data = market_data
        self._max_spread_bps = max_spread_bps
        self._max_adv_pct = max_order_adv_pct
        self._min_depth = min_book_depth_usd

    async def evaluate(self, signal, portfolio_state: dict) -> GateResult:
        symbol = signal.symbol

        # ---- 1. Spread check ----
        spread_bps = self._market_data.get_spread_bps(symbol)
        if spread_bps > self._max_spread_bps:
            logger.debug(
                "[LiquidityGate] REJECTED %s  spread=%.1f bps  limit=%.1f bps",
                symbol,
                spread_bps,
                self._max_spread_bps,
            )
            return GateResult.reject(f"liquidity_spread_too_wide:{spread_bps:.1f}bps")

        # ---- 2. ADV check (only when qty is known) ----
        if signal.qty is not None:
            adv = self._market_data.get_adv(symbol)
            if adv > 0:
                order_adv_pct = signal.qty / adv
                if order_adv_pct > self._max_adv_pct:
                    logger.debug(
                        "[LiquidityGate] REJECTED %s  order_adv_pct=%.4f  limit=%.4f",
                        symbol,
                        order_adv_pct,
                        self._max_adv_pct,
                    )
                    return GateResult.reject(
                        f"liquidity_adv_exceeded:{order_adv_pct:.4f}>{self._max_adv_pct:.4f}"
                    )

        # ---- 3. Book depth check ----
        depth = self._market_data.get_book_depth_usd(symbol)
        if depth < self._min_depth:
            logger.debug(
                "[LiquidityGate] REJECTED %s  depth=$%.0f  min=$%.0f",
                symbol,
                depth,
                self._min_depth,
            )
            return GateResult.reject(
                f"liquidity_book_depth_insufficient:{depth:.0f}<{self._min_depth:.0f}"
            )

        return GateResult.approve()
