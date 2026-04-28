"""
Phase 1 Pre-Trade Gateway — simple stub adapters + factory.

The full gate implementations (PortfolioRiskGate, LiquidityGate,
ExecutionReadinessGate) require injected Protocol collaborators for live
data (spread, book depth, TCA metrics, sector exposure, etc.).

In Phase 1 we don't yet have:
  - Live spread / book-depth feeds (Alpaca free tier doesn't provide L2)
  - Historical TCA metrics (fill rate, slippage — no trade history)
  - Sector classification data

Strategy
--------
Stub adapters provide conservative-but-passing defaults so all gates are
wired in production code from day 1. When Phase 2 data sources come online,
replace each stub class with a real implementation — no changes to the
gateway, gate, or main.py needed.

Gates active in Phase 1
-----------------------
  1. LiquidityGate  — ADV check only (spread/depth use conservative stubs)
  2. PortfolioRiskGate — single-position cap (sector/correlation use stubs)
  3. ExecutionReadinessGate — always passes in Phase 1 (broker health stubs)

Gates deferred to Phase 2
--------------------------
  - SignalViabilityGate — needs win-rate history (90+ trades)
  - TailRiskGate        — needs ES/VaR model calibrated on real fills
"""

from __future__ import annotations

from ..risk.gates import (
    ExecutionReadinessGate,
    LiquidityGate,
    PortfolioRiskGate,
)
from ..risk.pre_trade_gateway import PreTradeGateway


# ---------------------------------------------------------------------------
# Stub: LiquidityGate.MarketDataProvider
# ---------------------------------------------------------------------------


class _Phase1MarketData:
    """
    Phase 1 stub. Spread and depth always pass (no L2 data on free tier).
    ADV is provided as a rough estimate — real filter already done by
    MarketDataFetcher.filter_by_adv() before signals are generated.
    """

    def get_spread_bps(self, symbol: str) -> float:
        # Conservative stub: assume 5 bps (well under 15 bps threshold)
        return 5.0

    def get_adv(self, symbol: str) -> float:
        # Conservative stub: 500k shares (will pass 2% ADV limit for small orders)
        return 500_000.0

    def get_book_depth_usd(self, symbol: str) -> float:
        # Conservative stub: well above $50k floor
        return 500_000.0


# ---------------------------------------------------------------------------
# Stub: PortfolioRiskGate.PortfolioStateProvider
# ---------------------------------------------------------------------------


class _Phase1PortfolioStateProvider:
    """
    Phase 1 stub.
    - Sector: all symbols map to "unknown" (no sector database)
    - Correlation: always 0.0 (no correlation model yet)
    - Position weight: derived from PortfolioManager live state
    """

    def __init__(self, portfolio_manager) -> None:
        self._pm = portfolio_manager

    def get_sector(self, symbol: str) -> str:
        return "unknown"

    def get_sector_exposure(self, sector: str) -> float:
        # No sector data → treat all positions as different sectors → always 0
        return 0.0

    def get_max_pairwise_correlation(self, symbol: str) -> float:
        # No correlation model yet → always 0
        return 0.0

    def get_position_weight(self, symbol: str) -> float:
        """Live position weight from PortfolioManager in-memory state."""
        equity = self._pm.equity
        if equity <= 0:
            return 0.0
        pos = self._pm.open_positions.get(symbol)
        if pos is None:
            return 0.0
        return (pos.qty * pos.avg_entry_price) / equity


# ---------------------------------------------------------------------------
# Stub: ExecutionReadinessGate.TCAMetrics
# ---------------------------------------------------------------------------


class _Phase1TCAMetrics:
    """
    Phase 1 stub — no TCA history yet. Always reports healthy values.
    Real TCA module will be wired in Phase 2 from the fills ledger.
    """

    def get_broker_latency_p95_ms(self) -> float:
        return 50.0  # well under 1500ms threshold

    def get_avg_slippage_bps(self) -> float:
        return 1.0  # well under 25 bps pause threshold

    def get_fill_rate_p95(self) -> float:
        return 0.99  # well above 60% floor


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_phase1_gateway(portfolio_manager, settings) -> PreTradeGateway:
    """
    Build a PreTradeGateway configured for Phase 1 paper trading.

    Parameters
    ----------
    portfolio_manager : PortfolioManager — provides live position weights
    settings          : Settings — provides risk config (max position pct, etc.)
    """
    return PreTradeGateway(
        gates=[
            LiquidityGate(
                market_data=_Phase1MarketData(),
                max_spread_bps=15.0,
                max_order_adv_pct=0.02,
                min_book_depth_usd=50_000.0,
            ),
            PortfolioRiskGate(
                portfolio_provider=_Phase1PortfolioStateProvider(portfolio_manager),
                max_sector_exposure=0.30,
                max_correlation=0.70,
                absolute_max_position_pct=settings.risk.absolute_max_position_pct,
            ),
            ExecutionReadinessGate(
                tca_metrics=_Phase1TCAMetrics(),
            ),
        ]
    )
