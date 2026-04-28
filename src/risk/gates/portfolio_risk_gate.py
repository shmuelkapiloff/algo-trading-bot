"""
Gate 2 — Portfolio Constraints

Enforces three portfolio-level limits before a new position is opened:

  1. Sector concentration  — no single sector may exceed max_sector_exposure.
  2. Pairwise correlation  — no new holding whose rolling Pearson correlation
                            with any current holding exceeds max_correlation.
  3. Single-position cap   — notional weight of new position stays below
                            absolute_max_position_pct of portfolio NAV.

From TRADING_BOT_PLAN.md §4:
    max_sector_exposure: 0.30   (dynamic per regime — pass via constructor)
    max_correlation:     0.70
    absolute_max_position_pct: 0.03

Dependencies (constructor-injected):
    portfolio_state_provider    Live query interface for positions/exposures.
"""

from __future__ import annotations

import logging
from typing import Protocol

from .base_gate import GateResult, RiskGate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Injected collaborator protocol
# ---------------------------------------------------------------------------


class PortfolioStateProvider(Protocol):
    def get_sector(self, symbol: str) -> str:
        """GICS sector for the symbol."""
        ...

    def get_sector_exposure(self, sector: str) -> float:
        """Current fraction of portfolio NAV in this sector (0.0–1.0)."""
        ...

    def get_max_pairwise_correlation(self, symbol: str) -> float:
        """
        Max rolling Pearson correlation between `symbol` and any current holding.
        Returns 0.0 if portfolio is empty (no holdings to correlate against).
        """
        ...

    def get_position_weight(self, symbol: str) -> float:
        """
        Current notional weight of `symbol` in the portfolio (0.0–1.0).
        Returns 0.0 if not currently held.
        """
        ...


# ---------------------------------------------------------------------------
# Gate implementation
# ---------------------------------------------------------------------------


class PortfolioRiskGate(RiskGate):
    """
    Enforces concentration, correlation, and sector limits.

    max_sector_exposure should be set dynamically by the Portfolio Manager
    based on the current market regime (see risk_management.max_sector_exposure_by_regime
    in TRADING_BOT_PLAN.md).
    """

    def __init__(
        self,
        portfolio_provider: PortfolioStateProvider,
        max_sector_exposure: float = 0.30,
        max_correlation: float = 0.70,
        absolute_max_position_pct: float = 0.03,
    ) -> None:
        self._provider = portfolio_provider
        self._max_sector = max_sector_exposure
        self._max_corr = max_correlation
        self._max_position = absolute_max_position_pct

    async def evaluate(self, signal, portfolio_state: dict) -> GateResult:
        symbol = signal.symbol

        # ---- 1. Sector concentration ----
        sector = self._provider.get_sector(symbol)
        sector_exposure = self._provider.get_sector_exposure(sector)
        if sector_exposure >= self._max_sector:
            logger.debug(
                "[PortfolioRiskGate] REJECTED %s  sector=%s  "
                "current_exposure=%.2f  limit=%.2f",
                symbol,
                sector,
                sector_exposure,
                self._max_sector,
            )
            return GateResult.reject(
                f"portfolio_sector_overweight:{sector}:{sector_exposure:.3f}"
            )

        # ---- 2. Pairwise correlation ----
        max_corr = self._provider.get_max_pairwise_correlation(symbol)
        if max_corr >= self._max_corr:
            logger.debug(
                "[PortfolioRiskGate] REJECTED %s  max_corr=%.2f  limit=%.2f",
                symbol,
                max_corr,
                self._max_corr,
            )
            return GateResult.reject(f"correlation_cluster_detected:{max_corr:.3f}")

        # ---- 3. Single-position concentration cap ----
        current_weight = self._provider.get_position_weight(symbol)
        if current_weight >= self._max_position:
            logger.debug(
                "[PortfolioRiskGate] REJECTED %s  position_weight=%.3f  limit=%.3f",
                symbol,
                current_weight,
                self._max_position,
            )
            return GateResult.reject(
                f"position_concentration_breach:{current_weight:.3f}"
            )

        return GateResult.approve()
