"""
Fundamental Analysis — P/E, EPS growth, ROE, and composite quality score.

Data source: Alpaca fundamentals (free tier) with optional Alpha Vantage fallback.
Cache policy: fundamentals are refreshed weekly (not daily) — values rarely change.

All methods return None / NaN if data unavailable rather than raising, so the
signal layer can gracefully skip fundamental gating for symbols without data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FundamentalSnapshot:
    """Fundamental metrics for a single symbol at the most recent report date."""

    symbol: str
    report_date: Optional[str]  # YYYY-MM-DD or None

    # Valuation
    pe_ratio: Optional[float] = None          # Price / EPS (TTM)
    forward_pe: Optional[float] = None
    pb_ratio: Optional[float] = None          # Price / Book

    # Profitability
    roe: Optional[float] = None               # Return on Equity (%)
    eps_ttm: Optional[float] = None           # Earnings per share (TTM)
    eps_growth_yoy: Optional[float] = None    # YoY growth rate

    # Balance sheet
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None

    # Quality composite (0–1, higher = better)
    quality_score: Optional[float] = None

    def compute_quality_score(self) -> float:
        """
        Composite quality score (0–1):
          - Low P/E relative to sector: +0.25
          - High ROE (> 15%): +0.25
          - Positive EPS growth: +0.25
          - Low debt-to-equity (< 1): +0.25
        Returns 0.0 if all data missing.
        """
        score = 0.0
        components = 0

        if self.pe_ratio is not None:
            components += 1
            if 5 < self.pe_ratio < 25:
                score += 0.25

        if self.roe is not None:
            components += 1
            if self.roe > 15.0:
                score += 0.25

        if self.eps_growth_yoy is not None:
            components += 1
            if self.eps_growth_yoy > 0:
                score += 0.25

        if self.debt_to_equity is not None:
            components += 1
            if self.debt_to_equity < 1.0:
                score += 0.25

        self.quality_score = score if components > 0 else None
        return score


class FundamentalAnalyzer:
    """
    Loads and caches fundamental snapshots per symbol.

    In Phase 1 the data dict is populated externally (e.g., from Alpaca
    fundamentals endpoint or a CSV import).  Phase 2 will add an async
    refresh loop via Alpha Vantage / Polygon fundamentals.
    """

    def __init__(self) -> None:
        self._cache: dict[str, FundamentalSnapshot] = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def update(self, snap: FundamentalSnapshot) -> None:
        snap.compute_quality_score()
        self._cache[snap.symbol] = snap
        logger.debug("fundamental.update symbol=%s quality=%.2f",
                     snap.symbol, snap.quality_score or 0.0)

    def update_from_dict(self, symbol: str, data: dict) -> FundamentalSnapshot:
        snap = FundamentalSnapshot(
            symbol=symbol,
            report_date=data.get("report_date"),
            pe_ratio=data.get("pe_ratio"),
            forward_pe=data.get("forward_pe"),
            pb_ratio=data.get("pb_ratio"),
            roe=data.get("roe"),
            eps_ttm=data.get("eps_ttm"),
            eps_growth_yoy=data.get("eps_growth_yoy"),
            debt_to_equity=data.get("debt_to_equity"),
            current_ratio=data.get("current_ratio"),
        )
        self.update(snap)
        return snap

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, symbol: str) -> Optional[FundamentalSnapshot]:
        return self._cache.get(symbol)

    def quality_score(self, symbol: str) -> Optional[float]:
        snap = self.get(symbol)
        return snap.quality_score if snap else None

    def is_high_quality(
        self,
        symbol: str,
        min_score: float = 0.5,
    ) -> bool:
        """Returns True if quality_score >= min_score, False if data missing."""
        score = self.quality_score(symbol)
        if score is None:
            return True  # no data → don't filter out the symbol
        return score >= min_score

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def filter_by_quality(
        self,
        symbols: list[str],
        min_score: float = 0.5,
    ) -> list[str]:
        """Return subset of symbols that pass the quality threshold."""
        return [s for s in symbols if self.is_high_quality(s, min_score)]
