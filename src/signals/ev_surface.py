"""
EV Surface — 2D Expected Value by Volatility Regime × Liquidity Tier.

Tracks realised returns in a 12-cell grid:
  vol_regime   : ("low_vol", "normal", "high_vol")
  liquidity_tier: ("high", "mid", "low", "illiquid")

Each cell accumulates observed trade returns and exposes:
  - get_ev(vol_regime, liquidity_tier) → float  (mean return)
  - async update(vol_regime, liquidity_tier, realized_return)
  - get_surface() → full snapshot for dashboard

From TRADING_BOT_PLAN.md §11ב.

Regime / Tier Classification
----------------------------
Vol regime is determined by VIX level (or rolling portfolio vol):
  low_vol   : VIX < 15  (or annualised vol < 12%)
  normal    : VIX 15-25 (or vol 12-25%)
  high_vol  : VIX > 25  (or vol > 25%)

Liquidity tier is determined by spread and average daily volume:
  high      : spread ≤ 5 bps  and ADV ≥ 1M shares
  mid       : spread 5-15 bps or ADV 100K-1M
  low       : spread 15-50 bps or ADV 10K-100K
  illiquid  : spread > 50 bps or ADV < 10K

Usage
-----
    surface = EVSurface()

    # After a trade is closed:
    await surface.update("normal", "high", realized_return=0.012)

    # Before entering a trade:
    ev = surface.get_ev("high_vol", "mid")
    if ev < 0:
        logger.warning("EV negative in high_vol/mid — skip trade")
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

VolRegime = str  # "low_vol" | "normal" | "high_vol"
LiquidityTier = str  # "high" | "mid" | "low" | "illiquid"
Cell = Tuple[VolRegime, LiquidityTier]

# Vol regime thresholds
_VIX_LOW = 15.0
_VIX_HIGH = 25.0
_VOL_LOW = 0.12  # annualised
_VOL_HIGH = 0.25

# Liquidity tier thresholds
_SPREAD_HIGH_BPS = 5.0
_SPREAD_MID_BPS = 15.0
_SPREAD_LOW_BPS = 50.0
_ADV_HIGH = 1_000_000
_ADV_MID = 100_000
_ADV_LOW = 10_000

# Minimum observations before using cell EV (else fallback = 0.0)
_MIN_OBS_FOR_EV = 10


@dataclass
class CellStats:
    """Running statistics for one EV surface cell."""

    n: int = 0
    sum_return: float = 0.0
    sum_sq_return: float = 0.0

    def update(self, r: float) -> None:
        self.n += 1
        self.sum_return += r
        self.sum_sq_return += r * r

    @property
    def mean(self) -> float:
        return self.sum_return / self.n if self.n > 0 else 0.0

    @property
    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        return (self.sum_sq_return / self.n) - (self.mean**2)

    @property
    def std(self) -> float:
        v = self.variance
        return math.sqrt(v) if v > 0 else 0.0

    @property
    def sharpe(self) -> Optional[float]:
        """Sharpe proxy: mean / std (no risk-free adjustment)."""
        if self.n < _MIN_OBS_FOR_EV or self.std == 0:
            return None
        return self.mean / self.std

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "ev": round(self.mean, 6),
            "std": round(self.std, 6),
            "sharpe": round(self.sharpe, 3) if self.sharpe is not None else None,
        }


class EVSurface:
    """
    2D EV surface: volatility_regime × liquidity_tier.

    Thread-safe via asyncio.Lock for concurrent fill handler access.
    Persistent across sessions if you call load_state()/save_state().

    Parameters
    ----------
    initial_ev_override:
        Optional dict of (vol_regime, liquidity_tier) → EV to pre-populate
        the surface (e.g., from backtests or previous calibration).
    """

    VOL_REGIMES = ("low_vol", "normal", "high_vol")
    LIQUIDITY_TIERS = ("high", "mid", "low", "illiquid")

    def __init__(
        self,
        initial_ev_override: Optional[Dict[Cell, float]] = None,
    ) -> None:
        self._lock = asyncio.Lock()
        self._cells: Dict[Cell, CellStats] = {
            (v, l): CellStats() for v in self.VOL_REGIMES for l in self.LIQUIDITY_TIERS
        }

        # Pre-populate from backtest if provided
        if initial_ev_override:
            for (vol, liq), ev in initial_ev_override.items():
                if (vol, liq) in self._cells:
                    # Inject as synthetic observations (n=10 with mean=ev)
                    for _ in range(10):
                        self._cells[(vol, liq)].update(ev)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_ev(
        self,
        vol_regime: VolRegime,
        liquidity_tier: LiquidityTier,
    ) -> float:
        """
        Return expected value for the given regime/tier cell.

        Returns 0.0 if cell has fewer than MIN_OBS_FOR_EV observations.
        """
        cell = self._cells.get((vol_regime, liquidity_tier))
        if cell is None:
            logger.debug(
                "[ev_surface] Unknown cell: (%s, %s)", vol_regime, liquidity_tier
            )
            return 0.0
        if cell.n < _MIN_OBS_FOR_EV:
            return 0.0
        return cell.mean

    async def update(
        self,
        vol_regime: VolRegime,
        liquidity_tier: LiquidityTier,
        realized_return: float,
    ) -> None:
        """
        Record a realised trade return in the appropriate cell.
        Thread-safe (asyncio.Lock).
        """
        async with self._lock:
            key = (vol_regime, liquidity_tier)
            if key not in self._cells:
                logger.warning(
                    "[ev_surface] Unknown cell (%s, %s) — skipping update",
                    vol_regime,
                    liquidity_tier,
                )
                return
            self._cells[key].update(realized_return)

    def get_surface(self) -> dict:
        """Return the full surface snapshot for dashboard display."""
        return {
            f"{vol}/{liq}": stats.to_dict() for (vol, liq), stats in self._cells.items()
        }

    def get_cell_stats(
        self,
        vol_regime: VolRegime,
        liquidity_tier: LiquidityTier,
    ) -> Optional[CellStats]:
        """Return raw CellStats for a specific cell."""
        return self._cells.get((vol_regime, liquidity_tier))

    def best_cell(self) -> Optional[Tuple[Cell, float]]:
        """Return the (regime, tier) cell with highest EV (min 10 obs)."""
        candidates = [
            ((v, l), stats.mean)
            for (v, l), stats in self._cells.items()
            if stats.n >= _MIN_OBS_FOR_EV
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[1])

    def save_state(self) -> dict:
        """Serialise surface to a plain dict for persistence (JSON/Redis)."""
        return {
            f"{v}|{l}": {
                "n": s.n,
                "sum_return": s.sum_return,
                "sum_sq_return": s.sum_sq_return,
            }
            for (v, l), s in self._cells.items()
        }

    def load_state(self, state: dict) -> None:
        """
        Restore surface from a previously saved state dict.
        Call before the trading session starts.
        """
        for key, data in state.items():
            if "|" not in key:
                continue
            vol, liq = key.split("|", 1)
            cell_key = (vol, liq)
            if cell_key in self._cells:
                cell = self._cells[cell_key]
                cell.n = data.get("n", 0)
                cell.sum_return = data.get("sum_return", 0.0)
                cell.sum_sq_return = data.get("sum_sq_return", 0.0)

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def classify_vol_regime(
        vix: Optional[float] = None,
        annualised_vol: Optional[float] = None,
    ) -> VolRegime:
        """
        Classify current volatility regime from VIX or annualised vol.
        VIX takes precedence if provided.
        """
        if vix is not None:
            if vix < _VIX_LOW:
                return "low_vol"
            if vix <= _VIX_HIGH:
                return "normal"
            return "high_vol"

        if annualised_vol is not None:
            if annualised_vol < _VOL_LOW:
                return "low_vol"
            if annualised_vol <= _VOL_HIGH:
                return "normal"
            return "high_vol"

        return "normal"  # safe default

    @staticmethod
    def classify_liquidity_tier(
        spread_bps: Optional[float] = None,
        avg_daily_volume: Optional[float] = None,
    ) -> LiquidityTier:
        """
        Classify symbol liquidity tier from spread and/or ADV.
        More conservative of the two dimensions wins.
        """
        tier_from_spread: Optional[LiquidityTier] = None
        if spread_bps is not None:
            if spread_bps <= _SPREAD_HIGH_BPS:
                tier_from_spread = "high"
            elif spread_bps <= _SPREAD_MID_BPS:
                tier_from_spread = "mid"
            elif spread_bps <= _SPREAD_LOW_BPS:
                tier_from_spread = "low"
            else:
                tier_from_spread = "illiquid"

        tier_from_adv: Optional[LiquidityTier] = None
        if avg_daily_volume is not None:
            if avg_daily_volume >= _ADV_HIGH:
                tier_from_adv = "high"
            elif avg_daily_volume >= _ADV_MID:
                tier_from_adv = "mid"
            elif avg_daily_volume >= _ADV_LOW:
                tier_from_adv = "low"
            else:
                tier_from_adv = "illiquid"

        _rank = {"high": 0, "mid": 1, "low": 2, "illiquid": 3}

        if tier_from_spread and tier_from_adv:
            # Take the more conservative (worse) tier
            if _rank[tier_from_adv] >= _rank[tier_from_spread]:
                return tier_from_adv
            return tier_from_spread
        if tier_from_spread:
            return tier_from_spread
        if tier_from_adv:
            return tier_from_adv

        return "mid"  # safe default
