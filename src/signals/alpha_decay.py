"""
Alpha Decay Measurement + Signal TTL Calibration.

Measures how quickly alpha decays after signal generation by analysing
completed trades. Used to calibrate signal TTL (how long a signal is valid).

From TRADING_BOT_PLAN.md §11א.

Algorithm
---------
1. Fetch all completed trades for a strategy in the lookback window.
2. For each completed trade, compute the cumulative return at every
   hold-day horizon (1 to max_hold_days).
3. Average returns across trades for each horizon → decay curve.
4. Optimal TTL = last horizon where average return > 0.
5. Apply 80% safety buffer to avoid overfitting.

Usage
-----
    detector = AlphaDecayDetector(session_factory=factory)

    # Run weekly during paper trading:
    result = await detector.measure("momentum", lookback_days=63)
    print(f"Momentum optimal TTL: {result.optimal_ttl_days} days")
    print(f"Half-life: {result.half_life_days} days")

    # Get calibrated TTL seconds for strategy config:
    ttl_secs = await detector.get_ttl_seconds("momentum")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default TTL per strategy (before calibration — from plan §11א)
DEFAULT_TTL_DAYS: Dict[str, int] = {
    "momentum": 10,
    "mean_reversion": 5,
    "trend_following": 30,
}

_SAFETY_BUFFER = 0.80  # Use 80% of optimal TTL (plan §11א)
_MIN_TTL_DAYS = 1
_MAX_HOLD_DAYS = 20
_MIN_TRADES_FOR_CALIBRATION = 20


@dataclass
class AlphaDecayResult:
    """Result of an alpha decay measurement for one strategy."""

    strategy_name: str
    lookback_days: int
    n_trades: int
    decay_curve: Dict[int, float]  # hold_day → avg_return_pct
    optimal_ttl_days: int  # last day with positive avg_return
    half_life_days: Optional[int]  # day when avg_return = 50% of peak
    calibrated_ttl_days: int  # optimal_ttl × safety_buffer
    calibrated_ttl_seconds: int
    calibrated: bool  # False if not enough data → used default


class AlphaDecayDetector:
    """
    Measures alpha decay from completed trade history.

    Parameters
    ----------
    session_factory:
        SQLAlchemy async_sessionmaker. If None, operates in estimation-only
        mode using default TTL values (useful for testing/paper trading start).
    """

    def __init__(self, session_factory=None) -> None:
        self._factory = session_factory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def measure(
        self,
        strategy_name: str,
        lookback_days: int = 63,
    ) -> AlphaDecayResult:
        """
        Measure alpha decay for a strategy from trade history.

        Returns a default result with calibrated=False if fewer than
        MIN_TRADES_FOR_CALIBRATION trades are found.
        """
        trades = await self._load_completed_trades(strategy_name, lookback_days)

        if len(trades) < _MIN_TRADES_FOR_CALIBRATION:
            logger.info(
                "[alpha_decay] %s: only %d trades (need %d) — using default TTL",
                strategy_name,
                len(trades),
                _MIN_TRADES_FOR_CALIBRATION,
            )
            return self._default_result(strategy_name, lookback_days, len(trades))

        decay_curve = self._compute_decay_curve(trades)
        optimal_ttl = self._find_optimal_ttl(decay_curve)
        half_life = self._find_half_life(decay_curve)
        calibrated_ttl = max(_MIN_TTL_DAYS, int(optimal_ttl * _SAFETY_BUFFER))

        logger.info(
            "[alpha_decay] %s: optimal_ttl=%dd  half_life=%s  calibrated=%dd  "
            "trades=%d",
            strategy_name,
            optimal_ttl,
            f"{half_life}d" if half_life else "N/A",
            calibrated_ttl,
            len(trades),
        )

        return AlphaDecayResult(
            strategy_name=strategy_name,
            lookback_days=lookback_days,
            n_trades=len(trades),
            decay_curve=decay_curve,
            optimal_ttl_days=optimal_ttl,
            half_life_days=half_life,
            calibrated_ttl_days=calibrated_ttl,
            calibrated_ttl_seconds=calibrated_ttl * 86_400,
            calibrated=True,
        )

    async def get_ttl_seconds(
        self,
        strategy_name: str,
        lookback_days: int = 63,
    ) -> int:
        """
        Convenience method: return calibrated TTL in seconds.

        Falls back to DEFAULT_TTL_DAYS if not enough trade history.
        """
        result = await self.measure(strategy_name, lookback_days)
        return result.calibrated_ttl_seconds

    @staticmethod
    def get_default_ttl_seconds(strategy_name: str) -> int:
        """Return default (pre-calibration) TTL in seconds for a strategy."""
        days = DEFAULT_TTL_DAYS.get(strategy_name, 5)
        return days * 86_400

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _load_completed_trades(
        self, strategy_name: str, lookback_days: int
    ) -> List[dict]:
        """
        Load completed trades (BUY+SELL pairs) for a strategy from the OMS ledger.

        Returns list of dicts with: entry_date, exit_date, entry_price, exit_price,
        hold_days, return_pct.
        """
        if self._factory is None:
            return []

        try:
            from datetime import datetime, timedelta, timezone
            from sqlalchemy import select
            from ..data.models import TradeEvent, OrderEventType

            cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

            async with self._factory() as session:
                stmt = (
                    select(TradeEvent)
                    .where(
                        TradeEvent.strategy == strategy_name,
                        TradeEvent.event_type == OrderEventType.FILLED,
                        TradeEvent.occurred_at >= cutoff,
                    )
                    .order_by(TradeEvent.occurred_at)
                )
                result = await session.execute(stmt)
                events = result.scalars().all()

            return self._pair_fills(events)

        except Exception as exc:
            logger.warning("[alpha_decay] Failed to load trades: %s", exc)
            return []

    @staticmethod
    def _pair_fills(events) -> List[dict]:
        """Pair BUY and SELL fills to compute per-trade returns."""
        trades = []
        open_buys: Dict[str, dict] = {}

        for event in events:
            payload = event.payload or {}
            side = payload.get("side", "")
            symbol = event.symbol
            price = float(payload.get("fill_price", 0))
            dt = event.occurred_at

            if side == "buy":
                open_buys[symbol] = {"entry_date": dt, "entry_price": price}
            elif side == "sell" and symbol in open_buys:
                entry = open_buys.pop(symbol)
                hold_days = (dt - entry["entry_date"]).days
                if hold_days > 0 and entry["entry_price"] > 0:
                    ret = (price - entry["entry_price"]) / entry["entry_price"]
                    trades.append(
                        {
                            "entry_date": entry["entry_date"],
                            "exit_date": dt,
                            "entry_price": entry["entry_price"],
                            "exit_price": price,
                            "hold_days": hold_days,
                            "return_pct": ret,
                        }
                    )

        return trades

    @staticmethod
    def _compute_decay_curve(trades: List[dict]) -> Dict[int, float]:
        """
        Average return at each hold-day horizon across all trades.

        For a trade held for H days, we simulate returns at horizons 1..H
        using a linear interpolation (conservative: assumes linear decay).
        """
        horizon_returns: Dict[int, List[float]] = {}

        for trade in trades:
            hold_days = trade["hold_days"]
            total_return = trade["return_pct"]

            for day in range(1, min(hold_days, _MAX_HOLD_DAYS) + 1):
                # Linear interpolation of where return was at each day
                partial_return = total_return * (day / hold_days)
                horizon_returns.setdefault(day, []).append(partial_return)

        decay_curve: Dict[int, float] = {}
        for day in range(1, _MAX_HOLD_DAYS + 1):
            returns_at_day = horizon_returns.get(day, [])
            if len(returns_at_day) >= 5:  # need at least 5 trades at this horizon
                decay_curve[day] = float(np.mean(returns_at_day))

        return decay_curve

    @staticmethod
    def _find_optimal_ttl(decay_curve: Dict[int, float]) -> int:
        """Last hold day where average return is still positive."""
        positive_days = [d for d, r in decay_curve.items() if r > 0]
        if not positive_days:
            return _MIN_TTL_DAYS
        return max(positive_days)

    @staticmethod
    def _find_half_life(decay_curve: Dict[int, float]) -> Optional[int]:
        """Day at which average return drops to 50% of its peak value."""
        if not decay_curve:
            return None
        peak = max(decay_curve.values())
        if peak <= 0:
            return None
        half = peak * 0.5
        for day in sorted(decay_curve.keys()):
            if decay_curve[day] <= half:
                return day
        return None

    @staticmethod
    def _default_result(
        strategy_name: str,
        lookback_days: int,
        n_trades: int,
    ) -> AlphaDecayResult:
        """Return a default result using pre-defined TTL values."""
        default_days = DEFAULT_TTL_DAYS.get(strategy_name, 5)
        return AlphaDecayResult(
            strategy_name=strategy_name,
            lookback_days=lookback_days,
            n_trades=n_trades,
            decay_curve={},
            optimal_ttl_days=default_days,
            half_life_days=None,
            calibrated_ttl_days=default_days,
            calibrated_ttl_seconds=default_days * 86_400,
            calibrated=False,
        )
