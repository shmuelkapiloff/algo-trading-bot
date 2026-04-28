"""
Performance Tracker — daily P&L, cumulative return, Sharpe, max drawdown.

Reads from the `trade_events` table (append-only OMS ledger).
All calculations are read-only — no writes to DB.

Phase 1: computes daily metrics on-demand.
Phase 2: will pre-aggregate metrics into a `daily_performance` table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Optional, Set

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import select

from ..data.models import TradeEvent, OrderEventType

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


@dataclass
class DailyMetrics:
    as_of: date
    realized_pnl: float  # USD
    cumulative_return: float  # fraction (0.05 = 5%)
    sharpe_ratio: float | None  # annualised, None if < 30 days of history
    max_drawdown: float  # fraction (−0.10 = −10%)
    win_rate: float  # fraction (0.60 = 60%)
    total_trades: int


class PerformanceTracker:
    """
    Reads from OMS ledger to compute trading performance metrics.

    Parameters
    ----------
    session_factory : SQLAlchemy async_sessionmaker
    initial_equity  : Portfolio starting value (USD) for % calculations.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        initial_equity: float,
    ) -> None:
        self._factory = session_factory
        self._initial_equity = initial_equity

    async def get_daily_metrics(self, as_of: date | None = None) -> DailyMetrics:
        """Compute performance metrics as of today (or a specific date)."""
        if as_of is None:
            as_of = datetime.now(timezone.utc).date()

        fills = await self._load_fills(as_of)
        if not fills:
            return DailyMetrics(
                as_of=as_of,
                realized_pnl=0.0,
                cumulative_return=0.0,
                sharpe_ratio=None,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
            )

        pnl_df = self._compute_trade_pnl(fills)
        daily_pnl = self._daily_pnl_series(pnl_df, as_of)

        total_pnl = float(pnl_df["pnl"].sum())
        total_trades = len(pnl_df)
        win_rate = (
            float((pnl_df["pnl"] > 0).sum() / total_trades) if total_trades else 0.0
        )
        cum_return = (
            total_pnl / self._initial_equity if self._initial_equity > 0 else 0.0
        )

        sharpe = self._compute_sharpe(daily_pnl)
        mdd = self._compute_max_drawdown(daily_pnl)

        return DailyMetrics(
            as_of=as_of,
            realized_pnl=total_pnl,
            cumulative_return=cum_return,
            sharpe_ratio=sharpe,
            max_drawdown=mdd,
            win_rate=win_rate,
            total_trades=total_trades,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _load_fills(self, as_of: date) -> list[dict[str, Any]]:
        """Load all FILLED events up to (and including) as_of."""
        cutoff = datetime(
            as_of.year, as_of.month, as_of.day, 23, 59, 59, tzinfo=timezone.utc
        )
        stmt = (
            select(TradeEvent)
            .where(
                TradeEvent.event_type == OrderEventType.FILLED,
                TradeEvent.occurred_at <= cutoff,
            )
            .order_by(TradeEvent.occurred_at)
        )
        async with self._factory() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()

        return [
            {
                "order_id": row.order_id,
                "symbol": row.symbol,
                "strategy": row.strategy,
                "occurred_at": row.occurred_at,
                "payload": row.payload or {},
            }
            for row in rows
        ]

    def _compute_trade_pnl(self, fills: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Pair BUY fills with their corresponding SELL fills to compute P&L.
        Simplified pairing: FIFO per symbol.
        """
        records = []
        open_positions: dict[str, list[dict[str, Any]]] = {}

        for fill in fills:
            symbol = fill["symbol"]
            payload = fill["payload"]
            side = payload.get("side", "")
            qty = float(payload.get("filled_qty", 0))
            price = float(payload.get("fill_price", 0))

            if side == "buy":
                open_positions.setdefault(symbol, []).append(
                    {"qty": qty, "price": price, "fill": fill}
                )
            elif side == "sell" and open_positions.get(symbol):
                # FIFO matching
                remaining = qty
                while remaining > 0 and open_positions.get(symbol):
                    entry = open_positions[symbol][0]
                    matched = min(remaining, entry["qty"])
                    pnl = (price - entry["price"]) * matched
                    records.append(
                        {
                            "symbol": symbol,
                            "strategy": fill["strategy"],
                            "pnl": pnl,
                            "exit_date": (
                                fill["occurred_at"].date()
                                if hasattr(fill["occurred_at"], "date")
                                else fill["occurred_at"]
                            ),
                        }
                    )
                    entry["qty"] -= matched
                    remaining -= matched
                    if entry["qty"] <= 0:
                        open_positions[symbol].pop(0)

        return (
            pd.DataFrame(records)
            if records
            else pd.DataFrame(columns=["symbol", "strategy", "pnl", "exit_date"])
        )

    def _daily_pnl_series(self, pnl_df: pd.DataFrame, as_of: date) -> pd.Series:
        """Return a Series of daily P&L indexed by date."""
        if pnl_df.empty:
            return pd.Series(dtype=float)
        daily = (
            pnl_df.groupby("exit_date")["pnl"]
            .sum()
            .reindex(
                pd.bdate_range(
                    start=pnl_df["exit_date"].min(),
                    end=as_of,
                    freq="B",
                ).date,
                fill_value=0.0,
            )
        )
        return daily

    def _compute_sharpe(self, daily_pnl: pd.Series) -> float | None:
        """Annualised Sharpe ratio. Returns None if < 30 days of history."""
        if len(daily_pnl) < 30:
            return None
        daily_returns = daily_pnl / self._initial_equity
        mean_r = daily_returns.mean()
        std_r = daily_returns.std()
        if std_r == 0:
            return None
        return float((mean_r / std_r) * (_TRADING_DAYS_PER_YEAR**0.5))

    def _compute_max_drawdown(self, daily_pnl: pd.Series) -> float:
        """Peak-to-trough max drawdown as a negative fraction."""
        if daily_pnl.empty:
            return 0.0
        cumulative = daily_pnl.cumsum()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / self._initial_equity
        return float(drawdown.min())


# ===========================================================================
# Per-Strategy Drawdown Monitor
# ===========================================================================


@dataclass
class DrawdownAlert:
    """Fired when a strategy's drawdown from its high-water mark breaches -limit."""

    strategy_name: str
    current_pnl: float          # cumulative P&L of this strategy (dollars)
    hwm_pnl: float              # high-water mark P&L (dollars)
    drawdown_pct: float         # (current_pnl - hwm_pnl) / initial_equity, negative
    limit_pct: float            # threshold that was breached (positive, e.g. 0.08)


@dataclass
class StrategySnapshot:
    """Current drawdown status for a single strategy."""

    strategy_name: str
    cumulative_pnl: float
    hwm_pnl: float
    current_drawdown_pct: float   # negative; 0.0 = at high-water mark
    is_paused: bool
    limit_pct: float


class StrategyDrawdownTracker:
    """Tracks per-strategy high-water mark and fires a DrawdownAlert at -limit.

    Usage
    -----
        tracker = StrategyDrawdownTracker(drawdown_limit=0.08)

        # After each trade fill for a strategy:
        alert = tracker.update("momentum", current_cumulative_pnl=950.0)
        if alert:
            log.warning("Strategy paused: %s", alert)

        # Check before opening new positions:
        if tracker.is_paused("momentum"):
            return  # skip signal

        # Manual review resume (e.g. Monday morning):
        tracker.resume("momentum")

    Parameters
    ----------
    drawdown_limit:
        Fraction of initial_equity. Default 0.08 = 8%.
    initial_equity:
        Portfolio starting equity used to normalise drawdown %.
        Pass the same value as PerformanceTracker._initial_equity.
    """

    def __init__(
        self,
        drawdown_limit: float = 0.08,
        initial_equity: float = 10_000.0,
    ) -> None:
        self._limit = drawdown_limit
        self._initial_equity = initial_equity
        self._hwm: Dict[str, float] = {}      # strategy → peak cumulative P&L
        self._current: Dict[str, float] = {}  # strategy → latest cumulative P&L
        self._paused: Set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, strategy_name: str, current_pnl: float
    ) -> Optional[DrawdownAlert]:
        """Update P&L for a strategy; return DrawdownAlert if limit breached.

        Parameters
        ----------
        strategy_name:
            Identifier for the strategy (e.g. "momentum", "mean_reversion").
        current_pnl:
            Cumulative P&L in dollars for this strategy since inception.

        Returns
        -------
        DrawdownAlert if the strategy newly breached the drawdown limit,
        None otherwise.
        """
        prev_hwm = self._hwm.get(strategy_name, current_pnl)
        new_hwm = max(prev_hwm, current_pnl)
        self._hwm[strategy_name] = new_hwm
        self._current[strategy_name] = current_pnl

        drawdown_pct = (current_pnl - new_hwm) / max(self._initial_equity, 1.0)
        was_paused = strategy_name in self._paused

        if drawdown_pct <= -self._limit and not was_paused:
            self._paused.add(strategy_name)
            logger.warning(
                "[drawdown] Strategy '%s' paused: drawdown=%.1f%% breached limit=%.1f%%",
                strategy_name,
                drawdown_pct * 100,
                self._limit * 100,
            )
            return DrawdownAlert(
                strategy_name=strategy_name,
                current_pnl=current_pnl,
                hwm_pnl=new_hwm,
                drawdown_pct=drawdown_pct,
                limit_pct=self._limit,
            )
        return None

    def is_paused(self, strategy_name: str) -> bool:
        """Return True if this strategy is paused due to drawdown breach."""
        return strategy_name in self._paused

    def resume(self, strategy_name: str) -> None:
        """Re-enable a paused strategy (call after manual EOW review)."""
        self._paused.discard(strategy_name)
        logger.info("[drawdown] Strategy '%s' manually resumed.", strategy_name)

    def get_snapshot(self, strategy_name: str) -> StrategySnapshot:
        """Return current drawdown status for a strategy."""
        pnl = self._current.get(strategy_name, 0.0)
        hwm = self._hwm.get(strategy_name, 0.0)
        dd_pct = (pnl - hwm) / max(self._initial_equity, 1.0)
        return StrategySnapshot(
            strategy_name=strategy_name,
            cumulative_pnl=pnl,
            hwm_pnl=hwm,
            current_drawdown_pct=dd_pct,
            is_paused=strategy_name in self._paused,
            limit_pct=self._limit,
        )

    def get_all_snapshots(self) -> Dict[str, StrategySnapshot]:
        """Return snapshots for every tracked strategy."""
        all_names = set(self._hwm) | set(self._current)
        return {name: self.get_snapshot(name) for name in all_names}
