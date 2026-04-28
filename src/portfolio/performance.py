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
from typing import Any

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
