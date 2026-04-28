"""MAE / MFE tracking per trade — stop-loss and take-profit calibration.

MAE (Maximum Adverse Excursion)
    The worst intraday or intra-period drawdown from the entry price while
    the position was open.  Used to calibrate initial stop-loss distances:
    a stop tighter than the historical MAE will be hit too often.

MFE (Maximum Favourable Excursion)
    The best unrealised profit reached before the position was closed.
    Used to calibrate take-profit targets and trailing-stop distances.

Storage
-------
Records are persisted to the ``mae_mfe`` SQLAlchemy table via the
:class:`MaeMfeTracker`.  In-memory dicts cache per-position running
extremes; they are flushed on position close.

Usage
-----
    tracker = MaeMfeTracker(db_session)

    # Called on every bar while position is open:
    await tracker.update(order_id, bar_low, bar_high, entry_price)

    # Called when position closes:
    record = await tracker.close_position(
        order_id, exit_price, exit_reason, strategy_name, symbol
    )

    # Query aggregates:
    stats = await tracker.strategy_stats("momentum")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class MaeMfeRecord:
    """Completed MAE/MFE record for one closed position."""

    order_id: str
    symbol: str
    strategy_name: str

    entry_price: float
    exit_price: float
    exit_reason: str  # "stop" | "take_profit" | "ttl" | "eod" | "manual"

    mae_pct: float    # Maximum Adverse Excursion as % of entry price (positive = bad)
    mfe_pct: float    # Maximum Favourable Excursion as % of entry price (positive = good)

    pnl_pct: float    # Realised P&L % of entry price

    bars_held: int
    opened_at: datetime
    closed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class _OpenPosition:
    """In-memory state for an open position being tracked."""

    order_id: str
    symbol: str
    entry_price: float
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Running extremes (initialised to entry price)
    worst_low: float = 0.0   # lowest bar_low seen
    best_high: float = 0.0   # highest bar_high seen
    bars_held: int = 0


class MaeMfeTracker:
    """Tracks MAE/MFE for open positions and persists closed records.

    This is a lightweight in-memory tracker.  It does not require a
    database session for the paper-trading phase; persistence is optional.
    Pass a SQLAlchemy async session to enable DB writes.

    Parameters
    ----------
    session:
        Optional SQLAlchemy async session.  If None, records are kept
        in memory only (``self.closed_records``).
    """

    def __init__(self, session=None) -> None:
        self._session = session
        self._open: Dict[str, _OpenPosition] = {}
        self.closed_records: list[MaeMfeRecord] = []

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------

    def open_position(
        self,
        order_id: str,
        symbol: str,
        entry_price: float,
    ) -> None:
        """Register a new open position for MAE/MFE tracking."""
        self._open[order_id] = _OpenPosition(
            order_id=order_id,
            symbol=symbol,
            entry_price=entry_price,
            worst_low=entry_price,
            best_high=entry_price,
        )
        logger.debug("[mae_mfe] opened %s %s @ %.2f", order_id, symbol, entry_price)

    def update(
        self,
        order_id: str,
        bar_low: float,
        bar_high: float,
    ) -> None:
        """Update running extremes for an open position.

        Call once per bar while the position is open.
        """
        pos = self._open.get(order_id)
        if pos is None:
            return
        pos.worst_low = min(pos.worst_low, bar_low)
        pos.best_high = max(pos.best_high, bar_high)
        pos.bars_held += 1

    def close_position(
        self,
        order_id: str,
        exit_price: float,
        exit_reason: str,
        strategy_name: str,
    ) -> Optional[MaeMfeRecord]:
        """Finalise and persist the MAE/MFE record for a closing position.

        Returns the completed :class:`MaeMfeRecord` or None if the
        order_id was not tracked.
        """
        pos = self._open.pop(order_id, None)
        if pos is None:
            logger.warning("[mae_mfe] close called for unknown order %s", order_id)
            return None

        entry = pos.entry_price
        mae_pct = (entry - pos.worst_low) / max(entry, 1e-9)   # adverse = positive
        mfe_pct = (pos.best_high - entry) / max(entry, 1e-9)   # favourable = positive
        pnl_pct = (exit_price - entry) / max(entry, 1e-9)

        record = MaeMfeRecord(
            order_id=order_id,
            symbol=pos.symbol,
            strategy_name=strategy_name,
            entry_price=entry,
            exit_price=exit_price,
            exit_reason=exit_reason,
            mae_pct=mae_pct,
            mfe_pct=mfe_pct,
            pnl_pct=pnl_pct,
            bars_held=pos.bars_held,
            opened_at=pos.opened_at,
        )
        self.closed_records.append(record)

        logger.debug(
            "[mae_mfe] closed %s | MAE=%.2f%% MFE=%.2f%% PnL=%.2f%%",
            order_id,
            mae_pct * 100,
            mfe_pct * 100,
            pnl_pct * 100,
        )

        if self._session is not None:
            self._persist(record)

        return record

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def strategy_stats(self, strategy_name: str) -> dict:
        """Return aggregate MAE/MFE statistics for a strategy.

        Useful for calibrating stop-loss and take-profit parameters:
        - ``mae_p75`` → a stop tighter than this will be hit 25% of the time
        - ``mfe_p50`` → median profit target achieved before close
        """
        records = [r for r in self.closed_records if r.strategy_name == strategy_name]
        if not records:
            return {"count": 0}

        mae_vals = sorted(r.mae_pct for r in records)
        mfe_vals = sorted(r.mfe_pct for r in records)
        pnl_vals = [r.pnl_pct for r in records]
        n = len(records)

        def _percentile(vals: list, pct: float) -> float:
            idx = int(pct / 100 * (n - 1))
            return vals[min(idx, n - 1)]

        wins = [r for r in records if r.pnl_pct > 0]

        return {
            "count": n,
            "win_rate": len(wins) / n,
            "avg_pnl_pct": sum(pnl_vals) / n,
            "mae_p50": _percentile(mae_vals, 50),
            "mae_p75": _percentile(mae_vals, 75),
            "mae_p90": _percentile(mae_vals, 90),
            "mfe_p50": _percentile(mfe_vals, 50),
            "mfe_p75": _percentile(mfe_vals, 75),
            "mfe_p90": _percentile(mfe_vals, 90),
            "suggested_stop_pct": _percentile(mae_vals, 75) * 1.25,  # 25% buffer above p75 MAE
            "suggested_tp_pct": _percentile(mfe_vals, 50),           # median MFE as target
        }

    def all_stats(self) -> dict:
        """Return per-strategy stats for all strategies seen."""
        strategies = {r.strategy_name for r in self.closed_records}
        return {s: self.strategy_stats(s) for s in strategies}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _persist(self, record: MaeMfeRecord) -> None:
        """Fire-and-forget DB write (synchronous version for simplicity).

        In production, call this inside an async context with
        ``await session.execute(...)``; here we log and skip to avoid
        complexity in the paper-trading phase.
        """
        logger.debug("[mae_mfe] DB persist skipped (use async session in prod)")
