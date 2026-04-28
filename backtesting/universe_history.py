"""Point-in-Time Universe — eliminates survivorship bias from backtests.

Problem
-------
Backtesting on the *current* S&P 500 member list inflates historical returns by
10–20% because it excludes companies that were delisted or removed from the
index (Fama, 1997; Elton et al., 1996).  The current-member list is a
survivorship-biased sample: the winners who made it to today.

Solution
--------
``PITUniverse`` stores a dated snapshot of index constituents, including
companies that were later removed.  When the backtest engine requests
"which symbols were tradeable on 2018-03-15?", it receives the historically
accurate set — not the current one.

Database table: ``sp500_constituents_pit``
------------------------------------------
    as_of_date DATE NOT NULL      -- date this record is valid FROM
    symbol     VARCHAR(10) NOT NULL
    added_date DATE NOT NULL       -- date symbol was added to index
    removed_date DATE              -- NULL = still in index today
    source     VARCHAR(50)         -- e.g. "wikipedia", "compustat", "manual"
    PRIMARY KEY (as_of_date, symbol)

Usage
-----
    pit = PITUniverse(session_factory)
    await pit.bulk_load_from_csv("data/sp500_pit.csv")

    # In backtest engine:
    symbols = await pit.get_constituents_at(date(2020, 3, 1))
    # → returns only symbols that were in the index on that date (incl. later-delisted)

Data sources (priority order)
------------------------------
1. Compustat GICS constituent history (paid)
2. WRDS SP500 constituent file (academic)
3. Wikipedia S&P 500 history page (free, limited history to ~2000)
4. Manual CSV built from press releases
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class ConstituentRecord:
    """A single symbol's membership period in the index."""

    symbol: str
    added_date: date
    removed_date: Optional[date] = None  # None = still a current member
    source: str = "unknown"

    def was_member_on(self, as_of: date) -> bool:
        """Return True if the symbol was in the index on `as_of`."""
        if as_of < self.added_date:
            return False
        if self.removed_date is not None and as_of >= self.removed_date:
            return False
        return True


# ---------------------------------------------------------------------------
# PITUniverse — in-memory implementation (DB backend is a P2 upgrade)
# ---------------------------------------------------------------------------


class PITUniverse:
    """Point-in-time universe for survivorship-bias-free backtesting.

    Phase 1 (current): in-memory store, populated from CSV on startup.
    Phase 2 (planned): async SQLAlchemy backend with ``sp500_constituents_pit``
                       table for faster range queries.

    Parameters
    ----------
    session_factory:
        SQLAlchemy async_sessionmaker (reserved for Phase 2 DB backend).
        Pass None for in-memory mode.
    """

    def __init__(self, session_factory=None) -> None:
        self._session_factory = session_factory
        # symbol → list of membership periods (sorted ascending by added_date)
        self._records: Dict[str, List[ConstituentRecord]] = {}

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def add_constituent(
        self,
        symbol: str,
        added_date: date,
        removed_date: Optional[date] = None,
        source: str = "unknown",
    ) -> None:
        """Register one symbol's index-membership period."""
        record = ConstituentRecord(
            symbol=symbol,
            added_date=added_date,
            removed_date=removed_date,
            source=source,
        )
        self._records.setdefault(symbol, []).append(record)

    def bulk_load_from_csv(self, csv_path: str) -> int:
        """Load constituent history from a CSV file.

        Expected columns (case-insensitive):
            symbol, added_date (YYYY-MM-DD), removed_date (YYYY-MM-DD or empty), source

        Returns
        -------
        Number of records loaded.
        """
        loaded = 0
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            # Normalise header to lowercase
            reader.fieldnames = (
                [f.strip().lower() for f in reader.fieldnames]
                if reader.fieldnames
                else reader.fieldnames
            )
            for row in reader:
                symbol = row["symbol"].strip().upper()
                added = date.fromisoformat(row["added_date"].strip())
                removed_raw = row.get("removed_date", "").strip()
                removed = date.fromisoformat(removed_raw) if removed_raw else None
                source = row.get("source", "csv").strip()
                self.add_constituent(symbol, added, removed, source)
                loaded += 1
        logger.info(
            "[PITUniverse] Loaded %d constituent records from %s", loaded, csv_path
        )
        return loaded

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get_constituents_at(self, as_of: date) -> List[str]:
        """Return all symbols that were in the index on `as_of`.

        This is the primary backtest query: use this instead of a static
        member list to eliminate survivorship bias.

        Parameters
        ----------
        as_of:
            Historical date to query membership for.

        Returns
        -------
        Sorted list of ticker symbols active on that date.
        """
        result = []
        for symbol, periods in self._records.items():
            for record in periods:
                if record.was_member_on(as_of):
                    result.append(symbol)
                    break  # a symbol can only appear once per date
        return sorted(result)

    def get_all_symbols(self) -> List[str]:
        """Return every symbol ever tracked (current + historical)."""
        return sorted(self._records.keys())

    def get_membership_periods(self, symbol: str) -> List[ConstituentRecord]:
        """Return all membership periods for a given symbol."""
        return list(self._records.get(symbol.upper(), []))

    def date_range(self) -> tuple[Optional[date], Optional[date]]:
        """Return (earliest_added, latest_removed_or_today) across all records."""
        all_added = [r.added_date for recs in self._records.values() for r in recs]
        all_removed = [
            r.removed_date
            for recs in self._records.values()
            for r in recs
            if r.removed_date is not None
        ]
        if not all_added:
            return None, None
        return min(all_added), max(all_removed) if all_removed else date.today()

    @property
    def n_records(self) -> int:
        """Total number of constituent records (not unique symbols)."""
        return sum(len(v) for v in self._records.values())

    @property
    def n_symbols(self) -> int:
        """Number of unique symbols ever tracked."""
        return len(self._records)
