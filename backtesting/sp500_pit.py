"""
S&P 500 Point-in-Time Snapshot Table Management.

Solves look-ahead bias in backtesting:
  Polygon /v3/reference/tickers?index=SPX returns the CURRENT constituent list.
  For backtest we need to know which symbols were in the S&P 500 on a given date.

Solution: maintain a local table of (as_of_date, symbol, added_date, removed_date).
  - removed_date = None means the symbol is still a member.
  - Source can be 'wiki_snapshot' (Wikipedia historical S&P 500) or 'manual'.

Usage::

    pit = Sp500PitManager()
    pit.load_from_records([...])  # seed from CSV / manual data
    members = pit.get_members(as_of_date=date(2010, 1, 15))
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Sp500Record:
    symbol: str
    added_date: Optional[date]
    removed_date: Optional[date]   # None = still a member
    source: str = "manual"


class Sp500PitManager:
    """
    Point-in-time S&P 500 constituent manager.

    Stores add/remove events per symbol. For any as_of_date returns
    the set of symbols that were constituents on that date.
    """

    def __init__(self) -> None:
        self._records: list[Sp500Record] = []

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_from_records(self, records: list[Sp500Record]) -> None:
        self._records.extend(records)
        logger.info("sp500_pit.loaded records=%d", len(records))

    def load_from_csv(self, path: str | Path) -> int:
        """
        Load from CSV with columns: symbol, added_date, removed_date, source.
        Dates in YYYY-MM-DD format; empty removed_date = still a member.
        Returns number of records loaded.
        """
        records: list[Sp500Record] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                added = _parse_date(row.get("added_date"))
                removed = _parse_date(row.get("removed_date"))
                records.append(Sp500Record(
                    symbol=row["symbol"].strip().upper(),
                    added_date=added,
                    removed_date=removed,
                    source=row.get("source", "csv"),
                ))
        self.load_from_records(records)
        return len(records)

    def load_from_csv_string(self, csv_data: str) -> int:
        """Load from an in-memory CSV string (useful for tests)."""
        records: list[Sp500Record] = []
        reader = csv.DictReader(io.StringIO(csv_data))
        for row in reader:
            added = _parse_date(row.get("added_date"))
            removed = _parse_date(row.get("removed_date"))
            records.append(Sp500Record(
                symbol=row["symbol"].strip().upper(),
                added_date=added,
                removed_date=removed,
                source=row.get("source", "csv"),
            ))
        self.load_from_records(records)
        return len(records)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_members(self, as_of_date: date) -> set[str]:
        """
        Returns the set of symbols that were S&P 500 constituents on as_of_date.
        A symbol is a member if: added_date <= as_of_date AND
                                  (removed_date is None OR removed_date > as_of_date)
        """
        members: set[str] = set()
        for rec in self._records:
            added_ok = rec.added_date is None or rec.added_date <= as_of_date
            removed_ok = rec.removed_date is None or rec.removed_date > as_of_date
            if added_ok and removed_ok:
                members.add(rec.symbol)
        return members

    def is_member(self, symbol: str, as_of_date: date) -> bool:
        return symbol.upper() in self.get_members(as_of_date)

    def add_record(
        self,
        symbol: str,
        added_date: Optional[date],
        removed_date: Optional[date] = None,
        source: str = "manual",
    ) -> None:
        self._records.append(Sp500Record(
            symbol=symbol.upper(),
            added_date=added_date,
            removed_date=removed_date,
            source=source,
        ))

    def __len__(self) -> int:
        return len(self._records)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_date(value: str | None) -> Optional[date]:
    if not value or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None
