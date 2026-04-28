"""S&P 500 constituents provider (official-source ready adapter).

Design:
- Production can use a licensed official provider by implementing the same
  interface as `Sp500Provider.fetch_constituents`.
- Default implementation supports a CSV endpoint (for example, licensed
  vendor export or curated mirror) and validates payload schema.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO

import httpx


@dataclass(frozen=True)
class Sp500Constituent:
    symbol: str
    name: str = ""
    sector: str = ""


class Sp500Provider:
    """Fetch constituents from a configured CSV endpoint."""

    def __init__(self, csv_url: str, timeout_s: float = 15.0) -> None:
        self._url = csv_url
        self._timeout = timeout_s

    async def fetch_constituents(self) -> list[Sp500Constituent]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(self._url)
            resp.raise_for_status()
            return _parse_constituents_csv(resp.text)


def _parse_constituents_csv(raw: str) -> list[Sp500Constituent]:
    """Parse CSV with expected columns: symbol,ticker,name,sector."""
    rows = csv.DictReader(StringIO(raw))
    result: list[Sp500Constituent] = []

    for row in rows:
        symbol = (
            (row.get("symbol") or row.get("Symbol") or row.get("ticker") or row.get("Ticker") or "")
            .strip()
            .upper()
        )
        if not symbol:
            continue
        name = (row.get("name") or row.get("Name") or "").strip()
        sector = (row.get("sector") or row.get("Sector") or "").strip()
        result.append(Sp500Constituent(symbol=symbol, name=name, sector=sector))

    if len(result) < 400:
        raise ValueError(
            f"Constituent list too small ({len(result)}). Check provider/source integrity."
        )
    return result
