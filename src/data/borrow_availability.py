"""Borrow availability checks for shortable symbols.

This module provides a broker-agnostic service to determine which symbols
cannot be shorted at the current time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class BorrowSnapshot:
    available_to_short: set[str]
    unavailable_to_short: set[str]


class BorrowAvailabilityService:
    """Resolve borrow availability from broker asset metadata.

    The broker object is expected to expose either:
    - async list_assets() -> iterable of objects/dicts
    - async get_assets()  -> iterable of objects/dicts

    Each asset entry should include:
    - symbol (str)
    - shortable (bool)
    """

    def __init__(self, broker) -> None:
        self._broker = broker

    async def get_unavailable_symbols(
        self,
        symbols: Iterable[str],
    ) -> set[str]:
        """Return symbols that are NOT available for shorting."""
        assets = await self._fetch_assets()
        by_symbol: dict[str, bool] = {}
        for a in assets:
            symbol = _asset_get(a, "symbol")
            shortable = bool(_asset_get(a, "shortable", False))
            if symbol:
                by_symbol[str(symbol).upper()] = shortable

        unavailable: set[str] = set()
        for s in symbols:
            sym = str(s).upper()
            if sym in by_symbol and not by_symbol[sym]:
                unavailable.add(sym)
        return unavailable

    async def get_snapshot(self, symbols: Iterable[str]) -> BorrowSnapshot:
        unavailable = await self.get_unavailable_symbols(symbols)
        wanted = {str(s).upper() for s in symbols}
        available = wanted - unavailable
        return BorrowSnapshot(
            available_to_short=available,
            unavailable_to_short=unavailable,
        )

    async def _fetch_assets(self):
        if hasattr(self._broker, "list_assets"):
            return await self._broker.list_assets()
        if hasattr(self._broker, "get_assets"):
            return await self._broker.get_assets()
        raise AttributeError("broker has neither list_assets() nor get_assets()")


def _asset_get(asset, key: str, default=None):
    if isinstance(asset, dict):
        return asset.get(key, default)
    return getattr(asset, key, default)
