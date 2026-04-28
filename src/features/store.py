"""
Feature Store — versioned, point-in-time snapshots.

Prevents look-ahead bias: each snapshot is keyed by (symbol, as_of_date).
Features are computed once and cached; any recalculation uses only data
available *before* as_of_date.

Storage backend: in-process dict for development; can be swapped to Redis
or a DB table for production by replacing FeatureStore._backend.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class FeatureSnapshot:
    """Immutable feature vector for a single (symbol, as_of_date)."""

    symbol: str
    as_of_date: date
    features: dict[str, Any]
    content_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        payload = json.dumps(self.features, sort_keys=True, default=str)
        self.content_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]


class FeatureStore:
    """
    Versioned point-in-time feature cache.

    Keys: (symbol, as_of_date) → FeatureSnapshot
    On cache miss, returns None; caller should use FeatureBuilder to compute.
    """

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], FeatureSnapshot] = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def put(self, snapshot: FeatureSnapshot) -> None:
        key = (snapshot.symbol, str(snapshot.as_of_date))
        self._data[key] = snapshot
        logger.debug("feature_store.put symbol=%s date=%s hash=%s",
                     snapshot.symbol, snapshot.as_of_date, snapshot.content_hash)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, symbol: str, as_of_date: date) -> Optional[FeatureSnapshot]:
        return self._data.get((symbol, str(as_of_date)))

    def get_or_raise(self, symbol: str, as_of_date: date) -> FeatureSnapshot:
        snap = self.get(symbol, as_of_date)
        if snap is None:
            raise KeyError(f"No feature snapshot for {symbol} @ {as_of_date}")
        return snap

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def evict_before(self, cutoff_date: date) -> int:
        """Remove snapshots older than cutoff_date. Returns count evicted."""
        to_del = [k for k in self._data if k[1] < str(cutoff_date)]
        for k in to_del:
            del self._data[k]
        return len(to_del)

    def __len__(self) -> int:
        return len(self._data)
