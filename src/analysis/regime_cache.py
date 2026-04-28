"""
Redis-backed regime cache — 30-minute TTL.

The regime is expensive to compute (requires 200 bars of SPY).
We cache the result in Redis so all workers share a single computed value
and don't re-fetch SPY bars on every scan.

Keys
----
  algotrader:market_regime   → "bull" | "bear" | "sideways"
  algotrader:regime_updated  → ISO timestamp of last update

TTL: 1800 seconds (30 minutes). If the key is absent or stale, callers
should recompute and call set_regime().
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import redis.asyncio as aioredis

from .market_regime import MarketRegime

logger = logging.getLogger(__name__)

_REGIME_KEY = "algotrader:market_regime"
_UPDATED_KEY = "algotrader:regime_updated"
_TTL_SECONDS = 1800  # 30 minutes


async def get_cached_regime(redis: aioredis.Redis) -> MarketRegime | None:
    """
    Return the cached regime, or None if the cache is empty / stale.
    Callers should recompute and call set_regime() on a None response.
    """
    raw = await redis.get(_REGIME_KEY)
    if raw is None:
        return None
    try:
        return MarketRegime(raw.decode())
    except ValueError:
        logger.warning("Corrupt regime cache value %r — treating as cache miss", raw)
        return None


async def set_regime(redis: aioredis.Redis, regime: MarketRegime) -> None:
    """Store the regime in Redis with a 30-minute TTL."""
    await redis.set(_REGIME_KEY, regime.value, ex=_TTL_SECONDS)
    await redis.set(
        _UPDATED_KEY,
        datetime.now(timezone.utc).isoformat(),
        ex=_TTL_SECONDS,
    )
    logger.info("Regime cache updated: %s (TTL=%ds)", regime.value, _TTL_SECONDS)


async def get_regime_or_recompute(
    redis: aioredis.Redis,
    spy_df,  # pd.DataFrame — passed lazily to avoid re-fetching
) -> MarketRegime:
    """
    Return cached regime if fresh, otherwise recompute and cache.

    Parameters
    ----------
    spy_df : DataFrame of SPY daily bars (200+ rows, close_adj column).
             Only used when cache is cold — avoids unnecessary computation.
    """
    cached = await get_cached_regime(redis)
    if cached is not None:
        logger.debug("Regime served from cache: %s", cached.value)
        return cached

    from .market_regime import detect_regime

    regime = detect_regime(spy_df)
    await set_regime(redis, regime)
    return regime
