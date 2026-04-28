"""Alpaca API rate limiter — token bucket for 200 requests/minute.

Alpaca's REST API enforces a 200 requests/minute limit per account.
This module tracks the remaining budget and provides a simple
``acquire()`` interface that blocks (async) when the budget is
exhausted, preventing HTTP 429 errors.

Implementation: Token Bucket algorithm
  - Bucket capacity: 200 tokens
  - Refill rate:     200 tokens / 60 seconds (full refill every minute)
  - Each API call consumes one token
  - ``acquire(n)`` waits until n tokens are available

Usage (inject into AlpacaBroker)
----------------------------------
    limiter = AlpacaRateLimiter()

    async def submit_order(order):
        await limiter.acquire()           # blocks if budget exhausted
        return await _alpaca_api.post(order)

Thread safety
-------------
The limiter is implemented with asyncio locks and is safe for
concurrent coroutines in a single asyncio event loop. It is NOT safe
to share across multiple OS processes (use Redis-based rate limiting
for multi-process setups).
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class AlpacaRateLimiter:
    """Async token bucket rate limiter for Alpaca REST API.

    Parameters
    ----------
    max_requests:
        Maximum requests allowed per ``window_seconds`` (default 200).
    window_seconds:
        Rolling window duration in seconds (default 60).
    min_wait_seconds:
        Minimum sleep between acquire() calls when bucket is empty
        (default 0.1 seconds to avoid busy-wait).
    """

    def __init__(
        self,
        max_requests: int = 200,
        window_seconds: float = 60.0,
        min_wait_seconds: float = 0.1,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.min_wait_seconds = min_wait_seconds

        # Token bucket state
        self._tokens: float = float(max_requests)
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

        # Stats
        self._total_acquired: int = 0
        self._total_waited_ms: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def acquire(self, n: int = 1) -> None:
        """Acquire n tokens, waiting if the bucket is empty.

        This is the primary throttling point. Call before every
        Alpaca REST API request.

        Parameters
        ----------
        n: Number of tokens to consume (default 1 per API call).
        """
        if n <= 0:
            return

        wait_start = time.monotonic()
        async with self._lock:
            self._refill()

            while self._tokens < n:
                # Calculate how long to wait for n tokens to become available
                deficit = n - self._tokens
                refill_rate = self.max_requests / self.window_seconds  # tokens/sec
                wait_time = max(deficit / refill_rate, self.min_wait_seconds)

                logger.debug(
                    "[rate_limiter] Budget exhausted (%.1f tokens). Waiting %.2fs",
                    self._tokens,
                    wait_time,
                )
                # Release lock while sleeping to allow other coroutines to proceed
                self._lock.release()
                await asyncio.sleep(wait_time)
                await self._lock.acquire()
                self._refill()

            self._tokens -= n
            self._total_acquired += n

        elapsed_ms = (time.monotonic() - wait_start) * 1000.0
        self._total_waited_ms += elapsed_ms

        if elapsed_ms > 100:
            logger.warning(
                "[rate_limiter] acquire(%d) waited %.0f ms (budget was exhausted)",
                n,
                elapsed_ms,
            )

    def available(self) -> float:
        """Return current token count (approximate — no lock taken)."""
        self._refill()
        return max(self._tokens, 0.0)

    def stats(self) -> dict:
        """Return usage statistics."""
        return {
            "available_tokens": self.available(),
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "total_acquired": self._total_acquired,
            "total_waited_ms": round(self._total_waited_ms, 1),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        """Refill tokens based on elapsed time (no lock — caller holds it)."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        refill_rate = self.max_requests / self.window_seconds  # tokens/sec
        self._tokens = min(
            self._max_tokens(),
            self._tokens + elapsed * refill_rate,
        )
        self._last_refill = now

    def _max_tokens(self) -> float:
        return float(self.max_requests)


# ---------------------------------------------------------------------------
# Global singleton (shared by all AlpacaBroker instances in one process)
# ---------------------------------------------------------------------------

_default_limiter: AlpacaRateLimiter | None = None


def get_default_limiter() -> AlpacaRateLimiter:
    """Return the process-wide default rate limiter (singleton)."""
    global _default_limiter
    if _default_limiter is None:
        _default_limiter = AlpacaRateLimiter()
    return _default_limiter
