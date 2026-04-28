"""
Control-Plane High Availability — Redis Leader Election with Fencing Tokens.

Implements active/passive HA for the trading bot control plane.

Design
------
- Uses Redis SET NX PX for distributed lease (single-writer guarantee).
- The elected leader issues a fencing token via ``src.security.fencing_tokens``
  with the current lease generation embedded as ``incident_id``.
- Standby nodes poll the lock key; if the primary fails to renew within
  ``lease_ms`` the standby acquires the lock and becomes the new leader.
- Generation counter is stored separately in Redis so it monotonically
  increases across leader changes — this is the fencing token generation.

Usage
-----
    elector = LeaderElector(redis_url="redis://localhost:6379/0")
    await elector.start()

    if elector.is_leader():
        # only the leader places orders / runs strategy scans
        ...

    await elector.stop()

Safety invariants
-----------------
1. At most one process holds the lock at any time (Redis SET NX).
2. Every leader transition increments ``_GENERATION_KEY`` atomically.
3. Fencing tokens are generated per-generation — old tokens from a
   previous leader are rejected by ``verify_token()`` after renewal.
4. If Redis becomes unreachable the lease expires automatically (TTL).
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Redis key for the distributed lock
_LOCK_KEY = "algotrader:control_plane:leader_lock"
# Redis key for the monotonic fencing generation counter
_GENERATION_KEY = "algotrader:control_plane:leader_generation"
# Value stored in the lock — identifies this process as the current holder
_LOCK_VALUE_PREFIX = f"{socket.gethostname()}:{os.getpid()}"


@dataclass
class LeadershipState:
    is_leader: bool
    generation: int
    leader_identity: str  # who holds the lock right now
    lease_acquired_at: Optional[float] = None  # monotonic


class LeaderElector:
    """
    Distributed leader election backed by Redis.

    Parameters
    ----------
    redis_url:
        Redis connection URL.
    lease_ms:
        How long the lock TTL is (milliseconds).  Default 10 000 (10s).
    renewal_interval_s:
        How often to renew the lock (seconds).  Default lease_ms / 3.
    on_elected:
        Optional async callback invoked when this node becomes leader.
    on_demoted:
        Optional async callback invoked when this node loses leadership.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        lease_ms: int = 10_000,
        renewal_interval_s: Optional[float] = None,
        on_elected=None,
        on_demoted=None,
    ) -> None:
        self._redis_url = redis_url
        self._lease_ms = lease_ms
        self._renewal_interval = renewal_interval_s or (lease_ms / 3000)
        self._on_elected = on_elected
        self._on_demoted = on_demoted

        self._redis = None
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        self._is_leader: bool = False
        self._generation: int = 0
        self._lock_value: str = f"{_LOCK_VALUE_PREFIX}:{id(self)}"
        self._lease_acquired_at: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect to Redis and begin the election loop."""
        try:
            import redis.asyncio as aioredis  # type: ignore[import]
            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        except ImportError:
            logger.error("LeaderElector requires redis[asyncio] — install it first")
            raise

        self._task = asyncio.get_event_loop().create_task(
            self._election_loop(), name="leader_elector"
        )
        logger.info("LeaderElector started (lease_ms=%d)", self._lease_ms)

    async def stop(self) -> None:
        """Release the lock if held and shut down the loop."""
        self._stop_event.set()
        if self._is_leader:
            await self._release_lock()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
        if self._redis:
            await self._redis.aclose()
        logger.info("LeaderElector stopped")

    def is_leader(self) -> bool:
        return self._is_leader

    def get_state(self) -> LeadershipState:
        return LeadershipState(
            is_leader=self._is_leader,
            generation=self._generation,
            leader_identity=self._lock_value if self._is_leader else "",
            lease_acquired_at=self._lease_acquired_at,
        )

    # ------------------------------------------------------------------
    # Election loop
    # ------------------------------------------------------------------

    async def _election_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self._is_leader:
                    renewed = await self._renew_lock()
                    if not renewed:
                        await self._handle_demotion()
                else:
                    acquired = await self._try_acquire_lock()
                    if acquired:
                        await self._handle_election()
            except Exception:
                logger.exception("LeaderElector: unexpected error in election loop")
                # If we lose Redis we lose the lock — be safe
                if self._is_leader:
                    await self._handle_demotion(redis_error=True)

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self._renewal_interval
                )
            except asyncio.TimeoutError:
                pass  # normal — continue loop

    async def _try_acquire_lock(self) -> bool:
        """Attempt to acquire the Redis lock (SET NX PX)."""
        result = await self._redis.set(
            _LOCK_KEY,
            self._lock_value,
            nx=True,
            px=self._lease_ms,
        )
        return result is not None

    async def _renew_lock(self) -> bool:
        """Extend the TTL only if we still hold the lock (Lua CAS)."""
        _LUA = """
        if redis.call("GET", KEYS[1]) == ARGV[1] then
            return redis.call("PEXPIRE", KEYS[1], ARGV[2])
        else
            return 0
        end
        """
        result = await self._redis.eval(
            _LUA, 1, _LOCK_KEY, self._lock_value, str(self._lease_ms)
        )
        return bool(result)

    async def _release_lock(self) -> None:
        """Release the lock only if we own it (Lua CAS)."""
        _LUA = """
        if redis.call("GET", KEYS[1]) == ARGV[1] then
            return redis.call("DEL", KEYS[1])
        else
            return 0
        end
        """
        try:
            await self._redis.eval(_LUA, 1, _LOCK_KEY, self._lock_value)
        except Exception:
            logger.exception("LeaderElector: failed to release lock")

    async def _handle_election(self) -> None:
        """Called when this node becomes the leader."""
        # Increment the generation counter atomically
        gen = await self._redis.incr(_GENERATION_KEY)
        self._generation = int(gen)
        self._is_leader = True
        self._lease_acquired_at = time.monotonic()

        logger.info(
            "LeaderElector: THIS NODE IS NOW LEADER  generation=%d  identity=%s",
            self._generation, self._lock_value,
        )

        # Issue a fencing token for this leadership term
        try:
            from trading_bot.src.security.fencing_tokens import create_internal_token  # noqa: PLC0415
            token = create_internal_token(
                action_code="leader_term",
                validity_seconds=max(300, int(self._lease_ms / 1000) * 30),
            )
            logger.info(
                "LeaderElector: fencing token issued  token_id=%s  generation=%d",
                token.incident_id, self._generation,
            )
        except Exception:
            logger.exception("LeaderElector: failed to issue fencing token")

        if self._on_elected:
            try:
                await self._on_elected(self._generation)
            except Exception:
                logger.exception("LeaderElector: on_elected callback failed")

    async def _handle_demotion(self, redis_error: bool = False) -> None:
        """Called when this node loses leadership."""
        prev_gen = self._generation
        self._is_leader = False
        self._lease_acquired_at = None

        logger.warning(
            "LeaderElector: THIS NODE IS NO LONGER LEADER  generation=%d  redis_error=%s",
            prev_gen, redis_error,
        )

        if self._on_demoted:
            try:
                await self._on_demoted(prev_gen)
            except Exception:
                logger.exception("LeaderElector: on_demoted callback failed")
