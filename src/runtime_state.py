"""
Atomic cross-process trading state via Redis.

Replaces the split-brain pattern of config.ORDERS_PAUSED / config.CIRCUIT_BREAKER_ACTIVE.

All processes that share the same Redis instance (Gunicorn workers, the main
bot loop, the Telegram bot) see the same trading state through a single Redis key.
State transitions are atomic via WATCH + MULTI/EXEC (optimistic concurrency
control). A valid fencing token is required for every transition.

Legal state transitions
-----------------------
    ACTIVE     → PAUSED | CLOSE_ONLY | HALTED
    PAUSED     → ACTIVE | CLOSE_ONLY | HALTED
    CLOSE_ONLY → ACTIVE | HALTED
    HALTED     → ACTIVE   (requires explicit operator action)

HALTED is not truly terminal — the operator can resume. However, nothing
can transition FROM halted automatically; only a verified human command can.

Usage
-----
    # At startup (main.py)
    redis_client = aioredis.from_url("redis://localhost:6379")
    state_store = RuntimeStateStore(redis_client)

    # Check before submitting an order
    if not await state_store.allows_new_orders():
        return  # do not proceed

    # Transition from the control-plane API
    token = FencingToken.from_string(request_body.fencing_token)
    success, reason = await state_store.transition(
        expected=TradingState.ACTIVE,
        target=TradingState.PAUSED,
        fencing_token=token,
    )

    # Internal safety transition (graceful shutdown, auto circuit-breaker)
    success, reason = await state_store.force_transition_internal(
        target=TradingState.HALTED,
        reason="drawdown_breach",
    )
"""

from __future__ import annotations

import logging
from enum import Enum

import redis.asyncio as aioredis

from .security.fencing import (
    create_internal_token,
    verify_token,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_STATE_KEY = "algotrader:trading_state"

# Legal state transitions: {from_state: [allowed_to_states]}
_LEGAL_TRANSITIONS: dict[str, list[str]] = {
    "active": ["paused", "close_only", "halted"],
    "paused": ["active", "close_only", "halted"],
    "close_only": ["active", "halted"],
    "halted": ["active"],
}


# ---------------------------------------------------------------------------
# TradingState enum
# ---------------------------------------------------------------------------


class TradingState(str, Enum):
    """
    Operational mode of the trading system.

    ACTIVE:     Normal operation. New orders allowed.
    PAUSED:     No new orders. Existing orders / stop-losses continue.
    CLOSE_ONLY: No new orders. Close / stop-loss orders allowed.
    HALTED:     All order activity suspended. Requires explicit resume.
    """

    ACTIVE = "active"
    PAUSED = "paused"
    CLOSE_ONLY = "close_only"
    HALTED = "halted"

    def allows_new_orders(self) -> bool:
        return self == TradingState.ACTIVE

    def allows_close_orders(self) -> bool:
        return self in (
            TradingState.ACTIVE,
            TradingState.PAUSED,
            TradingState.CLOSE_ONLY,
        )

    def requires_operator_resume(self) -> bool:
        """HALTED cannot be exited by automated logic."""
        return self == TradingState.HALTED


# ---------------------------------------------------------------------------
# RuntimeStateStore
# ---------------------------------------------------------------------------


class RuntimeStateStore:
    """
    Atomic cross-process trading state backed by Redis.

    All public methods are async-native (redis.asyncio).
    Thread-safety is provided by Redis's single-threaded command execution
    combined with WATCH/MULTI/EXEC optimistic locking.
    """

    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._redis = redis_client

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_state(self) -> TradingState:
        """
        Return the current trading state.

        If the key is absent (first run), initialises to ACTIVE and returns it.
        If the stored value is unrecognised (data corruption), returns HALTED
        to fail safe, and logs an error.
        """
        raw = await self._redis.get(TRADING_STATE_KEY)
        if raw is None:
            await self._redis.set(TRADING_STATE_KEY, TradingState.ACTIVE.value)
            logger.info("Trading state key absent — initialised to ACTIVE")
            return TradingState.ACTIVE
        try:
            return TradingState(raw.decode())
        except ValueError:
            logger.error(
                "Unrecognised trading state in Redis: %r — defaulting to HALTED (fail-safe)",
                raw,
            )
            return TradingState.HALTED

    async def allows_new_orders(self) -> bool:
        """Convenience wrapper used at order-intake hot path."""
        return (await self.get_state()).allows_new_orders()

    async def allows_close_orders(self) -> bool:
        """Convenience wrapper for close/stop-loss order checks."""
        return (await self.get_state()).allows_close_orders()

    # ------------------------------------------------------------------
    # Authenticated transition (for external commands via API / Telegram)
    # ------------------------------------------------------------------

    async def transition(
        self,
        expected: TradingState,
        target: TradingState,
        fencing_token: str,
    ) -> tuple[bool, str]:
        """
        Atomic CAS state transition guarded by an HMAC-SHA256 fencing token.

        Parameters
        ----------
        expected      : The state the caller believes is current.
        target        : The desired state after the transition.
        fencing_token : Token string from create_token() / create_internal_token().

        Returns (success: bool, reason: str).

        Failure cases:
          - Token is invalid or expired         → (False, "token_invalid:<why>")
          - Transition not in legal table        → (False, "illegal_transition:…")
          - Current state ≠ expected (CAS miss)  → (False, "cas_miss:current_is_…")
          - Concurrent modification (WATCH)      → (False, "watch_error:…")
        """
        # 1. Validate fencing token before touching Redis
        valid, reason = verify_token(fencing_token)
        if not valid:
            logger.warning(
                "Fencing token rejected (%s) for attempted transition %s → %s",
                reason,
                expected.value,
                target.value,
            )
            return False, f"token_invalid:{reason}"

        # 2. Validate the transition is in the legal table
        if target.value not in _LEGAL_TRANSITIONS.get(expected.value, []):
            logger.warning(
                "Illegal trading state transition attempted: %s → %s",
                expected.value,
                target.value,
            )
            return False, f"illegal_transition:{expected.value}_to_{target.value}"

        # 3. Atomic CAS via WATCH + MULTI/EXEC
        async with self._redis.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(TRADING_STATE_KEY)

                current_raw = await pipe.get(TRADING_STATE_KEY)
                current = (
                    TradingState(current_raw.decode())
                    if current_raw
                    else TradingState.ACTIVE
                )

                if current != expected:
                    await pipe.reset()
                    logger.info(
                        "CAS miss: expected=%s, actual=%s — another process already transitioned",
                        expected.value,
                        current.value,
                    )
                    return False, f"cas_miss:current_is_{current.value}"

                pipe.multi()
                pipe.set(TRADING_STATE_KEY, target.value)
                await pipe.execute()

                logger.warning(
                    "Trading state transition: %s → %s",
                    expected.value,
                    target.value,
                )
                return True, "ok"

            except aioredis.WatchError:
                logger.warning(
                    "WatchError during transition %s → %s — concurrent modification detected",
                    expected.value,
                    target.value,
                )
                return False, "watch_error:concurrent_modification"

    # ------------------------------------------------------------------
    # Internal / automated transitions (graceful shutdown, auto circuit-breaker)
    # ------------------------------------------------------------------

    async def force_transition_internal(
        self,
        target: TradingState,
        reason: str = "internal",
    ) -> tuple[bool, str]:
        """
        Force a transition to a safe state without a CAS check on expected.

        Intended ONLY for safety-critical automated actions:
          - Graceful shutdown (SIGTERM)
          - Drawdown breach auto-halt
          - Watchdog-detected frozen event loop

        The target must be a protective state (PAUSED, CLOSE_ONLY, or HALTED).
        ACTIVE is not a valid target here — resumption always requires a human
        via the authenticated transition() path.

        Generates an internal fencing token automatically.
        """
        if target not in (
            TradingState.PAUSED,
            TradingState.CLOSE_ONLY,
            TradingState.HALTED,
        ):
            return False, "internal_transition_target_must_be_safe_state"

        # Create (and implicitly validate) an internal token. This also
        # ensures the fencing secret has been initialised before we proceed.
        try:
            create_internal_token(action_code=target.value)
        except RuntimeError as e:
            return False, f"internal_token_error:{e}"

        current = await self.get_state()

        async with self._redis.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(TRADING_STATE_KEY)
                pipe.multi()
                pipe.set(TRADING_STATE_KEY, target.value)
                await pipe.execute()

                logger.warning(
                    "INTERNAL forced transition: %s → %s  (reason=%s)",
                    current.value,
                    target.value,
                    reason,
                )
                return True, "ok"

            except aioredis.WatchError:
                return False, "watch_error:concurrent_modification"
