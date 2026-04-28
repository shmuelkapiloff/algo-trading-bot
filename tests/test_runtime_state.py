"""
Tests for RuntimeStateStore (src/runtime_state.py).

Covers:
  - get_state() on empty Redis → returns ACTIVE and initialises key
  - get_state() with corrupt data → returns HALTED (fail-safe)
  - transition() happy path with valid fencing token
  - transition() CAS miss when current state ≠ expected
  - transition() rejection for illegal state transition
  - transition() rejection for expired / invalid fencing token
  - force_transition_internal() succeeds for safe target states
  - force_transition_internal() fails for ACTIVE target
  - allows_new_orders() reflects state correctly
  - allows_close_orders() reflects state correctly

All Redis interactions are mocked via fakeredis.aioredis (no real Redis needed).
Run: pytest tests/test_runtime_state.py -v
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap HMAC fencing secret before importing runtime_state.
#
# trading_bot.src.runtime_state uses relative imports, so when loaded via
# the trading_bot package its fencing module is trading_bot.src.security.fencing.
# We must init_secret on THAT module instance.
# ---------------------------------------------------------------------------
import trading_bot.src.security.fencing as _fencing_mod
from trading_bot.src.security.fencing import create_token, init_secret

_TEST_SECRET = b"test-secret-for-unit-tests-only-32b"
init_secret(_TEST_SECRET)

from trading_bot.src.runtime_state import (  # noqa: E402
    TradingState,
    RuntimeStateStore,
    TRADING_STATE_KEY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def valid_token(action_code: str = "paused") -> str:
    """Return a valid HMAC token string."""
    return create_token(action_code=action_code, validity_seconds=300)


def expired_token(action_code: str = "paused") -> str:
    """Return a well-formed but expired HMAC token string (valid signature, past expiry)."""
    import base64, json, uuid

    payload = {
        "incident_id": str(uuid.uuid4()),
        "action_code": action_code,
        "severity": "internal",
        "issued_by": "test",
        "issued_at": time.time() - 10,
        "valid_until": time.time() - 1,  # already expired
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload_b64 = base64.urlsafe_b64encode(canonical.encode()).decode()
    # Sign using the same _sign() helper the module uses
    mac = _fencing_mod._sign(payload_b64.encode())
    return f"{payload_b64}.{mac}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def redis_client():
    """In-memory fake Redis (fakeredis). Install: pip install fakeredis[aioredis]"""
    try:
        import fakeredis.aioredis as fakeredis
    except ImportError:
        pytest.skip("fakeredis not installed — run: pip install fakeredis[aioredis]")
    client = fakeredis.FakeRedis()
    yield client
    await client.aclose()


@pytest.fixture
def store(redis_client):
    return RuntimeStateStore(redis_client)


# ---------------------------------------------------------------------------
# get_state tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_state_initialises_to_active_when_key_absent(store, redis_client):
    state = await store.get_state()
    assert state == TradingState.ACTIVE
    # Key should now be set
    raw = await redis_client.get(TRADING_STATE_KEY)
    assert raw.decode() == "active"


@pytest.mark.asyncio
async def test_get_state_returns_stored_value(store, redis_client):
    await redis_client.set(TRADING_STATE_KEY, "paused")
    state = await store.get_state()
    assert state == TradingState.PAUSED


@pytest.mark.asyncio
async def test_get_state_returns_halted_on_corrupt_data(store, redis_client):
    await redis_client.set(TRADING_STATE_KEY, "CORRUPTED_GARBAGE")
    state = await store.get_state()
    assert state == TradingState.HALTED  # fail-safe


# ---------------------------------------------------------------------------
# transition() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transition_active_to_paused_succeeds(store, redis_client):
    await redis_client.set(TRADING_STATE_KEY, "active")
    token = valid_token("paused")

    success, reason = await store.transition(
        expected=TradingState.ACTIVE,
        target=TradingState.PAUSED,
        fencing_token=token,
    )

    assert success is True
    assert reason == "ok"
    raw = await redis_client.get(TRADING_STATE_KEY)
    assert raw.decode() == "paused"


@pytest.mark.asyncio
async def test_transition_rejected_when_current_state_differs(store, redis_client):
    # State is PAUSED but we expect ACTIVE → CAS miss
    await redis_client.set(TRADING_STATE_KEY, "paused")
    token = valid_token("close_only")

    success, reason = await store.transition(
        expected=TradingState.ACTIVE,
        target=TradingState.CLOSE_ONLY,
        fencing_token=token,
    )

    assert success is False
    assert "cas_miss" in reason


@pytest.mark.asyncio
async def test_transition_rejected_for_illegal_transition(store, redis_client):
    # HALTED → PAUSED is not in the legal table
    await redis_client.set(TRADING_STATE_KEY, "halted")
    token = valid_token("paused")

    success, reason = await store.transition(
        expected=TradingState.HALTED,
        target=TradingState.PAUSED,
        fencing_token=token,
    )

    assert success is False
    assert "illegal_transition" in reason


@pytest.mark.asyncio
async def test_transition_rejected_for_expired_token(store, redis_client):
    await redis_client.set(TRADING_STATE_KEY, "active")
    token = expired_token("paused")

    success, reason = await store.transition(
        expected=TradingState.ACTIVE,
        target=TradingState.PAUSED,
        fencing_token=token,
    )

    assert success is False
    assert "token_invalid" in reason
    assert "expired" in reason


# ---------------------------------------------------------------------------
# force_transition_internal() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_force_transition_to_halted_succeeds(store, redis_client):
    await redis_client.set(TRADING_STATE_KEY, "active")

    success, reason = await store.force_transition_internal(
        target=TradingState.HALTED,
        reason="drawdown_breach",
    )

    assert success is True
    raw = await redis_client.get(TRADING_STATE_KEY)
    assert raw.decode() == "halted"


@pytest.mark.asyncio
async def test_force_transition_to_active_is_rejected(store):
    # ACTIVE is not a valid internal target — resumption requires human action
    success, reason = await store.force_transition_internal(
        target=TradingState.ACTIVE,
    )

    assert success is False
    assert "safe_state" in reason


# ---------------------------------------------------------------------------
# Convenience method tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_allows_new_orders_true_when_active(store, redis_client):
    await redis_client.set(TRADING_STATE_KEY, "active")
    assert await store.allows_new_orders() is True


@pytest.mark.asyncio
async def test_allows_new_orders_false_when_paused(store, redis_client):
    await redis_client.set(TRADING_STATE_KEY, "paused")
    assert await store.allows_new_orders() is False


@pytest.mark.asyncio
async def test_allows_close_orders_true_when_close_only(store, redis_client):
    await redis_client.set(TRADING_STATE_KEY, "close_only")
    assert await store.allows_close_orders() is True


@pytest.mark.asyncio
async def test_allows_close_orders_false_when_halted(store, redis_client):
    await redis_client.set(TRADING_STATE_KEY, "halted")
    assert await store.allows_close_orders() is False
