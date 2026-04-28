"""
Tests for src/monitoring/tca.py — TcaMonitor

Covers:
  - record_fill() persists to Redis and computes slippage correctly
  - slippage is sign-adjusted (BUY: positive when overpaying, SELL: positive when underselling)
  - get_metrics() computes rolling averages correctly
  - get_metrics() on empty Redis returns safe defaults (fill_rate=1.0)
  - Auto-throttle activates when avg_slippage > 12 bps (with ≥ 5 fills)
  - Auto-throttle capped at MAX_THROTTLE_STEPS (3)
  - throttle_multiplier: 0.5^steps
  - Auto-pause activates when avg_slippage > 25 bps
  - Auto-pause activates when fill_rate < 0.60
  - No circuit breaker triggered below 5 fills (insufficient sample)
  - reset_throttle() clears throttle steps
  - clear_history() removes fills and resets throttle

Run: pytest tests/test_tca.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import trading_bot.src.security.fencing as _fencing_mod
from trading_bot.src.security.fencing import init_secret

# Initialise HMAC secret so force_transition_internal can create internal tokens
_TEST_SECRET = b"test-secret-for-unit-tests-only-32b"
init_secret(_TEST_SECRET)

import fakeredis.aioredis

from trading_bot.src.monitoring.tca import (
    TcaMonitor,
    TcaMetrics,
    TCA_WINDOW,
    PAUSE_SLIPPAGE_BPS,
    PAUSE_FILL_RATE,
    THROTTLE_SLIPPAGE_BPS,
    MAX_THROTTLE_STEPS,
    THROTTLE_FACTOR,
    REDIS_KEY_FILLS,
    REDIS_KEY_THROTTLE,
)
from trading_bot.src.runtime_state import RuntimeStateStore, TradingState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
async def redis():
    """In-memory fakeredis instance, fresh for each test."""
    return fakeredis.aioredis.FakeRedis(decode_responses=False)


@pytest.fixture()
async def state_store(redis):
    return RuntimeStateStore(redis)


@pytest.fixture()
async def tca(redis, state_store):
    return TcaMonitor(redis_client=redis, state_store=state_store)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _fill(
    monitor: TcaMonitor,
    *,
    symbol: str = "AAPL",
    fill_price: float = 100.0,
    vwap: float = 100.0,
    filled_qty: int = 10,
    requested_qty: int = 10,
    latency_ms: float = 100.0,
    side: str = "buy",
    order_id: str = "o1",
) -> None:
    await monitor.record_fill(
        order_id=order_id,
        symbol=symbol,
        fill_price=fill_price,
        vwap_benchmark=vwap,
        filled_qty=filled_qty,
        requested_qty=requested_qty,
        fill_latency_ms=latency_ms,
        side=side,
    )


async def _fill_n(monitor: TcaMonitor, n: int, slippage_bps: float = 0.0) -> None:
    """Record n fills with the given slippage (BUY side)."""
    for i in range(n):
        # fill_price = vwap * (1 + slippage_bps/10000)
        vwap = 100.0
        fill_price = vwap * (1 + slippage_bps / 10_000)
        await _fill(
            monitor,
            fill_price=fill_price,
            vwap=vwap,
            order_id=f"o{i}",
        )


# ---------------------------------------------------------------------------
# Slippage calculation
# ---------------------------------------------------------------------------


class TestSlippageCalculation:
    async def test_zero_slippage_buy(self, tca):
        """Buy at exact VWAP → 0 slippage."""
        await _fill(tca, fill_price=100.0, vwap=100.0, side="buy")
        metrics = await tca.get_metrics()
        assert metrics.avg_slippage_bps == pytest.approx(0.0)

    async def test_positive_slippage_buy_overpay(self, tca):
        """Buy above VWAP → positive slippage (bad for buyer)."""
        # 10 bps above VWAP
        await _fill(tca, fill_price=100.10, vwap=100.0, side="buy")
        metrics = await tca.get_metrics()
        assert metrics.avg_slippage_bps == pytest.approx(10.0, abs=0.01)

    async def test_negative_slippage_buy_underpay(self, tca):
        """Buy below VWAP → negative slippage (good for buyer)."""
        await _fill(tca, fill_price=99.90, vwap=100.0, side="buy")
        metrics = await tca.get_metrics()
        assert metrics.avg_slippage_bps == pytest.approx(-10.0, abs=0.01)

    async def test_positive_slippage_sell_undersell(self, tca):
        """Sell below VWAP → positive slippage (bad for seller)."""
        # fill_price=99.90 < vwap=100.0; side_sign=-1 → slippage = -10 * -1 = +10 bps
        await _fill(tca, fill_price=99.90, vwap=100.0, side="sell")
        metrics = await tca.get_metrics()
        assert metrics.avg_slippage_bps == pytest.approx(10.0, abs=0.01)

    async def test_invalid_vwap_skipped(self, tca):
        """Zero/negative vwap_benchmark is ignored."""
        await tca.record_fill(
            order_id="bad",
            symbol="AAPL",
            fill_price=100.0,
            vwap_benchmark=0.0,
            filled_qty=10,
            requested_qty=10,
            fill_latency_ms=100.0,
        )
        metrics = await tca.get_metrics()
        assert metrics.sample_size == 0  # not recorded


# ---------------------------------------------------------------------------
# Fill rate
# ---------------------------------------------------------------------------


class TestFillRate:
    async def test_full_fill_rate(self, tca):
        await _fill(tca, filled_qty=10, requested_qty=10)
        metrics = await tca.get_metrics()
        assert metrics.fill_rate_pct == pytest.approx(1.0)

    async def test_partial_fill_rate(self, tca):
        await _fill(tca, filled_qty=7, requested_qty=10)
        metrics = await tca.get_metrics()
        assert metrics.fill_rate_pct == pytest.approx(0.7)

    async def test_average_fill_rate_across_fills(self, tca):
        await _fill(tca, order_id="o1", filled_qty=10, requested_qty=10)
        await _fill(tca, order_id="o2", filled_qty=6, requested_qty=10)
        metrics = await tca.get_metrics()
        assert metrics.fill_rate_pct == pytest.approx(0.8)  # (1.0 + 0.6) / 2


# ---------------------------------------------------------------------------
# Empty metrics (safe defaults)
# ---------------------------------------------------------------------------


class TestEmptyMetrics:
    async def test_empty_redis_returns_safe_defaults(self, tca):
        metrics = await tca.get_metrics()
        assert metrics.sample_size == 0
        assert metrics.fill_rate_pct == pytest.approx(1.0)
        assert metrics.avg_slippage_bps == pytest.approx(0.0)
        assert metrics.throttle_steps == 0

    async def test_throttle_multiplier_default_is_one(self, tca):
        metrics = await tca.get_metrics()
        assert metrics.throttle_multiplier == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Throttle mechanism
# ---------------------------------------------------------------------------


class TestThrottle:
    async def test_no_throttle_below_threshold(self, tca, redis):
        """5 fills well below THROTTLE_SLIPPAGE_BPS do NOT trigger throttle."""
        await _fill_n(tca, n=5, slippage_bps=THROTTLE_SLIPPAGE_BPS - 2.0)
        steps = int(await redis.get(REDIS_KEY_THROTTLE) or 0)
        assert steps == 0

    async def test_throttle_activates_above_threshold(self, tca, redis):
        """5 fills above threshold → throttle step 1."""
        await _fill_n(tca, n=5, slippage_bps=THROTTLE_SLIPPAGE_BPS + 1.0)
        steps = int(await redis.get(REDIS_KEY_THROTTLE) or 0)
        assert steps == 1

    async def test_no_throttle_below_5_fills(self, tca, redis):
        """Fewer than 5 fills → no throttle regardless of slippage."""
        await _fill_n(tca, n=4, slippage_bps=50.0)  # very high slippage
        steps = int(await redis.get(REDIS_KEY_THROTTLE) or 0)
        assert steps == 0

    async def test_throttle_capped_at_max_steps(self, tca, redis):
        """Throttle cannot exceed MAX_THROTTLE_STEPS even with many bad fills."""
        # Prime to max
        await redis.set(REDIS_KEY_THROTTLE, MAX_THROTTLE_STEPS)
        await _fill_n(tca, n=5, slippage_bps=50.0)  # would normally increment
        steps = int(await redis.get(REDIS_KEY_THROTTLE) or 0)
        assert steps == MAX_THROTTLE_STEPS

    async def test_throttle_multiplier_step1(self, tca, redis):
        await redis.set(REDIS_KEY_THROTTLE, 1)
        metrics = await tca.get_metrics()
        assert metrics.throttle_multiplier == pytest.approx(THROTTLE_FACTOR**1)

    async def test_throttle_multiplier_step3(self, tca, redis):
        await redis.set(REDIS_KEY_THROTTLE, 3)
        metrics = await tca.get_metrics()
        assert metrics.throttle_multiplier == pytest.approx(THROTTLE_FACTOR**3)

    async def test_reset_throttle(self, tca, redis):
        await redis.set(REDIS_KEY_THROTTLE, 2)
        await tca.reset_throttle()
        steps = int(await redis.get(REDIS_KEY_THROTTLE) or 0)
        assert steps == 0

    async def test_get_throttle_multiplier_convenience(self, tca, redis):
        await redis.set(REDIS_KEY_THROTTLE, 2)
        mult = await tca.get_throttle_multiplier()
        assert mult == pytest.approx(THROTTLE_FACTOR**2)


# ---------------------------------------------------------------------------
# Circuit breaker (auto-pause)
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    async def test_no_pause_below_slippage_threshold(self, tca, state_store):
        """5 fills at 24 bps (< 25 bps threshold) → no pause."""
        await _fill_n(tca, n=5, slippage_bps=PAUSE_SLIPPAGE_BPS - 1.0)
        state = await state_store.get_state()
        assert state == TradingState.ACTIVE

    async def test_circuit_breaker_slippage(self, tca, state_store):
        """5 fills above PAUSE_SLIPPAGE_BPS → system paused."""
        await _fill_n(tca, n=5, slippage_bps=PAUSE_SLIPPAGE_BPS + 1.0)
        state = await state_store.get_state()
        assert state == TradingState.PAUSED

    async def test_circuit_breaker_fill_rate(self, tca, state_store):
        """5 fills with fill_rate below PAUSE_FILL_RATE → system paused."""
        for i in range(5):
            await _fill(
                tca,
                order_id=f"o{i}",
                filled_qty=5,  # 50% fill rate
                requested_qty=10,
            )
        state = await state_store.get_state()
        assert state == TradingState.PAUSED

    async def test_no_circuit_breaker_below_5_fills(self, tca, state_store):
        """4 fills (even terrible ones) must NOT trigger circuit breaker."""
        await _fill_n(tca, n=4, slippage_bps=100.0)
        state = await state_store.get_state()
        assert state == TradingState.ACTIVE

    async def test_circuit_breaker_broker_latency(self, tca, state_store):
        """Broker latency > 2000 ms with ≥ 5 fills → system paused."""
        # Need ≥ 5 fills first so sample_size passes
        await _fill_n(tca, n=5, slippage_bps=0.0)
        # Force latency check
        from trading_bot.src.monitoring.tca import PAUSE_LATENCY_MS

        await tca.record_broker_latency(PAUSE_LATENCY_MS + 100.0)
        state = await state_store.get_state()
        assert state == TradingState.PAUSED


# ---------------------------------------------------------------------------
# Clear history
# ---------------------------------------------------------------------------


class TestClearHistory:
    async def test_clear_history_removes_fills_and_throttle(self, tca, redis):
        await _fill_n(tca, n=3, slippage_bps=5.0)
        await redis.set(REDIS_KEY_THROTTLE, 2)

        await tca.clear_history()

        metrics = await tca.get_metrics()
        assert metrics.sample_size == 0
        assert metrics.throttle_steps == 0


# ---------------------------------------------------------------------------
# Rolling window
# ---------------------------------------------------------------------------


class TestRollingWindow:
    async def test_window_trims_to_tca_window(self, tca):
        """Recording more than TCA_WINDOW fills keeps only the most recent."""
        for i in range(TCA_WINDOW + 10):
            await _fill(tca, order_id=f"o{i}")
        metrics = await tca.get_metrics()
        assert metrics.sample_size == TCA_WINDOW
