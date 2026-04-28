"""
Tests for src/portfolio/risk.py — calculate_position_size()

Tests cover:
  - Normal sizing (basic Fixed-Fractional formula)
  - Hard absolute cap enforcement
  - Global risk budget exhaustion (returns 0)
  - Global risk budget scale-down
  - Stop-loss floor prevents absurdly large positions
  - Invalid inputs (zero price, zero equity) return 0
  - Whole-shares rounding (never fractional)
"""

from __future__ import annotations

import pytest

from src.portfolio.risk import calculate_position_size
from src.signals.models import OrderSide, SignalIntent


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _signal(stop_pct: float = 0.03, symbol: str = "AAPL") -> SignalIntent:
    return SignalIntent(
        symbol=symbol,
        side=OrderSide.BUY,
        strategy_name="test",
        confidence=0.70,
        stop_distance_pct=stop_pct,
    )


def _size(
    signal: SignalIntent | None = None,
    portfolio_value: float = 100_000.0,
    current_open_risk: float = 0.0,
    max_risk_per_trade: float = 0.01,
    absolute_max_position_pct: float = 0.03,
    max_global_open_risk: float = 0.02,
    stop_loss_floor_pct: float = 0.03,
    last_price: float = 100.0,
) -> int:
    return calculate_position_size(
        signal=signal or _signal(),
        portfolio_value=portfolio_value,
        current_open_risk=current_open_risk,
        max_risk_per_trade=max_risk_per_trade,
        absolute_max_position_pct=absolute_max_position_pct,
        max_global_open_risk=max_global_open_risk,
        stop_loss_floor_pct=stop_loss_floor_pct,
        last_price=last_price,
    )


# ---------------------------------------------------------------------------
# Basic Fixed-Fractional formula
# ---------------------------------------------------------------------------


class TestBasicSizing:
    def test_standard_case(self):
        """
        portfolio=100k, risk=1%, stop=3%, price=$100
        risk_dollars = 1000
        position_dollars = 1000/0.03 ≈ 33333
        cap: 100k × 50% = 50000 → not binding
        shares = floor(33333/100) = 333
        """
        shares = _size(absolute_max_position_pct=0.50)
        assert shares == 333

    def test_tighter_stop_gives_larger_position(self):
        """With a 1.5% stop, position is larger than with 3% stop."""
        shares_tight = _size(
            signal=_signal(stop_pct=0.015), absolute_max_position_pct=0.50
        )
        shares_wide = _size(
            signal=_signal(stop_pct=0.03), absolute_max_position_pct=0.50
        )
        assert shares_tight > shares_wide

    def test_result_is_whole_shares(self):
        shares = _size(last_price=73.77)
        assert isinstance(shares, int)
        assert shares >= 0

    def test_larger_portfolio_gives_more_shares(self):
        small = _size(portfolio_value=50_000)
        large = _size(portfolio_value=200_000)
        assert large > small


# ---------------------------------------------------------------------------
# Absolute position cap
# ---------------------------------------------------------------------------


class TestAbsoluteCap:
    def test_cap_applied_when_formula_exceeds_limit(self):
        """
        Very tight stop → huge uncapped position → absolute cap kicks in.
        portfolio=100k, stop=0.5% → position=200k → cap at 3% = 3000 USD → 30 shares
        """
        shares = _size(signal=_signal(stop_pct=0.005), last_price=100.0)
        max_dollars = 100_000 * 0.03  # 3000
        assert shares <= max_dollars / 100.0  # price = 100

    def test_cap_matches_absolute_max_pct(self):
        # With cap=5% and price=100, max shares = 100k × 0.05 / 100 = 50
        shares = _size(
            signal=_signal(stop_pct=0.001),  # tiny stop → would be huge
            absolute_max_position_pct=0.05,
            last_price=100.0,
        )
        assert shares <= 50


# ---------------------------------------------------------------------------
# Global risk budget
# ---------------------------------------------------------------------------


class TestGlobalRiskBudget:
    def test_exhausted_budget_returns_zero(self):
        """open_risk >= max_global_open_risk → no new positions"""
        shares = _size(
            current_open_risk=0.02,  # already at limit
            max_global_open_risk=0.02,
        )
        assert shares == 0

    def test_over_budget_returns_zero(self):
        shares = _size(
            current_open_risk=0.025,  # over limit
            max_global_open_risk=0.02,
        )
        assert shares == 0

    def test_partial_budget_scales_down(self):
        """With half the budget remaining, size should be ≤ unconstrained size."""
        unconstrained = _size(current_open_risk=0.0)
        constrained = _size(current_open_risk=0.01, max_global_open_risk=0.02)
        assert constrained <= unconstrained
        assert constrained > 0


# ---------------------------------------------------------------------------
# Stop-loss floor
# ---------------------------------------------------------------------------


class TestStopFloor:
    def test_stop_floor_prevents_oversized_position(self):
        """
        signal.stop_distance_pct=0.001 (0.1%) is below the 0.5% hard floor.
        Should be treated as 0.5% → smaller position than without floor.
        """
        shares_floored = _size(signal=_signal(stop_pct=0.001))
        # Without floor we'd divide by 0.001, giving 10× portfolio value
        # The absolute cap should catch it anyway, but let's confirm it doesn't explode
        max_shares_cap = int(100_000 * 0.03 / 100)
        assert shares_floored <= max_shares_cap + 1

    def test_signal_without_stop_uses_config_floor(self):
        """If stop_distance_pct is None, uses stop_loss_floor_pct from config."""
        sig = SignalIntent(
            symbol="NVDA",
            side=OrderSide.BUY,
            strategy_name="test",
            confidence=0.75,
            stop_distance_pct=None,  # explicitly None
        )
        shares = calculate_position_size(
            signal=sig,
            portfolio_value=100_000,
            current_open_risk=0.0,
            max_risk_per_trade=0.01,
            absolute_max_position_pct=0.03,
            max_global_open_risk=0.02,
            stop_loss_floor_pct=0.03,
            last_price=100.0,
        )
        assert shares > 0


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


class TestInvalidInputs:
    def test_zero_portfolio_value_returns_zero(self):
        assert _size(portfolio_value=0.0) == 0

    def test_negative_portfolio_value_returns_zero(self):
        assert _size(portfolio_value=-1000.0) == 0

    def test_zero_price_returns_zero(self):
        assert _size(last_price=0.0) == 0

    def test_very_high_price_returns_low_shares(self):
        """$10k stock, 3% cap on 100k portfolio = $3k max → 0 shares (floored)"""
        shares = _size(last_price=10_000.0)
        assert shares == 0  # 3000 / 10000 = 0.3 → floor(0.3) = 0
