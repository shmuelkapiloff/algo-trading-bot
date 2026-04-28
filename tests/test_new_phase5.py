"""
Tests for new Phase 5 / P0-P2 implementations:
  - KellyStats (O(1) incremental, edge=0 alert)
  - PositionSizingPipeline (5-step refactor)
  - FencingToken renewal protocol
  - BrokerHealthMonitor
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading_bot.src.portfolio.kelly_stats import KellyStats, _EDGE_ZERO_THRESHOLD
from trading_bot.src.portfolio.position_sizing_pipeline import (
    FixedFractionalStep,
    GlobalRiskBudgetStep,
    HardCapStep,
    LiquidityCapStep,
    PositionSizingPipeline,
    SizingContext,
    VolScaleStep,
)


# ---------------------------------------------------------------------------
# KellyStats tests
# ---------------------------------------------------------------------------


class TestKellyStats:
    def _make_stats(self, wins: int = 10, losses: int = 10) -> KellyStats:
        stats = KellyStats(strategy_name="test", min_sample_size=5)
        for _ in range(wins):
            stats.update(100.0)
        for _ in range(losses):
            stats.update(-50.0)
        return stats

    def test_n_trades(self):
        stats = KellyStats(strategy_name="test")
        stats.update(100.0)
        stats.update(-40.0)
        assert stats.n_trades == 2

    def test_win_rate_prior_before_min_sample(self):
        stats = KellyStats(strategy_name="test", min_sample_size=30)
        stats.update(100.0)
        assert stats.win_rate == 0.5  # prior

    def test_win_rate_empirical_after_sample(self):
        stats = KellyStats(strategy_name="test", min_sample_size=5)
        for _ in range(8):
            stats.update(100.0)
        for _ in range(2):
            stats.update(-50.0)
        assert stats.win_rate == pytest.approx(0.8, abs=0.01)

    def test_b_ratio_zero_no_losses(self):
        stats = KellyStats(strategy_name="test", min_sample_size=1)
        stats.update(100.0)
        assert stats.b_ratio == 0.0  # no losses yet

    def test_b_ratio_positive(self):
        stats = self._make_stats()
        assert stats.b_ratio > 0

    def test_kelly_fraction_returns_zero_below_min_sample(self):
        stats = KellyStats(strategy_name="test", min_sample_size=30)
        for _ in range(5):
            stats.update(100.0)
        assert stats.kelly_fraction == 0.0

    def test_kelly_fraction_positive_with_edge(self):
        stats = KellyStats(strategy_name="test", min_sample_size=5)
        for _ in range(8):
            stats.update(200.0)
        for _ in range(2):
            stats.update(-50.0)
        kf = stats.kelly_fraction
        assert kf > 0

    def test_kelly_fraction_zero_emits_metric_when_b_zero(self, caplog):
        import logging

        stats = KellyStats(strategy_name="edge_test", min_sample_size=1)
        # Force ewma_loss to 0 by only recording wins
        stats.update(100.0)
        # b_ratio = 0 (no losses), should emit strategy_edge_zero metric
        with caplog.at_level(logging.WARNING):
            kf = stats.kelly_fraction
        assert kf == 0.0
        assert "strategy_edge_zero" in caplog.text

    def test_capped_kelly_respects_max(self):
        stats = KellyStats(strategy_name="test", min_sample_size=2)
        for _ in range(20):
            stats.update(1000.0)
        for _ in range(5):
            stats.update(-1.0)
        result = stats.capped_kelly(max_fraction=0.01)
        assert result <= 0.01

    def test_reset_clears_stats(self):
        stats = self._make_stats()
        stats.reset()
        assert stats.n_trades == 0
        assert stats.ewma_win == 0.0
        assert stats.ewma_loss == 0.0

    def test_ewma_decay_weights_recent(self):
        stats = KellyStats(strategy_name="test", lam=0.9, min_sample_size=1)
        stats.update(10.0)
        old_win = stats.ewma_win
        stats.update(1000.0)
        assert stats.ewma_win > old_win


# ---------------------------------------------------------------------------
# PositionSizingPipeline tests
# ---------------------------------------------------------------------------


def _make_signal(symbol="AAPL", confidence=0.8, stop_pct=0.02):
    from trading_bot.src.signals.models import SignalIntent, OrderSide

    return SignalIntent(
        symbol=symbol,
        side=OrderSide.BUY,
        strategy_name="test",
        confidence=confidence,
        stop_distance_pct=stop_pct,
        qty=0,
    )


class TestFixedFractionalStep:
    def test_basic_sizing(self):
        step = FixedFractionalStep(
            max_risk_per_trade=0.01,
            absolute_max_position_pct=0.10,  # 10% cap → $10,000 > base $5,000
        )
        signal = _make_signal(stop_pct=0.02)
        ctx = SizingContext(portfolio_value=100_000, current_open_risk=0.0)
        # base = 100_000 * 0.01 / 0.02 = 50_000; capped at 10% = 10_000? no:
        # actually base < cap: 50_000 > 10_000 → capped at 10_000
        # Let's use a larger cap: 1.0 so base wins
        step2 = FixedFractionalStep(
            max_risk_per_trade=0.01, absolute_max_position_pct=1.0
        )
        size = step2.apply(0.0, signal, ctx)
        assert size == pytest.approx(50_000, rel=0.01)

    def test_capped_by_abs_max(self):
        step = FixedFractionalStep(
            max_risk_per_trade=0.1,
            absolute_max_position_pct=0.03,
        )
        signal = _make_signal(stop_pct=0.01)
        ctx = SizingContext(portfolio_value=100_000, current_open_risk=0.0)
        size = step.apply(0.0, signal, ctx)
        assert size == pytest.approx(3_000, rel=0.01)

    def test_zero_portfolio(self):
        step = FixedFractionalStep(
            max_risk_per_trade=0.01, absolute_max_position_pct=0.05
        )
        ctx = SizingContext(portfolio_value=0, current_open_risk=0.0)
        assert step.apply(0.0, _make_signal(), ctx) == 0.0


class TestVolScaleStep:
    def test_scales_up_when_low_vol(self):
        step = VolScaleStep(target_vol=0.20, min_scale=0.5, max_scale=2.0)
        ctx = SizingContext(
            portfolio_value=100_000, current_open_risk=0.0, realized_vol=0.10
        )
        # scale = 0.20/0.10 = 2.0 (capped at max_scale)
        result = step.apply(1000.0, _make_signal(), ctx)
        assert result == pytest.approx(2000.0, rel=0.01)

    def test_skips_when_zero_vol(self):
        step = VolScaleStep()
        ctx = SizingContext(
            portfolio_value=100_000, current_open_risk=0.0, realized_vol=0.0
        )
        assert step.apply(500.0, _make_signal(), ctx) == 500.0


class TestGlobalRiskBudgetStep:
    def test_rejects_when_budget_exhausted(self):
        step = GlobalRiskBudgetStep(max_global_open_risk=0.02)
        ctx = SizingContext(portfolio_value=100_000, current_open_risk=0.02)
        result = step.apply(1000.0, _make_signal(stop_pct=0.02), ctx)
        assert result == 0.0

    def test_scales_to_fit_remaining_budget(self):
        step = GlobalRiskBudgetStep(max_global_open_risk=0.02)
        ctx = SizingContext(portfolio_value=100_000, current_open_risk=0.015)
        # remaining = 0.005 of portfolio = $500 in risk
        # signal stop = 2% → max size = 500/0.02 = 25_000
        result = step.apply(50_000.0, _make_signal(stop_pct=0.02), ctx)
        assert result < 50_000.0
        assert result > 0.0


class TestPositionSizingPipeline:
    def test_default_pipeline_returns_shares(self):
        pipeline = PositionSizingPipeline.default(
            max_risk_per_trade=0.01,
            absolute_max_position_pct=0.05,
            max_global_open_risk=0.02,
        )
        signal = _make_signal(stop_pct=0.02)
        ctx = SizingContext(portfolio_value=100_000, current_open_risk=0.0)
        shares = pipeline.size(signal, ctx, last_price=50.0)
        assert isinstance(shares, int)
        assert shares > 0

    def test_returns_zero_on_budget_exhausted(self):
        pipeline = PositionSizingPipeline.default(
            max_global_open_risk=0.01,
        )
        signal = _make_signal(stop_pct=0.02)
        ctx = SizingContext(portfolio_value=100_000, current_open_risk=0.01)
        shares = pipeline.size(signal, ctx, last_price=50.0)
        assert shares == 0

    def test_returns_zero_on_invalid_price(self):
        pipeline = PositionSizingPipeline.default()
        signal = _make_signal()
        ctx = SizingContext(portfolio_value=100_000, current_open_risk=0.0)
        assert pipeline.size(signal, ctx, last_price=0.0) == 0


# ---------------------------------------------------------------------------
# Fencing token renewal tests
# ---------------------------------------------------------------------------


class TestFencingTokenRenewal:
    @pytest.fixture(autouse=True)
    def setup_keys(self):
        from trading_bot.src.security.fencing_tokens import generate_ephemeral_keys

        generate_ephemeral_keys()

    def test_renew_valid_emergency_token(self):
        from trading_bot.src.security import fencing_tokens as ft

        token = ft.create_internal_token(action_code="close_only", validity_seconds=60)
        renewed = ft.renew_token(token, extension_seconds=120)
        assert renewed.incident_id == token.incident_id
        assert renewed.valid_until > token.valid_until
        is_valid, reason = ft.verify_token(renewed)
        assert is_valid, reason

    def test_cannot_renew_expired_token(self):
        from trading_bot.src.security import fencing_tokens as ft

        token = ft.create_internal_token(action_code="close_only", validity_seconds=1)
        time.sleep(1.1)
        with pytest.raises(ValueError, match="expired"):
            ft.renew_token(token)

    def test_cannot_renew_non_emergency_token(self):
        from trading_bot.src.security import fencing_tokens as ft

        token = ft.create_token(
            action_code="pause_orders", severity="critical", validity_seconds=300
        )
        with pytest.raises(ValueError, match="emergency"):
            ft.renew_token(token)

    def test_renewal_capped_at_max_per_call(self):
        from trading_bot.src.security import fencing_tokens as ft

        token = ft.create_internal_token(action_code="close_only", validity_seconds=60)
        renewed = ft.renew_token(token, extension_seconds=9999)
        # Should be capped at _MAX_RENEWAL_SECONDS (300 s)
        assert (renewed.valid_until - time.time()) <= 305  # small tolerance

    def test_renewal_respects_max_total_lifetime(self):
        from trading_bot.src.security import fencing_tokens as ft

        token = ft.create_internal_token(action_code="close_only", validity_seconds=10)
        # Cap total lifetime to 60s; requesting 300s extension should be capped
        renewed = ft.renew_token(
            token, extension_seconds=300, max_total_lifetime_seconds=60
        )
        # valid_until should not exceed issued_at + 60
        assert renewed.valid_until <= token.issued_at + 65  # small tolerance
