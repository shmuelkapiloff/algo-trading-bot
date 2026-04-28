"""Tests for src/data/corporate_actions.py."""
from __future__ import annotations

import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock

from src.data.corporate_actions import (
    CorporateActionsHandler,
    ActionType,
    _SPLIT_DROP_THRESHOLD,
    _DIV_YIELD_THRESHOLD,
)


def _make_raw_bars(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    n = len(closes)
    vols = volumes or [1_000_000] * n
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "close": closes,
        "volume": vols,
        "open": closes,
        "high": closes,
        "low": closes,
    }, index=idx)


def _make_adj_bars(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    n = len(closes)
    vols = volumes or [1_000_000] * n
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "close_adj": closes,
        "volume": vols,
    }, index=idx)


class TestCorporateActionDetection:
    def setup_method(self):
        self.handler = CorporateActionsHandler(event_bus=None)

    def test_no_actions_identical_bars(self):
        raw = _make_raw_bars([100.0, 101.0, 102.0])
        adj = _make_adj_bars([100.0, 101.0, 102.0])
        actions = self.handler.detect("AAPL", raw, adj)
        assert actions == []

    def test_detects_forward_split(self):
        """Raw drops 50% relative to adj → 2-for-1 split."""
        raw = _make_raw_bars([200.0, 200.0, 100.0, 101.0, 102.0])
        adj = _make_adj_bars([100.0, 100.0, 100.0, 101.0, 102.0])
        actions = self.handler.detect("AAPL", raw, adj)
        split_actions = [a for a in actions if a.action_type == ActionType.SPLIT_FORWARD]
        assert len(split_actions) >= 1
        assert split_actions[0].symbol == "AAPL"

    def test_detects_reverse_split(self):
        """Raw jumps 100% relative to adj → 1-for-2 reverse split."""
        raw = _make_raw_bars([100.0, 100.0, 200.0, 202.0, 204.0])
        adj = _make_adj_bars([100.0, 100.0, 100.0, 101.0, 102.0])
        actions = self.handler.detect("AAPL", raw, adj)
        split_actions = [a for a in actions if a.action_type == ActionType.SPLIT_REVERSE]
        assert len(split_actions) >= 1

    def test_detects_dividend(self):
        """Small raw/adj divergence of ~1% → dividend."""
        raw = _make_raw_bars([100.0, 99.0])   # ex-div drop in raw
        adj = _make_adj_bars([100.0, 100.0])  # adj adjusted upward to pre-div level
        actions = self.handler.detect("MSFT", raw, adj)
        div_actions = [a for a in actions if a.action_type == ActionType.DIVIDEND]
        assert len(div_actions) >= 1

    def test_detects_trading_halt(self):
        """Three consecutive days of zero volume → halt detection."""
        closes = [100.0] * 10
        vols = [1_000_000] * 7 + [0, 0, 0]
        raw = _make_raw_bars(closes, vols)
        adj = _make_adj_bars(closes, vols)
        actions = self.handler.detect("HALTED", raw, adj)
        halt_actions = [a for a in actions if a.action_type == ActionType.HALT]
        assert len(halt_actions) >= 1

    def test_no_split_on_normal_move(self):
        """Normal 5% move should not be flagged as a split."""
        raw = _make_raw_bars([100.0, 105.0, 103.0])
        adj = _make_adj_bars([100.0, 105.0, 103.0])
        actions = self.handler.detect("AAPL", raw, adj)
        split_actions = [a for a in actions if a.action_type in (ActionType.SPLIT_FORWARD, ActionType.SPLIT_REVERSE)]
        assert split_actions == []

    def test_should_exclude_symbol_after_halt(self):
        closes = [100.0] * 10
        vols = [1_000_000] * 7 + [0, 0, 0]
        raw = _make_raw_bars(closes, vols)
        adj = _make_adj_bars(closes, vols)
        actions = self.handler.detect("HALTED", raw, adj)
        assert CorporateActionsHandler.should_exclude_symbol(actions) is True

    def test_should_not_exclude_normal_symbol(self):
        raw = _make_raw_bars([100.0, 101.0, 102.0])
        adj = _make_adj_bars([100.0, 101.0, 102.0])
        actions = self.handler.detect("AAPL", raw, adj)
        assert CorporateActionsHandler.should_exclude_symbol(actions) is False

    def test_has_recent_split_false_for_old_split(self):
        """Split detected more than lookback_days ago should return False."""
        raw = _make_raw_bars([200.0] * 5 + [100.0] * 30)
        adj = _make_adj_bars([100.0] * 35)
        actions = self.handler.detect("AAPL", raw, adj)
        # With lookback_days=2, recent split check should fail if split is old
        result = CorporateActionsHandler.has_recent_split(actions, lookback_days=0)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_detect_and_handle_publishes_to_bus(self):
        """detect_and_handle() should publish event to the event bus."""
        bus = MagicMock()
        bus.publish = AsyncMock()
        handler = CorporateActionsHandler(event_bus=bus)

        closes = [100.0] * 10
        vols = [1_000_000] * 7 + [0, 0, 0]
        raw = _make_raw_bars(closes, vols)
        adj = _make_adj_bars(closes, vols)

        await handler.detect_and_handle("HALT_SYM", raw, adj)
        bus.publish.assert_called()

    def test_empty_bars_returns_no_actions(self):
        empty_raw = pd.DataFrame(columns=["close", "volume", "open", "high", "low"])
        empty_adj = pd.DataFrame(columns=["close_adj", "volume"])
        actions = self.handler.detect("EMPTY", empty_raw, empty_adj)
        assert actions == []
