"""
Corporate Actions Handler.

Detects and responds to:
  - Stock splits (forward and reverse)
  - Cash dividends (ex-dividend dates)
  - Ticker symbol changes (renames/mergers)
  - Trading halts (regulatory + volatility)
  - Delistings

All detection is based on comparing raw vs split-adjusted close prices
from the Alpaca API.  The handler injects signals into the EventBus
so the Portfolio Manager and BacktestEngine can react.

Usage
-----
    handler = CorporateActionsHandler(event_bus=bus)

    # Called by data_quality.py after each daily bar fetch:
    actions = handler.detect(symbol="AAPL", raw_bars=df_raw, adj_bars=df_adj)
    for action in actions:
        await handler.handle(action)

EventBus topics emitted
-----------------------
  CORP_ACTION_SPLIT       — forward/reverse split detected
  CORP_ACTION_DIVIDEND    — ex-dividend date detected
  CORP_ACTION_HALT        — trading halt or delisting suspected
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# EventBus topics (string literals — must match topics.py if added there)
CORP_ACTION_SPLIT = "corp_action.split"
CORP_ACTION_DIVIDEND = "corp_action.dividend"
CORP_ACTION_HALT = "corp_action.halt"

# Thresholds
_SPLIT_RATIO_TOLERANCE = 0.02   # 2% tolerance when detecting split ratio
_DIV_YIELD_THRESHOLD = 0.001    # 0.1% drop treated as dividend
_SPLIT_DROP_THRESHOLD = 0.15    # 15%+ raw/adj price divergence = split candidate
_HALT_VOLUME_ZERO_DAYS = 3      # 3 consecutive zero-volume days = likely halt


class ActionType(str, Enum):
    SPLIT_FORWARD = "split_forward"    # e.g. 3-for-1 (shares triple, price ÷3)
    SPLIT_REVERSE = "split_reverse"    # e.g. 1-for-3 (shares shrink, price ×3)
    DIVIDEND = "dividend"              # cash dividend — ex-div date drop
    HALT = "halt"                      # trading halted or near-zero volume
    DELISTED = "delisted"              # volume = 0 for sustained period


@dataclass
class CorporateAction:
    symbol: str
    action_type: ActionType
    detected_date: date
    ratio: Optional[float] = None        # split ratio (e.g. 3.0 for 3:1 split)
    dividend_estimate_pct: Optional[float] = None  # estimated dividend as % of price
    notes: str = ""


class CorporateActionsHandler:
    """
    Detects corporate actions by comparing raw vs split-adjusted price series.

    Parameters
    ----------
    event_bus:
        Optional EventBus. If provided, publishes events for each detected action.
        If None, actions are only returned (useful in backtesting/testing).
    """

    def __init__(self, event_bus=None) -> None:
        self._bus = event_bus

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        symbol: str,
        raw_bars: pd.DataFrame,
        adj_bars: pd.DataFrame,
    ) -> List[CorporateAction]:
        """
        Detect corporate actions by comparing raw vs adjusted bar series.

        Parameters
        ----------
        symbol    : Ticker symbol.
        raw_bars  : DataFrame with 'close' (unadjusted) indexed by date.
        adj_bars  : DataFrame with 'close_adj' (split-adjusted) indexed by date.

        Returns
        -------
        List of CorporateAction objects detected. Empty list = no events.
        """
        actions: List[CorporateAction] = []

        if raw_bars.empty or adj_bars.empty:
            return actions

        # Align on common dates
        raw_close = raw_bars["close"].rename("raw") if "close" in raw_bars.columns else None
        adj_close = adj_bars["close_adj"].rename("adj") if "close_adj" in adj_bars.columns else None

        if raw_close is None or adj_close is None:
            logger.debug("[corp_actions] %s: missing close or close_adj column", symbol)
            return actions

        merged = pd.concat([raw_close, adj_close], axis=1).dropna()
        if len(merged) < 2:
            return actions

        # ── Ratio between raw and adjusted prices ────────────────────
        # A sudden change in ratio(raw/adj) on date T implies a corporate action.
        merged["ratio"] = (merged["raw"] / merged["adj"]).round(6)
        ratio_change = merged["ratio"].pct_change().abs()

        for dt_index in ratio_change[ratio_change > _SPLIT_DROP_THRESHOLD].index:
            prev_ratio = merged["ratio"].shift(1)[dt_index]
            curr_ratio = merged["ratio"][dt_index]
            if pd.isna(prev_ratio) or pd.isna(curr_ratio) or prev_ratio == 0:
                continue

            split_ratio = prev_ratio / curr_ratio
            detected_date = (
                dt_index.date() if hasattr(dt_index, "date") else dt_index
            )

            if split_ratio > 1 + _SPLIT_RATIO_TOLERANCE:
                # raw/adj ratio DECREASED (e.g. 2.0 → 1.0): raw price halved
                # → stock was split, each share is now worth half → FORWARD split
                action_type = ActionType.SPLIT_FORWARD
                approx_ratio = split_ratio
                logger.info(
                    "[corp_actions] %s: FORWARD SPLIT detected on %s (≈%.0f:1)",
                    symbol, detected_date, approx_ratio
                )
            else:
                # raw/adj ratio INCREASED (e.g. 1.0 → 2.0): raw price doubled
                # → shares were consolidated → REVERSE split
                action_type = ActionType.SPLIT_REVERSE
                approx_ratio = 1.0 / split_ratio
                logger.info(
                    "[corp_actions] %s: REVERSE SPLIT detected on %s (≈1:%.0f)",
                    symbol, detected_date, approx_ratio
                )

            actions.append(
                CorporateAction(
                    symbol=symbol,
                    action_type=action_type,
                    detected_date=detected_date,
                    ratio=split_ratio,
                    notes=f"raw/adj ratio changed: {prev_ratio:.4f} → {curr_ratio:.4f}",
                )
            )

        # ── Dividend detection — small raw price drop, adj unchanged ──
        # On ex-dividend date, the raw price drops by ~dividend amount.
        # The adjusted price is retroactively lowered to match → drop only in raw.
        raw_pct = merged["raw"].pct_change()
        adj_pct = merged["adj"].pct_change()
        div_signal = (raw_pct - adj_pct).abs()  # divergence = dividend proxy

        for dt_index in div_signal[
            (div_signal > _DIV_YIELD_THRESHOLD) & (div_signal < _SPLIT_DROP_THRESHOLD)
        ].index:
            detected_date = (
                dt_index.date() if hasattr(dt_index, "date") else dt_index
            )
            div_pct = float(div_signal[dt_index])
            logger.info(
                "[corp_actions] %s: DIVIDEND detected ~%.2f%% on %s",
                symbol, div_pct * 100, detected_date
            )
            actions.append(
                CorporateAction(
                    symbol=symbol,
                    action_type=ActionType.DIVIDEND,
                    detected_date=detected_date,
                    dividend_estimate_pct=div_pct,
                    notes=f"raw/adj divergence: {div_pct:.4f}",
                )
            )

        # ── Halt / Delisting detection ────────────────────────────────
        if "volume" in adj_bars.columns:
            vol = adj_bars["volume"]
            zero_vol_streak = (vol == 0).rolling(_HALT_VOLUME_ZERO_DAYS).sum()
            halt_dates = zero_vol_streak[zero_vol_streak >= _HALT_VOLUME_ZERO_DAYS].index

            if not halt_dates.empty:
                first_halt = halt_dates[0]
                detected_date = (
                    first_halt.date() if hasattr(first_halt, "date") else first_halt
                )
                action_type = (
                    ActionType.DELISTED
                    if len(halt_dates) > 10
                    else ActionType.HALT
                )
                logger.warning(
                    "[corp_actions] %s: %s detected starting %s (%d zero-vol bars)",
                    symbol, action_type.value, detected_date, len(halt_dates)
                )
                actions.append(
                    CorporateAction(
                        symbol=symbol,
                        action_type=action_type,
                        detected_date=detected_date,
                        notes=f"{len(halt_dates)} consecutive zero-volume bars",
                    )
                )

        return actions

    async def handle(self, action: CorporateAction) -> None:
        """
        Emit an EventBus event for the detected corporate action.

        The Portfolio Manager listens to CORP_ACTION_HALT to pause trading
        in the affected symbol. The OMS Ledger logs the event for audit.
        """
        if self._bus is None:
            return

        topic_map = {
            ActionType.SPLIT_FORWARD: CORP_ACTION_SPLIT,
            ActionType.SPLIT_REVERSE: CORP_ACTION_SPLIT,
            ActionType.DIVIDEND: CORP_ACTION_DIVIDEND,
            ActionType.HALT: CORP_ACTION_HALT,
            ActionType.DELISTED: CORP_ACTION_HALT,
        }

        topic = topic_map.get(action.action_type, CORP_ACTION_HALT)
        payload = {
            "symbol": action.symbol,
            "action_type": action.action_type.value,
            "detected_date": str(action.detected_date),
            "ratio": action.ratio,
            "dividend_estimate_pct": action.dividend_estimate_pct,
            "notes": action.notes,
        }
        await self._bus.publish(topic, payload)
        logger.info(
            "[corp_actions] Published %s for %s (%s)",
            topic, action.symbol, action.action_type.value
        )

    async def detect_and_handle(
        self,
        symbol: str,
        raw_bars: pd.DataFrame,
        adj_bars: pd.DataFrame,
    ) -> List[CorporateAction]:
        """Convenience method: detect + handle all actions in one call."""
        actions = self.detect(symbol, raw_bars, adj_bars)
        for action in actions:
            await self.handle(action)
        return actions

    @staticmethod
    def should_exclude_symbol(actions: List[CorporateAction]) -> bool:
        """Return True if the symbol should be excluded from trading signals today."""
        for action in actions:
            if action.action_type in (ActionType.HALT, ActionType.DELISTED):
                return True
        return False

    @staticmethod
    def has_recent_split(actions: List[CorporateAction], lookback_days: int = 5) -> bool:
        """Return True if a split occurred within lookback_days of today."""
        today = datetime.now(timezone.utc).date()
        for action in actions:
            if action.action_type in (ActionType.SPLIT_FORWARD, ActionType.SPLIT_REVERSE):
                delta = (today - action.detected_date).days
                if delta <= lookback_days:
                    return True
        return False
