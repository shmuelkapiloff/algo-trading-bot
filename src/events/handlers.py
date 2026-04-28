"""
Event handlers — respond to ORDER_FILLED / ORDER_REJECTED / ORDER_CANCELED.

Each handler is a coroutine registered with EventBus.subscribe().
They are the ONLY components that mutate portfolio state in response to
exchange events. This enforces a single write-path.

Expected payload schemas
------------------------
ORDER_FILLED:
  {
    "order_id":         str,
    "symbol":           str,
    "side":             "buy" | "sell",
    "filled_qty":       int,
    "fill_price":       float,
    "strategy_name":    str,
    "stop_distance_pct": float,
    "occurred_at":      str (ISO8601)
  }

ORDER_REJECTED / ORDER_CANCELED:
  {
    "order_id":   str,
    "symbol":     str,
    "reason":     str,
    "strategy_name": str
  }
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from ..data.oms_ledger import OmsLedger
from ..data.models import OrderEventType
from ..portfolio.manager import PortfolioManager
from ..signals.models import OrderSide

logger = logging.getLogger(__name__)


class FillHandler:
    """
    Handles ORDER_FILLED events:
      1. Records a FILLED event in the OMS ledger (idempotent).
      2. Updates PortfolioManager position state.
      3. Records TCA metrics (slippage, fill rate, latency) if TcaMonitor present.
    """

    def __init__(
        self,
        ledger: OmsLedger,
        portfolio: PortfolioManager,
        tca_monitor: Optional[object] = None,
    ) -> None:
        self._ledger = ledger
        self._portfolio = portfolio
        self._tca = tca_monitor

    async def __call__(self, topic: str, payload: dict[str, Any]) -> None:
        order_id = payload.get("order_id", "")
        symbol = payload.get("symbol", "")
        side_str = payload.get("side", "buy")
        filled_qty = int(payload.get("filled_qty", 0))
        fill_price = float(payload.get("fill_price", 0.0))
        strategy_name = payload.get("strategy_name", "")
        stop_dist = float(payload.get("stop_distance_pct", 0.0))
        occurred_at_str = payload.get("occurred_at")

        occurred_at = (
            datetime.fromisoformat(occurred_at_str)
            if occurred_at_str
            else datetime.now(timezone.utc)
        )

        # ── OMS ledger record ─────────────────────────────────────────
        await self._ledger.record_event(
            order_id=order_id,
            symbol=symbol,
            event_type=OrderEventType.FILLED,
            payload=payload,
            strategy=strategy_name,
            occurred_at=occurred_at,
        )

        # ── Portfolio state update ────────────────────────────────────
        side = OrderSide.BUY if side_str == "buy" else OrderSide.SELL
        await self._portfolio.record_fill(
            order_id=order_id,
            symbol=symbol,
            side=side,
            filled_qty=filled_qty,
            fill_price=fill_price,
            strategy_name=strategy_name,
            stop_distance_pct=stop_dist,
        )

        logger.info(
            "FillHandler: %s %s %d@%.2f order_id=%s",
            side_str.upper(),
            symbol,
            filled_qty,
            fill_price,
            order_id,
        )

        # ── TCA recording ─────────────────────────────────────────────
        if self._tca is not None:
            # vwap_benchmark: price at order-submission time (included in payload
            # when available); fall back to fill_price (0 slippage) for Phase 1.
            vwap_benchmark = float(
                payload.get("vwap_benchmark", fill_price) or fill_price
            )
            requested_qty = int(payload.get("requested_qty", filled_qty) or filled_qty)
            fill_latency_ms = float(payload.get("fill_latency_ms", 0.0) or 0.0)

            try:
                await self._tca.record_fill(
                    order_id=order_id,
                    symbol=symbol,
                    fill_price=fill_price,
                    vwap_benchmark=vwap_benchmark,
                    filled_qty=filled_qty,
                    requested_qty=requested_qty,
                    fill_latency_ms=fill_latency_ms,
                    side=side_str,
                )
            except Exception:
                logger.exception("TCA record_fill failed for order_id=%s", order_id)


class RejectHandler:
    """
    Handles ORDER_REJECTED events:
      1. Records a REJECTED event in the OMS ledger.
      2. Logs a warning so the operator can investigate.
    """

    def __init__(self, ledger: OmsLedger) -> None:
        self._ledger = ledger

    async def __call__(self, topic: str, payload: dict[str, Any]) -> None:
        order_id = payload.get("order_id", "")
        symbol = payload.get("symbol", "")
        reason = payload.get("reason", "unknown")
        strategy_name = payload.get("strategy_name", "")

        await self._ledger.record_event(
            order_id=order_id,
            symbol=symbol,
            event_type=OrderEventType.REJECTED,
            payload=payload,
            strategy=strategy_name,
        )

        logger.warning(
            "RejectHandler: order REJECTED symbol=%s order_id=%s reason=%s",
            symbol,
            order_id,
            reason,
        )


class CancelHandler:
    """
    Handles ORDER_CANCELED events:
      1. Records a CANCELED event in the OMS ledger.
    """

    def __init__(self, ledger: OmsLedger) -> None:
        self._ledger = ledger

    async def __call__(self, topic: str, payload: dict[str, Any]) -> None:
        order_id = payload.get("order_id", "")
        symbol = payload.get("symbol", "")
        reason = payload.get("reason", "")
        strategy_name = payload.get("strategy_name", "")

        await self._ledger.record_event(
            order_id=order_id,
            symbol=symbol,
            event_type=OrderEventType.CANCELED,
            payload=payload,
            strategy=strategy_name,
        )

        logger.info(
            "CancelHandler: order CANCELED symbol=%s order_id=%s reason=%s",
            symbol,
            order_id,
            reason,
        )


def register_handlers(
    bus,
    ledger: OmsLedger,
    portfolio: PortfolioManager,
    tca_monitor: Optional[object] = None,
) -> None:
    """
    Register all event handlers with the EventBus.
    Call once during application startup.
    """
    from . import topics

    fill_handler = FillHandler(ledger, portfolio, tca_monitor)
    reject_handler = RejectHandler(ledger)
    cancel_handler = CancelHandler(ledger)

    bus.subscribe(topics.ORDER_FILLED, fill_handler)
    bus.subscribe(topics.ORDER_REJECTED, reject_handler)
    bus.subscribe(topics.ORDER_CANCELED, cancel_handler)

    logger.info("Event handlers registered: FillHandler, RejectHandler, CancelHandler")
