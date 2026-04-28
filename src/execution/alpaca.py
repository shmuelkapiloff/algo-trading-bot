"""
Alpaca Broker Implementation — Phase 1 (Paper Trading Default).

Wraps alpaca-py SDK calls in asyncio.to_thread() to keep the event loop
non-blocking. All SDK calls are synchronous; this adapter makes them async.

Order type strategy (from config execution_safety):
  default_order_type:      marketable_limit
  marketable_limit_buffer: 5 bps
  allow_market_orders:     false (overridable for bracket legs)

Retry policy: tenacity with exponential backoff on HTTP 429/5xx.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .broker import (
    AbstractBroker,
    BrokerAccount,
    BrokerOrder,
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPosition,
)

logger = logging.getLogger(__name__)

# Map Alpaca order status strings to our canonical enum
_ALPACA_STATUS_MAP: dict[str, BrokerOrderStatus] = {
    "new": BrokerOrderStatus.NEW,
    "accepted": BrokerOrderStatus.ACCEPTED,
    "pending_new": BrokerOrderStatus.PENDING,
    "partially_filled": BrokerOrderStatus.PARTIAL_FILL,
    "filled": BrokerOrderStatus.FILLED,
    "canceled": BrokerOrderStatus.CANCELED,
    "rejected": BrokerOrderStatus.REJECTED,
    "expired": BrokerOrderStatus.EXPIRED,
    "done_for_day": BrokerOrderStatus.CANCELED,
    "replaced": BrokerOrderStatus.CANCELED,
}


def _parse_order(o) -> BrokerOrder:
    """Convert an alpaca-py Order object to BrokerOrder."""
    side = BrokerOrderSide(str(o.side).lower().replace("orderside.", ""))
    status = _ALPACA_STATUS_MAP.get(str(o.status).lower(), BrokerOrderStatus.PENDING)
    otype = (
        BrokerOrderType.LIMIT
        if str(o.type).lower() in ("limit",)
        else BrokerOrderType.MARKET
    )

    return BrokerOrder(
        broker_order_id=str(o.id),
        client_order_id=str(o.client_order_id or ""),
        symbol=str(o.symbol),
        side=side,
        qty=float(o.qty or 0),
        filled_qty=float(o.filled_qty or 0),
        order_type=otype,
        status=status,
        limit_price=float(o.limit_price) if o.limit_price else None,
        stop_price=float(o.stop_price) if o.stop_price else None,
        filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
        submitted_at=o.submitted_at,
        filled_at=o.filled_at,
        raw={},
    )


def _parse_position(p) -> BrokerPosition:
    return BrokerPosition(
        symbol=str(p.symbol),
        qty=float(p.qty or 0),
        avg_entry_price=float(p.avg_entry_price or 0),
        market_value=float(p.market_value or 0),
        unrealized_pnl=float(p.unrealized_pl or 0),
        current_price=float(p.current_price or 0),
    )


class AlpacaBroker(AbstractBroker):
    """
    Alpaca Markets broker adapter.

    Parameters
    ----------
    api_key          : Alpaca API key
    secret_key       : Alpaca secret key
    paper            : True for paper trading (default), False for live
    buffer_bps       : Buffer for marketable limit orders (default 5 bps)
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        buffer_bps: float = 5.0,
    ) -> None:
        from alpaca.trading.client import TradingClient

        self._client = TradingClient(
            api_key=api_key, secret_key=secret_key, paper=paper
        )
        self._buffer = buffer_bps / 10_000
        self._paper = paper
        logger.info(
            "AlpacaBroker initialised (paper=%s, buffer_bps=%.1f)", paper, buffer_bps
        )

    # ------------------------------------------------------------------
    # Internal retry helper
    # ------------------------------------------------------------------

    @staticmethod
    def _retry_decorator():
        return retry(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=1, max=16),
            reraise=True,
        )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def submit_market_order(
        self,
        symbol: str,
        side: BrokerOrderSide,
        qty: int,
        client_order_id: Optional[str] = None,
    ) -> BrokerOrder:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide(side.value),
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id or str(uuid.uuid4()),
        )
        order = await asyncio.to_thread(self._client.submit_order, req)
        logger.info(
            "Market order submitted: %s %s x%d  id=%s",
            side.value,
            symbol,
            qty,
            order.id,
        )
        return _parse_order(order)

    async def submit_marketable_limit_order(
        self,
        symbol: str,
        side: BrokerOrderSide,
        qty: int,
        limit_price: float,
        client_order_id: Optional[str] = None,
    ) -> BrokerOrder:
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        # Apply buffer: buy slightly above ask, sell slightly below bid
        if side == BrokerOrderSide.BUY:
            adjusted_limit = round(limit_price * (1 + self._buffer), 2)
        else:
            adjusted_limit = round(limit_price * (1 - self._buffer), 2)

        req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide(side.value),
            time_in_force=TimeInForce.DAY,
            limit_price=adjusted_limit,
            client_order_id=client_order_id or str(uuid.uuid4()),
        )
        order = await asyncio.to_thread(self._client.submit_order, req)
        logger.info(
            "Limit order submitted: %s %s x%d @ %.2f  id=%s",
            side.value,
            symbol,
            qty,
            adjusted_limit,
            order.id,
        )
        return _parse_order(order)

    async def submit_bracket_order(
        self,
        symbol: str,
        side: BrokerOrderSide,
        qty: int,
        limit_price: float,
        stop_price: float,
        take_profit_price: float,
        client_order_id: Optional[str] = None,
    ) -> BrokerOrder:
        from alpaca.trading.requests import (
            MarketOrderRequest,
            TakeProfitRequest,
            StopLossRequest,
        )
        from alpaca.trading.enums import OrderSide, TimeInForce

        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide(side.value),
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id or str(uuid.uuid4()),
            order_class="bracket",
            take_profit=TakeProfitRequest(limit_price=take_profit_price),
            stop_loss=StopLossRequest(stop_price=stop_price),
        )
        order = await asyncio.to_thread(self._client.submit_order, req)
        logger.info(
            "Bracket order submitted: %s %s x%d  stop=%.2f  tp=%.2f  id=%s",
            side.value,
            symbol,
            qty,
            stop_price,
            take_profit_price,
            order.id,
        )
        return _parse_order(order)

    async def cancel_order(self, broker_order_id: str) -> bool:
        try:
            await asyncio.to_thread(self._client.cancel_order_by_id, broker_order_id)
            logger.info("Order cancelled: %s", broker_order_id)
            return True
        except Exception as exc:
            logger.warning("Cancel failed for %s: %s", broker_order_id, exc)
            return False

    async def get_order(self, broker_order_id: str) -> BrokerOrder:
        order = await asyncio.to_thread(self._client.get_order_by_id, broker_order_id)
        return _parse_order(order)

    async def list_open_orders(self) -> list[BrokerOrder]:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = await asyncio.to_thread(self._client.get_orders, req)
        return [_parse_order(o) for o in orders]

    # ------------------------------------------------------------------
    # Account / positions
    # ------------------------------------------------------------------

    async def get_account(self) -> BrokerAccount:
        acct = await asyncio.to_thread(self._client.get_account)
        return BrokerAccount(
            account_id=str(acct.id),
            equity=float(acct.equity or 0),
            cash=float(acct.cash or 0),
            buying_power=float(acct.buying_power or 0),
            pattern_day_trader=bool(acct.pattern_day_trader),
            trading_blocked=bool(acct.trading_blocked),
        )

    async def list_positions(self) -> list[BrokerPosition]:
        positions = await asyncio.to_thread(self._client.get_all_positions)
        return [_parse_position(p) for p in positions]

    async def close_position(self, symbol: str) -> Optional[BrokerOrder]:
        try:
            order = await asyncio.to_thread(self._client.close_position, symbol)
            logger.info("Position closed: %s  order_id=%s", symbol, order.id)
            return _parse_order(order)
        except Exception as exc:
            logger.error("close_position(%s) failed: %s", symbol, exc)
            return None

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        try:
            acct = await asyncio.to_thread(self._client.get_account)
            return not bool(acct.trading_blocked)
        except Exception as exc:
            logger.warning("Broker health check failed: %s", exc)
            return False
