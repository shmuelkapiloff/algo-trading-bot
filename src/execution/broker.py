"""
Abstract Broker Interface — Phase 1.

Decouples the execution layer from any specific brokerage API.
The AlpacaBroker implementation lives in alpaca.py; future brokers
(IBKR, TD Ameritrade, etc.) implement this same interface without
touching any other layer.

All methods are async. Blocking SDK calls must be wrapped in
asyncio.to_thread() by the implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class BrokerOrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    MARKETABLE_LIMIT = "marketable_limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    BRACKET = "bracket"


class BrokerOrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class BrokerOrderStatus(str, Enum):
    NEW = "new"
    ACCEPTED = "accepted"
    PENDING = "pending"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class BrokerOrder:
    """The canonical representation of an order returned by the broker."""

    broker_order_id: str
    client_order_id: str
    symbol: str
    side: BrokerOrderSide
    qty: float
    filled_qty: float
    order_type: BrokerOrderType
    status: BrokerOrderStatus
    limit_price: Optional[float]
    stop_price: Optional[float]
    filled_avg_price: Optional[float]
    submitted_at: Optional[datetime]
    filled_at: Optional[datetime]
    raw: dict[str, Any]  # broker-native response (for audit)


@dataclass
class BrokerPosition:
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pnl: float
    current_price: float


@dataclass
class BrokerAccount:
    account_id: str
    equity: float
    cash: float
    buying_power: float
    pattern_day_trader: bool
    trading_blocked: bool


class AbstractBroker(ABC):
    """
    Broker interface contract. All implementations must satisfy this.
    """

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    @abstractmethod
    async def submit_market_order(
        self,
        symbol: str,
        side: BrokerOrderSide,
        qty: int,
        client_order_id: Optional[str] = None,
    ) -> BrokerOrder:
        """Submit a market order. Use only when explicitly allowed by config."""

    @abstractmethod
    async def submit_marketable_limit_order(
        self,
        symbol: str,
        side: BrokerOrderSide,
        qty: int,
        limit_price: float,
        client_order_id: Optional[str] = None,
    ) -> BrokerOrder:
        """
        Submit a marketable limit order.
        Buy:  limit = ask × (1 + buffer_bps/10000)
        Sell: limit = bid × (1 - buffer_bps/10000)
        This is the default order type (execution_safety.default_order_type).
        """

    @abstractmethod
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
        """Submit a bracket order with stop-loss and take-profit legs."""

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order. Returns True if successfully cancelled."""

    @abstractmethod
    async def get_order(self, broker_order_id: str) -> BrokerOrder:
        """Fetch the current status of an order."""

    @abstractmethod
    async def list_open_orders(self) -> list[BrokerOrder]:
        """Return all open (non-terminal) orders."""

    # ------------------------------------------------------------------
    # Account / positions
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_account(self) -> BrokerAccount:
        """Return current account state (equity, buying power, PDT flag)."""

    @abstractmethod
    async def list_positions(self) -> list[BrokerPosition]:
        """Return all current open positions."""

    @abstractmethod
    async def close_position(self, symbol: str) -> Optional[BrokerOrder]:
        """Close the entire position for a symbol. Returns the closing order."""

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the broker API is reachable and functional."""
