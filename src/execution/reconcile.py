"""
Startup Reconciliation — synchronise DB state against live Alpaca positions.

Runs once at startup (before any scan) and periodically every 15 minutes
during market hours to catch any broker-side fills or stop-losses that
were not reflected in our DB (e.g. after an ungraceful shutdown).

Reconciliation steps
--------------------
1. Fetch live positions from Alpaca (api.list_positions())
2. Fetch open positions from PortfolioManager (Redis)
3. Detect and resolve 3 discrepancy types:
   a. Broker has position, Redis does not  → import it
   b. Redis has position, broker does not  → mark closed
   c. Both have position, qty differs      → update qty and warn
4. Fetch open orders from Alpaca
5. For each open order that exists in OMS as non-terminal:
   a. If broker says FILLED → record FILLED in OMS + update portfolio
   b. If broker says CANCELED/REJECTED → record accordingly
6. Log all discrepancies found.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from .broker import AbstractBroker, BrokerOrderStatus
from ..data.models import OrderEventType
from ..data.oms_ledger import OmsLedger
from ..portfolio.manager import PortfolioManager

logger = logging.getLogger(__name__)


class ReconciliationService:
    """
    Reconciles in-memory/Redis portfolio state against live broker state.

    Parameters
    ----------
    broker           : AbstractBroker (Alpaca or mock)
    portfolio_manager: PortfolioManager (Redis-backed)
    oms_ledger       : OmsLedger for recording discovered events
    """

    def __init__(
        self,
        broker: AbstractBroker,
        portfolio_manager: PortfolioManager,
        oms_ledger: OmsLedger,
    ) -> None:
        self._broker = broker
        self._portfolio = portfolio_manager
        self._ledger = oms_ledger

    async def run(self) -> dict:
        """
        Run a full reconciliation pass.
        Returns a summary dict: {imported, closed, updated, order_updates}.
        """
        summary = {"imported": 0, "closed": 0, "updated": 0, "order_updates": 0}

        # ── Position reconciliation ───────────────────────────────────
        try:
            broker_positions = await self._broker.list_positions()
        except Exception as exc:
            logger.error("reconcile: failed to fetch broker positions: %s", exc)
            return summary

        broker_by_symbol = {p.symbol: p for p in broker_positions}

        # Fetch portfolio state from in-memory cache (already hydrated from Redis)
        redis_by_symbol = dict(self._portfolio.open_positions)

        # Case a: broker has position, Redis does not → import
        for symbol, bp in broker_by_symbol.items():
            if symbol not in redis_by_symbol:
                logger.warning(
                    "reconcile: symbol %s found in broker but not Redis — importing "
                    "(qty=%.0f  avg_price=%.2f)",
                    symbol,
                    bp.qty,
                    bp.avg_entry_price,
                )
                await self._portfolio.import_position(
                    symbol=symbol,
                    qty=int(bp.qty),
                    avg_price=bp.avg_entry_price,
                )
                summary["imported"] += 1

        # Case b: Redis has position, broker does not → mark closed
        for symbol, rp in redis_by_symbol.items():
            if symbol not in broker_by_symbol:
                logger.warning(
                    "reconcile: symbol %s in Redis but not broker — marking closed",
                    symbol,
                )
                await self._portfolio.remove_position(symbol)
                summary["closed"] += 1

        # Case c: qty mismatch
        for symbol in set(broker_by_symbol) & set(redis_by_symbol):
            bp_qty = int(broker_by_symbol[symbol].qty)
            rp_qty = redis_by_symbol[symbol].qty
            if abs(bp_qty - rp_qty) >= 1:
                logger.warning(
                    "reconcile: %s qty mismatch — broker=%d  redis=%d — updating Redis",
                    symbol,
                    bp_qty,
                    rp_qty,
                )
                await self._portfolio.update_position_qty(symbol, bp_qty)
                summary["updated"] += 1

        # ── Order reconciliation ──────────────────────────────────────
        try:
            open_orders = await self._broker.list_open_orders()
        except Exception as exc:
            logger.error("reconcile: failed to fetch open orders: %s", exc)
            return summary

        for broker_order in open_orders:
            status = broker_order.status

            if status == BrokerOrderStatus.FILLED:
                logger.info(
                    "reconcile: order %s filled (discovered) — recording FILLED",
                    broker_order.broker_order_id,
                )
                await self._ledger.record_event(
                    order_id=broker_order.client_order_id
                    or broker_order.broker_order_id,
                    symbol=broker_order.symbol,
                    event_type=OrderEventType.FILLED,
                    payload={
                        "broker_order_id": broker_order.broker_order_id,
                        "filled_qty": broker_order.filled_qty,
                        "fill_price": broker_order.filled_avg_price,
                        "discovered_by": "reconcile",
                    },
                    strategy="reconcile",
                )
                summary["order_updates"] += 1

            elif status in (
                BrokerOrderStatus.CANCELED,
                BrokerOrderStatus.REJECTED,
                BrokerOrderStatus.EXPIRED,
            ):
                await self._ledger.record_event(
                    order_id=broker_order.client_order_id
                    or broker_order.broker_order_id,
                    symbol=broker_order.symbol,
                    event_type=(
                        OrderEventType.CANCELED
                        if status == BrokerOrderStatus.CANCELED
                        else OrderEventType.REJECTED
                    ),
                    payload={
                        "broker_status": status.value,
                        "discovered_by": "reconcile",
                    },
                    strategy="reconcile",
                )
                summary["order_updates"] += 1

        logger.info(
            "reconcile complete: imported=%d  closed=%d  updated=%d  order_updates=%d",
            summary["imported"],
            summary["closed"],
            summary["updated"],
            summary["order_updates"],
        )
        return summary
