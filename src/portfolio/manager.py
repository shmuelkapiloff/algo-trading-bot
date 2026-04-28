"""
Portfolio Manager — tracks open positions, equity, and PDT state.

Responsibilities
----------------
  1. Maintain an in-memory cache of open positions (reconciled at startup).
  2. Provide portfolio_state dict for PreTradeGateway risk checks.
  3. Enforce max_positions limit.
  4. Track PDT day-trade counter in Redis (atomic increment).
  5. Size new orders via FixedFractionalSizer.
  6. Record fills and cancellations.

Redis key conventions
---------------------
  algotrader:pdt_daytrades        → integer count (resets EOD or weekly)
  algotrader:positions:{symbol}   → JSON position record
  algotrader:equity               → current portfolio equity (USD)

Note: This is a Phase 1 in-memory + Redis implementation. Phase 2 will
migrate positions to the DB and use Redis Streams for event-driven updates.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis

from ..config import Settings
from ..signals.models import OrderSide, SignalIntent
from .risk import calculate_position_size

logger = logging.getLogger(__name__)

_POS_PREFIX = "algotrader:positions:"
_EQUITY_KEY = "algotrader:equity"
_PDT_KEY = "algotrader:pdt_daytrades"
_PDT_LIMIT = 3  # PDT rule: 3 day trades in rolling 5 business days


@dataclass
class Position:
    symbol: str
    side: str
    qty: int
    avg_entry_price: float
    stop_distance_pct: float
    strategy_name: str
    opened_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    order_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "avg_entry_price": self.avg_entry_price,
            "stop_distance_pct": self.stop_distance_pct,
            "strategy_name": self.strategy_name,
            "opened_at": self.opened_at,
            "order_id": self.order_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Position":
        return cls(**d)


class PortfolioManager:
    """
    In-memory portfolio state backed by Redis.

    Call `await load_state()` once at startup to hydrate from Redis.
    """

    def __init__(self, redis: aioredis.Redis, settings: Settings) -> None:
        self._redis = redis
        self._settings = settings
        self._positions: dict[str, Position] = {}
        self._equity: float = 0.0

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def load_state(self) -> None:
        """Hydrate in-memory state from Redis (call once at startup)."""
        # Load equity
        raw = await self._redis.get(_EQUITY_KEY)
        if raw:
            try:
                self._equity = float(raw)
            except (ValueError, TypeError):
                logger.warning("Could not parse equity from Redis: %r", raw)

        # Load open positions
        keys = await self._redis.keys(f"{_POS_PREFIX}*")
        for key in keys:
            data = await self._redis.get(key)
            if data:
                try:
                    pos = Position.from_dict(json.loads(data))
                    self._positions[pos.symbol] = pos
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    logger.warning("Corrupt position data for key %s: %s", key, e)

        logger.info(
            "Portfolio state loaded: equity=%.2f  open_positions=%d",
            self._equity,
            len(self._positions),
        )

    # ------------------------------------------------------------------
    # Public read interface
    # ------------------------------------------------------------------

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def open_positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def get_open_risk(self) -> float:
        """Return total open risk as fraction of equity (for risk gate)."""
        if self._equity <= 0:
            return 0.0
        total_risk = sum(
            (pos.avg_entry_price * pos.qty * pos.stop_distance_pct)
            for pos in self._positions.values()
        )
        return total_risk / self._equity

    def build_portfolio_state(self) -> dict[str, Any]:
        """Build the portfolio_state dict expected by PreTradeGateway."""
        return {
            "equity": self._equity,
            "open_positions": list(self._positions.keys()),
            "open_risk_fraction": self.get_open_risk(),
            "pdt_daytrade_count": 0,  # updated by can_open_position()
        }

    # ------------------------------------------------------------------
    # Pre-trade checks
    # ------------------------------------------------------------------

    async def can_open_position(self, signal: SignalIntent) -> tuple[bool, str]:
        """
        Fast pre-checks before sending to PreTradeGateway.
        Returns (allowed, reason).
        """
        # Already have a position in this symbol
        if signal.symbol in self._positions:
            return False, f"Already holding {signal.symbol}"

        # Max position count
        max_pos = self._settings.risk.max_positions
        if len(self._positions) >= max_pos:
            return False, f"Max positions reached ({max_pos})"

        # PDT check
        pdt_count = await self._get_pdt_count()
        if pdt_count >= _PDT_LIMIT and self._settings.pdt.pdt_protection_enabled:
            return False, f"PDT limit reached ({pdt_count}/{_PDT_LIMIT})"

        return True, "ok"

    async def size_signal(self, signal: SignalIntent, last_price: float) -> int:
        """Return the number of shares to buy (0 = skip this signal)."""
        cfg = self._settings.risk
        return calculate_position_size(
            signal=signal,
            portfolio_value=self._equity,
            current_open_risk=self.get_open_risk(),
            max_risk_per_trade=cfg.max_risk_per_trade,
            absolute_max_position_pct=cfg.absolute_max_position_pct,
            max_global_open_risk=cfg.max_global_open_risk,
            stop_loss_floor_pct=cfg.stop_loss_floor_pct,
            last_price=last_price,
        )

    # ------------------------------------------------------------------
    # State mutations
    # ------------------------------------------------------------------

    async def record_fill(
        self,
        order_id: str,
        symbol: str,
        side: OrderSide,
        filled_qty: int,
        fill_price: float,
        strategy_name: str,
        stop_distance_pct: float,
    ) -> None:
        """Record a confirmed fill and persist the position to Redis."""
        if side == OrderSide.BUY:
            if symbol in self._positions:
                # Average up / down (partial fill scenario)
                pos = self._positions[symbol]
                total_qty = pos.qty + filled_qty
                avg_price = (
                    pos.avg_entry_price * pos.qty + fill_price * filled_qty
                ) / total_qty
                pos.qty = total_qty
                pos.avg_entry_price = avg_price
            else:
                pos = Position(
                    symbol=symbol,
                    side=side.value,
                    qty=filled_qty,
                    avg_entry_price=fill_price,
                    stop_distance_pct=stop_distance_pct,
                    strategy_name=strategy_name,
                    order_id=order_id,
                )
                self._positions[symbol] = pos

            await self._persist_position(pos)

        elif side == OrderSide.SELL:
            pos = self._positions.pop(symbol, None)
            if pos:
                await self._redis.delete(f"{_POS_PREFIX}{symbol}")
            else:
                logger.warning("record_fill: SELL for %s but no open position", symbol)

        logger.info(
            "Fill recorded: %s %s %d@%.2f (strategy=%s)",
            side.value,
            symbol,
            filled_qty,
            fill_price,
            strategy_name,
        )

    async def update_equity(self, equity: float) -> None:
        """Update the portfolio equity value (called from Alpaca account poll)."""
        self._equity = equity
        await self._redis.set(_EQUITY_KEY, str(equity))
        logger.debug("Equity updated: %.2f", equity)

    async def increment_pdt_counter(self) -> int:
        """Atomically increment PDT day-trade counter. Returns new count."""
        count = await self._redis.incr(_PDT_KEY)
        logger.debug("PDT counter incremented: %d", count)
        return count

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def import_position(
        self,
        symbol: str,
        qty: int,
        avg_price: float,
        strategy_name: str = "reconcile",
        stop_distance_pct: float = 0.03,
    ) -> None:
        """Import a broker-discovered position not present in Redis."""
        pos = Position(
            symbol=symbol,
            side="buy",
            qty=qty,
            avg_entry_price=avg_price,
            stop_distance_pct=stop_distance_pct,
            strategy_name=strategy_name,
        )
        self._positions[symbol] = pos
        await self._persist_position(pos)
        logger.info(
            "Position imported from broker: %s  qty=%d  avg=%.2f",
            symbol,
            qty,
            avg_price,
        )

    async def remove_position(self, symbol: str) -> None:
        """Remove a position that is no longer open at the broker."""
        self._positions.pop(symbol, None)
        await self._redis.delete(f"{_POS_PREFIX}{symbol}")
        logger.info("Position removed (broker-reconcile): %s", symbol)

    async def update_position_qty(self, symbol: str, qty: int) -> None:
        """Update position qty after a qty mismatch discovered during reconciliation."""
        if symbol in self._positions:
            self._positions[symbol].qty = qty
            await self._persist_position(self._positions[symbol])
            logger.info("Position qty updated (reconcile): %s  qty=%d", symbol, qty)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _persist_position(self, pos: Position) -> None:
        await self._redis.set(
            f"{_POS_PREFIX}{pos.symbol}",
            json.dumps(pos.to_dict()),
        )

    async def _get_pdt_count(self) -> int:
        raw = await self._redis.get(_PDT_KEY)
        if raw is None:
            return 0
        try:
            return int(raw)
        except (ValueError, TypeError):
            return 0
