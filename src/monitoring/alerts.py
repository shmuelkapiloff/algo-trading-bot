"""
Alert dispatcher — Telegram + log fallback.

Supports two channels:
  1. Telegram Bot API (async, via httpx)
  2. Log fallback when Telegram is not configured

All send operations are fire-and-forget: failures are logged but never
propagate to the caller. The trading engine must not crash on alert failure.

Configuration (via environment variables or config.yaml):
  TELEGRAM_BOT_TOKEN : BotFather token
  TELEGRAM_CHAT_ID   : Numeric chat / channel ID

Usage
-----
    alerts = AlertDispatcher(settings)
    await alerts.send("Order filled: AAPL x100 @ $175.23")
    await alerts.send_fill("AAPL", 100, 175.23, "momentum")
    await alerts.send_halt("Redis connection lost")
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class AlertLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertDispatcher:
    """
    Send operational alerts via Telegram (with log fallback).

    Parameters
    ----------
    bot_token : Telegram bot token (None → log-only mode)
    chat_id   : Telegram chat/channel ID
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)

        if not self._enabled:
            logger.warning("AlertDispatcher: Telegram not configured — log-only mode")

    # ------------------------------------------------------------------
    # Generic send
    # ------------------------------------------------------------------

    async def send(
        self,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
    ) -> None:
        """Send an alert message. Never raises."""
        prefix_map = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.CRITICAL: "🚨",
        }
        formatted = f"{prefix_map[level]} [AlgoTrader] {message}"

        logger.log(level.value, "ALERT: %s", message)

        if self._enabled:
            await self._send_telegram(formatted)

    # ------------------------------------------------------------------
    # Semantic convenience helpers
    # ------------------------------------------------------------------

    async def send_fill(
        self,
        symbol: str,
        qty: int,
        price: float,
        strategy: str,
        pnl: Optional[float] = None,
    ) -> None:
        pnl_str = f"  |  P&L: ${pnl:+.2f}" if pnl is not None else ""
        await self.send(
            f"FILL: {symbol} x{qty} @ ${price:.2f}  ({strategy}){pnl_str}",
            AlertLevel.INFO,
        )

    async def send_reject(self, symbol: str, reason: str) -> None:
        await self.send(
            f"REJECTED: {symbol} — {reason}",
            AlertLevel.WARNING,
        )

    async def send_halt(self, reason: str) -> None:
        await self.send(
            f"TRADING HALTED — {reason}",
            AlertLevel.CRITICAL,
        )

    async def send_regime_change(self, old_regime: str, new_regime: str) -> None:
        await self.send(
            f"Regime change: {old_regime} → {new_regime}",
            AlertLevel.INFO,
        )

    async def send_daily_summary(
        self,
        realized_pnl: float,
        trades: int,
        win_rate: float,
        equity: float,
    ) -> None:
        await self.send(
            f"📊 Daily Summary | "
            f"P&L: ${realized_pnl:+.2f} | "
            f"Trades: {trades} | "
            f"Win rate: {win_rate:.0%} | "
            f"Equity: ${equity:,.0f}",
            AlertLevel.INFO,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _send_telegram(self, text: str) -> None:
        url = _TELEGRAM_API.format(token=self._token)
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    logger.warning(
                        "Telegram alert failed: HTTP %d — %s",
                        resp.status_code,
                        resp.text[:200],
                    )
        except Exception as exc:
            logger.warning("Telegram send error (non-fatal): %s", exc)
