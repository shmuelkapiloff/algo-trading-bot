"""
Telegram Bot — emergency control commands via Telegram.

Supported commands:
  /halt        — Emergency halt (sets TradingState.EMERGENCY_HALT)
  /pause       — Pause new entries
  /resume      — Resume normal trading
  /status      — Returns current TradingState + open position count
  /close_only  — Switch to close-only mode

Security:
  - Only messages from TELEGRAM_ALLOWED_CHAT_IDS are accepted
  - All commands are validated and forwarded to the Control API
    (localhost:8000) with Bearer token authentication
  - No direct state mutation from the Telegram handler — always proxied
    through control/api.py to ensure audit logging

Configuration (environment variables):
  TELEGRAM_BOT_TOKEN      — Bot token from @BotFather
  TELEGRAM_ALLOWED_CHAT_IDS — Comma-separated list of authorized chat IDs
  CONTROL_API_URL         — Default: http://localhost:8000
  CONTROL_API_TOKEN       — Bearer token for Control API
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes
    _TELEGRAM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TELEGRAM_AVAILABLE = False
    Update = None  # type: ignore
    ContextTypes = None  # type: ignore

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HTTPX_AVAILABLE = False


def _allowed_chat_ids() -> set[int]:
    raw = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "")
    ids: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit() or (part.startswith("-") and part[1:].isdigit()):
            ids.add(int(part))
    return ids


def _control_url() -> str:
    return os.getenv("CONTROL_API_URL", "http://localhost:8000")


def _control_token() -> str:
    return os.getenv("CONTROL_API_TOKEN", "")


class TelegramCommandBot:
    """
    Telegram bot that proxies emergency commands to the Control API.
    Requires python-telegram-bot >= 20 and httpx.
    """

    def __init__(self) -> None:
        if not _TELEGRAM_AVAILABLE:
            logger.warning(
                "telegram_bot: python-telegram-bot not installed. "
                "Install with: pip install python-telegram-bot"
            )
        if not _HTTPX_AVAILABLE:
            logger.warning(
                "telegram_bot: httpx not installed. "
                "Install with: pip install httpx"
            )
        self._token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._allowed = _allowed_chat_ids()
        self._app: Optional[object] = None

    # ------------------------------------------------------------------
    # Authorization
    # ------------------------------------------------------------------

    def _is_authorized(self, chat_id: int) -> bool:
        return chat_id in self._allowed

    # ------------------------------------------------------------------
    # Control API proxy
    # ------------------------------------------------------------------

    async def _call_api(self, endpoint: str, payload: dict | None = None) -> str:
        if not _HTTPX_AVAILABLE:
            return "Error: httpx not installed"
        headers = {"Authorization": f"Bearer {_control_token()}"}
        url = f"{_control_url()}{endpoint}"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(url, json=payload or {}, headers=headers)
                if resp.status_code == 200:
                    return resp.json().get("status", "ok")
                return f"Error {resp.status_code}: {resp.text[:100]}"
        except Exception as exc:  # noqa: BLE001
            return f"Connection error: {exc}"

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _cmd_halt(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:  # type: ignore[name-defined]
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text("Unauthorized.")
            return
        result = await self._call_api("/halt", {"reason": "telegram_command"})
        await update.message.reply_text(f"HALT: {result}")
        logger.warning("telegram_bot.halt chat_id=%d result=%s", chat_id, result)

    async def _cmd_pause(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:  # type: ignore[name-defined]
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text("Unauthorized.")
            return
        result = await self._call_api("/pause", {"reason": "telegram_command"})
        await update.message.reply_text(f"PAUSE: {result}")

    async def _cmd_resume(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:  # type: ignore[name-defined]
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text("Unauthorized.")
            return
        result = await self._call_api("/resume", {"reason": "telegram_command"})
        await update.message.reply_text(f"RESUME: {result}")

    async def _cmd_close_only(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:  # type: ignore[name-defined]
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text("Unauthorized.")
            return
        result = await self._call_api("/close_only", {"reason": "telegram_command"})
        await update.message.reply_text(f"CLOSE_ONLY: {result}")

    async def _cmd_status(self, update: "Update", context: "ContextTypes.DEFAULT_TYPE") -> None:  # type: ignore[name-defined]
        chat_id = update.effective_chat.id
        if not self._is_authorized(chat_id):
            await update.message.reply_text("Unauthorized.")
            return
        result = await self._call_api("/status")
        await update.message.reply_text(f"STATUS: {result}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def build_app(self) -> Optional[object]:
        """Build the telegram Application. Returns None if library unavailable."""
        if not _TELEGRAM_AVAILABLE or not self._token:
            logger.warning("telegram_bot.build_app: token missing or library not installed")
            return None
        app = Application.builder().token(self._token).build()
        app.add_handler(CommandHandler("halt", self._cmd_halt))
        app.add_handler(CommandHandler("pause", self._cmd_pause))
        app.add_handler(CommandHandler("resume", self._cmd_resume))
        app.add_handler(CommandHandler("close_only", self._cmd_close_only))
        app.add_handler(CommandHandler("status", self._cmd_status))
        self._app = app
        return app

    async def run_polling(self) -> None:
        """Start polling loop. Blocks until stopped."""
        app = self.build_app()
        if app is None:
            logger.error("telegram_bot: cannot start — missing token or library")
            return
        logger.info("telegram_bot.start polling authorized_chats=%s", self._allowed)
        await app.run_polling()
