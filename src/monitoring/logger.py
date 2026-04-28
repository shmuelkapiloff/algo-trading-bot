"""
Centralised logging configuration using loguru.

Call `configure_logging(level, log_file)` once at startup (in main.py).
All other modules use standard `logging.getLogger(__name__)` — loguru
intercepts them via the InterceptHandler below.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from loguru import logger


class InterceptHandler(logging.Handler):
    """
    Route stdlib logging calls through loguru.

    Install with:
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Find the corresponding Loguru level
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk the call stack to find the original caller
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def configure_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """
    Set up loguru as the global log handler.

    Parameters
    ----------
    level    : Log level string (DEBUG / INFO / WARNING / ERROR)
    log_file : Optional path to a rotating log file (e.g. "logs/trading.log")
    """
    # Remove default loguru sink
    logger.remove()

    # Console (stdout)
    logger.add(
        sys.stdout,
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File (rotating at 100 MB, retained for 30 days)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=level.upper(),
            rotation="100 MB",
            retention="30 days",
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} — {message}",
            backtrace=True,
            diagnose=False,  # no source vars in file (PII risk)
        )

    # Intercept all stdlib logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "httpx", "httpcore", "apscheduler", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info("Logging configured (level=%s  file=%s)", level, log_file or "none")
