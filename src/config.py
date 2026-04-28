"""
AlgoTrader Pro configuration — loaded once at startup.

Reads from environment variables first (12-factor), then falls back
to .env file, then to config.yaml defaults.

Usage
-----
    from src.config import settings
    max_risk = settings.risk.max_risk_per_trade
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent  # trading_bot/
_CONFIG_YAML = _ROOT / "config" / "config.yaml"


def _load_yaml() -> dict[str, Any]:
    if _CONFIG_YAML.exists():
        with _CONFIG_YAML.open() as f:
            return yaml.safe_load(f) or {}
    return {}


# ---------------------------------------------------------------------------
# Nested config sections
# ---------------------------------------------------------------------------


class CapitalConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CAPITAL_")
    initial_portfolio_value: float = 10_000.0


class RiskConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RISK_")
    max_risk_per_trade: float = 0.01
    absolute_max_position_pct: float = 0.03
    max_global_open_risk: float = 0.02
    stop_loss_floor_pct: float = 0.03
    max_positions: int = 10


class PdtConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PDT_")
    enabled: bool = True
    max_day_trades_per_5_days: int = 3
    portfolio_value_threshold: float = 25_000.0


class ExecutionConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EXEC_")
    order_type: str = "market"
    time_in_force: str = "day"
    retry_attempts: int = 3
    retry_backoff_seconds: float = 2.0


class DataQualityConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DQ_")
    max_bar_lag_seconds: int = 90
    volume_divergence_threshold: float = 0.30
    min_bars_required: int = 60


class ShutdownConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SHUTDOWN_")
    signal_queue_drain_timeout_seconds: float = 10.0
    oms_flush_timeout_seconds: float = 5.0


# ---------------------------------------------------------------------------
# Top-level settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    All runtime settings.  Env vars take precedence over .env, which takes
    precedence over defaults. YAML values are injected as defaults when the
    section is not overridden by env.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Infra ────────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///trading.db"
    redis_url: str = "redis://localhost:6379"
    fencing_secret: str = ""  # hex-encoded 32 bytes; empty = dev ephemeral

    # ── Alpaca ───────────────────────────────────────────────────────────────
    alpaca_api_key: str = ""
    alpaca_secret: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_data_url: str = "https://data.alpaca.markets"

    # ── Alerts ───────────────────────────────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Nested (read from YAML then overrideable by env) ─────────────────────
    capital: CapitalConfig = Field(default_factory=CapitalConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    pdt: PdtConfig = Field(default_factory=PdtConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    shutdown: ShutdownConfig = Field(default_factory=ShutdownConfig)

    def model_post_init(self, __context: Any) -> None:
        """Override nested defaults from config.yaml."""
        cfg = _load_yaml()

        def _merge(section: BaseSettings, key: str) -> None:
            data = cfg.get(key, {})
            for field_name, val in data.items():
                if hasattr(section, field_name):
                    object.__setattr__(section, field_name, val)

        _merge(self.capital, "capital")
        _merge(self.risk, "risk")
        _merge(self.pdt, "pdt")
        _merge(self.execution, "execution")
        _merge(self.data_quality, "data_quality")
        _merge(self.shutdown, "shutdown")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton settings instance. Cached after first call."""
    return Settings()


# Convenience alias
settings: Settings = get_settings()
