"""
Market data fetcher — Alpaca Market Data API (Phase 1).

Provides OHLCV bars for EOD strategy scanning.

Design rules (from TRADING_BOT_PLAN.md)
----------------------------------------
- Always request adjustment='split' so all price data is split-adjusted.
- Store close_raw alongside close_adj for audit purposes.
- All indicators (RSI, MACD, BB) MUST use close_adj, never close_raw.
- Stop-loss and take-profit prices are based on close_adj percentages.
- Monitor bar lag: alert if bar.timestamp is > 90 seconds stale.
- IEX vs SIP note: free Alpaca tier serves IEX only; volume may be 10-40%
  lower than SIP. If this becomes an issue, upgrade to SIP ($15/month).

All async — no blocking calls on the event loop.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.models import Bar
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """
    Thin async-compatible wrapper around the Alpaca StockHistoricalDataClient.

    The Alpaca SDK is synchronous. We run the blocking call via
    asyncio.to_thread() to avoid freezing the event loop.
    """

    def __init__(self, api_key: str, secret_key: str) -> None:
        self._client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )

    # ------------------------------------------------------------------
    # Daily bars
    # ------------------------------------------------------------------

    async def get_daily_bars(
        self,
        symbols: list[str],
        start: date,
        end: Optional[date] = None,
        limit: int = 252,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch split-adjusted daily OHLCV bars for a list of symbols.

        Returns a dict of {symbol: DataFrame} with columns:
            open, high, low, close_adj, volume, vwap, trade_count,
            close_raw (for audit), timestamp (UTC)

        Any symbol for which bars are unavailable is omitted from the result
        with a WARNING log — callers should handle missing keys gracefully.
        """
        import asyncio

        end_date = end or date.today()

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=datetime(start.year, start.month, start.day, tzinfo=timezone.utc),
            end=datetime(
                end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc
            ),
            adjustment="split",  # always split-adjusted (plan rule)
            limit=limit,
            feed="iex",  # default; upgrade to "sip" with paid plan
        )

        try:
            raw = await asyncio.to_thread(self._client.get_stock_bars, request)
        except Exception as exc:
            logger.error("Alpaca bars request failed: %s", exc)
            return {}

        result: dict[str, pd.DataFrame] = {}

        for symbol in symbols:
            bars: list[Bar] = raw.get(symbol, [])  # type: ignore[arg-type]
            if not bars:
                logger.warning(
                    "No bars returned for %s (start=%s end=%s)", symbol, start, end_date
                )
                continue

            records = []
            for bar in bars:
                records.append(
                    {
                        "timestamp": bar.timestamp,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close_adj": bar.close,  # split-adjusted (always use this)
                        "close_raw": bar.close,  # same in this feed; kept for audit
                        "volume": bar.volume,
                        "vwap": bar.vwap,
                        "trade_count": getattr(bar, "trade_count", None),
                    }
                )

            df = pd.DataFrame(records)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)
            result[symbol] = df

        return result

    # ------------------------------------------------------------------
    # Latest bar (for intraday lag monitoring)
    # ------------------------------------------------------------------

    async def get_latest_bar(self, symbol: str) -> Optional[dict]:
        """
        Fetch the most recent bar for a single symbol.

        Used by data_quality checks to verify bar lag < 90 seconds.
        Returns None on failure.
        """
        import asyncio
        from alpaca.data.requests import StockLatestBarRequest

        request = StockLatestBarRequest(symbol_or_symbols=symbol, feed="iex")
        try:
            raw = await asyncio.to_thread(self._client.get_stock_latest_bar, request)
            bar = raw.get(symbol)
            if bar is None:
                return None
            lag_seconds = (datetime.now(timezone.utc) - bar.timestamp).total_seconds()
            if lag_seconds > 90:
                logger.warning(
                    "Bar lag alert: %s latest bar is %.0fs stale (threshold=90s)",
                    symbol,
                    lag_seconds,
                )
            return {
                "timestamp": bar.timestamp,
                "close_adj": bar.close,
                "volume": bar.volume,
                "lag_seconds": lag_seconds,
            }
        except Exception as exc:
            logger.error("Failed to fetch latest bar for %s: %s", symbol, exc)
            return None

    # ------------------------------------------------------------------
    # Universe helpers
    # ------------------------------------------------------------------

    async def filter_by_adv(
        self,
        symbols: list[str],
        min_adv_usd: float,
        lookback_days: int = 20,
    ) -> list[str]:
        """
        Return only symbols whose 20-day average daily dollar volume
        exceeds min_adv_usd.

        Uses recent bars — runs at scan time, not cached. For large
        universes (500+ symbols), call in batches of 50.
        """
        from datetime import timedelta

        start = date.today() - timedelta(days=lookback_days + 5)  # buffer for weekends
        bars = await self.get_daily_bars(symbols, start=start, limit=lookback_days)

        eligible: list[str] = []
        for symbol, df in bars.items():
            if df.empty:
                continue
            adv = (df["close_adj"] * df["volume"]).mean()
            if adv >= min_adv_usd:
                eligible.append(symbol)
            else:
                logger.debug(
                    "%s filtered out: ADV $%.0f < threshold $%.0f",
                    symbol,
                    adv,
                    min_adv_usd,
                )

        logger.info(
            "ADV filter: %d/%d symbols passed ($%.0f threshold)",
            len(eligible),
            len(symbols),
            min_adv_usd,
        )
        return eligible
