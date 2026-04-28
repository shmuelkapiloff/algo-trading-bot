"""Stock Universe Management — Three-tier filter.

Tier 1 — Static Filter (updated quarterly):
    S&P 500 constituents list (from config watchlist + exclusions).
    Market cap > $2B, average volume > 500K/day, spread <= 15 bps.
    Leveraged ETF blacklist applied (from config.universe_exclusions).

Tier 2 — Daily Filter (run every market day at 09:00 ET):
    Removes: earnings within 3 days, ex-dividend tomorrow, halted,
    corporate action active, abnormal spread, no short borrow.

Tier 3 — Strategy Scan (run at 09:15 ET after market open):
    Each strategy returns its candidate list; Portfolio Manager applies
    correlation + sector exposure constraints before final ranking.

Usage
-----
    universe = StockUniverse(config)
    tier1 = universe.get_static_universe()
    tier2 = await universe.apply_daily_filters(tier1)
    tier3 = universe.apply_strategy_scan(tier2, strategy_name="momentum", bars=bars)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, FrozenSet, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default leveraged ETF blacklist (from config.universe_exclusions in plan)
# ---------------------------------------------------------------------------

_LEVERAGED_ETF_BLACKLIST: FrozenSet[str] = frozenset(
    [
        # ProShares 2×/3× leveraged
        "SSO", "SDS", "UPRO", "SPXU", "QLD", "QID", "TQQQ", "SQQQ",
        "UDOW", "SDOW", "URTY", "SRTY", "UVXY", "SVXY",
        # Direxion 3× leveraged
        "FAS", "FAZ", "TNA", "TZA", "LABU", "LABD", "NUGT", "DUST",
        "JNUG", "JDST", "TECL", "TECS", "SOXL", "SOXS",
        # Velocity Shares
        "TVIX", "ZIV",
        # MicroStrategy / volatility products
        "MSTU", "MSTX",
    ]
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class UniverseSymbol:
    symbol: str
    market_cap_b: float = 0.0        # Market cap in billions
    avg_volume_30d: float = 0.0      # Average 30-day volume (shares)
    avg_spread_bps: float = 0.0      # Average bid/ask spread in bps
    sector: str = ""
    in_sp500: bool = False
    excluded_reason: Optional[str] = None  # set when filtered out

    @property
    def is_active(self) -> bool:
        return self.excluded_reason is None


@dataclass
class UniverseSnapshot:
    as_of_date: date
    symbols: List[UniverseSymbol] = field(default_factory=list)

    @property
    def active_symbols(self) -> List[str]:
        return [s.symbol for s in self.symbols if s.is_active]

    @property
    def excluded_symbols(self) -> List[str]:
        return [s.symbol for s in self.symbols if not s.is_active]

    def __len__(self) -> int:
        return len(self.active_symbols)


# ---------------------------------------------------------------------------
# Universe Manager
# ---------------------------------------------------------------------------


class StockUniverse:
    """Three-tier stock universe manager.

    Parameters
    ----------
    watchlist:
        Base list of symbols (e.g. loaded from config/watchlist.txt).
    leveraged_etf_blacklist:
        Additional symbols to exclude (added to built-in blacklist).
    min_market_cap_b:
        Minimum market cap in billions (default 2.0).
    min_avg_volume:
        Minimum 30-day average daily volume in shares (default 500_000).
    max_spread_bps:
        Maximum allowable average spread in bps (default 15.0).
    earnings_buffer_days:
        Remove symbol if earnings within this many days (default 3).
    """

    def __init__(
        self,
        watchlist: Optional[List[str]] = None,
        leveraged_etf_blacklist: Optional[Set[str]] = None,
        min_market_cap_b: float = 2.0,
        min_avg_volume: float = 500_000.0,
        max_spread_bps: float = 15.0,
        earnings_buffer_days: int = 3,
    ) -> None:
        self.watchlist: List[str] = watchlist or []
        self._blacklist: FrozenSet[str] = _LEVERAGED_ETF_BLACKLIST | frozenset(
            leveraged_etf_blacklist or set()
        )
        self.min_market_cap_b = min_market_cap_b
        self.min_avg_volume = min_avg_volume
        self.max_spread_bps = max_spread_bps
        self.earnings_buffer_days = earnings_buffer_days

    # ------------------------------------------------------------------
    # Tier 1 — Static filter
    # ------------------------------------------------------------------

    def get_static_universe(
        self,
        as_of: Optional[date] = None,
        fundamentals: Optional[Dict[str, dict]] = None,
    ) -> UniverseSnapshot:
        """Apply Tier 1 static filters.

        Parameters
        ----------
        as_of:
            Reference date for the snapshot (defaults to today).
        fundamentals:
            Optional dict mapping symbol → {market_cap_b, avg_volume,
            avg_spread_bps, sector, in_sp500}.  If provided, market-cap
            and liquidity filters are applied.  If None, all watchlist
            symbols pass through (paper-trading convenience).
        """
        as_of = as_of or date.today()
        result: List[UniverseSymbol] = []

        for symbol in self.watchlist:
            sym = UniverseSymbol(symbol=symbol)

            # Leveraged ETF blacklist
            if symbol in self._blacklist:
                sym.excluded_reason = "leveraged_etf_blacklist"
                result.append(sym)
                continue

            if fundamentals:
                data = fundamentals.get(symbol, {})
                sym.market_cap_b = data.get("market_cap_b", 0.0)
                sym.avg_volume_30d = data.get("avg_volume", 0.0)
                sym.avg_spread_bps = data.get("avg_spread_bps", 0.0)
                sym.sector = data.get("sector", "")
                sym.in_sp500 = data.get("in_sp500", False)

                if sym.market_cap_b < self.min_market_cap_b:
                    sym.excluded_reason = f"market_cap_too_small ({sym.market_cap_b:.1f}B)"
                    result.append(sym)
                    continue

                if sym.avg_volume_30d < self.min_avg_volume:
                    sym.excluded_reason = f"avg_volume_too_low ({sym.avg_volume_30d:,.0f})"
                    result.append(sym)
                    continue

                if sym.avg_spread_bps > self.max_spread_bps:
                    sym.excluded_reason = f"spread_too_wide ({sym.avg_spread_bps:.1f} bps)"
                    result.append(sym)
                    continue

            result.append(sym)

        snapshot = UniverseSnapshot(as_of_date=as_of, symbols=result)
        active = len(snapshot.active_symbols)
        excluded = len(snapshot.excluded_symbols)
        logger.info(
            "[universe] Tier-1: %d active, %d excluded (as_of=%s)",
            active,
            excluded,
            as_of,
        )
        return snapshot

    # ------------------------------------------------------------------
    # Tier 2 — Daily filter
    # ------------------------------------------------------------------

    def apply_daily_filters(
        self,
        snapshot: UniverseSnapshot,
        as_of: Optional[date] = None,
        earnings_calendar: Optional[Dict[str, date]] = None,
        ex_div_dates: Optional[Dict[str, date]] = None,
        halted_symbols: Optional[Set[str]] = None,
        corp_action_symbols: Optional[Set[str]] = None,
        high_spread_symbols: Optional[Set[str]] = None,
        no_borrow_symbols: Optional[Set[str]] = None,
    ) -> UniverseSnapshot:
        """Apply Tier 2 daily filters.

        All parameters are optional; omitting a filter disables that check.

        Parameters
        ----------
        snapshot:         Output of ``get_static_universe()``.
        as_of:            Reference date (defaults to today).
        earnings_calendar: symbol → next earnings date.
        ex_div_dates:     symbol → ex-dividend date.
        halted_symbols:   Set of currently halted symbols.
        corp_action_symbols: Set of symbols with active corporate actions.
        high_spread_symbols: Set of symbols with abnormal spread today.
        no_borrow_symbols: Set of symbols unavailable to short.
        """
        as_of = as_of or date.today()
        removed: List[str] = []

        for sym in snapshot.symbols:
            if not sym.is_active:
                continue

            # Earnings within N days
            if earnings_calendar:
                earnings_date = earnings_calendar.get(sym.symbol)
                if earnings_date and (earnings_date - as_of).days <= self.earnings_buffer_days:
                    sym.excluded_reason = f"earnings_soon ({earnings_date})"
                    removed.append(sym.symbol)
                    continue

            # Ex-dividend tomorrow
            if ex_div_dates:
                exdiv = ex_div_dates.get(sym.symbol)
                if exdiv and exdiv == as_of + timedelta(days=1):
                    sym.excluded_reason = f"ex_div_tomorrow ({exdiv})"
                    removed.append(sym.symbol)
                    continue

            # Halted
            if halted_symbols and sym.symbol in halted_symbols:
                sym.excluded_reason = "halted"
                removed.append(sym.symbol)
                continue

            # Active corporate action
            if corp_action_symbols and sym.symbol in corp_action_symbols:
                sym.excluded_reason = "corp_action_active"
                removed.append(sym.symbol)
                continue

            # Abnormal spread
            if high_spread_symbols and sym.symbol in high_spread_symbols:
                sym.excluded_reason = "spread_abnormal"
                removed.append(sym.symbol)
                continue

            # No short borrow
            if no_borrow_symbols and sym.symbol in no_borrow_symbols:
                sym.excluded_reason = "no_short_borrow"
                removed.append(sym.symbol)
                continue

        if removed:
            logger.info(
                "[universe] Tier-2 daily filter removed %d symbols: %s",
                len(removed),
                removed[:10],
            )

        return snapshot

    # ------------------------------------------------------------------
    # Tier 3 — Strategy scan
    # ------------------------------------------------------------------

    def apply_strategy_scan(
        self,
        snapshot: UniverseSnapshot,
        strategy_name: str,
        bars: Optional[Dict[str, pd.DataFrame]] = None,
        scan_fn=None,
    ) -> List[str]:
        """Apply Tier 3 strategy-specific candidate scan.

        Parameters
        ----------
        snapshot:      Output of ``apply_daily_filters()``.
        strategy_name: Strategy identifier (for logging only).
        bars:          Full bar history (passed to ``scan_fn`` if provided).
        scan_fn:       Optional callable(symbol, df) → bool that returns True
                       if the symbol is a candidate for this strategy.
                       If None, all active symbols are returned.

        Returns
        -------
        List of candidate symbols for this strategy.
        """
        active = snapshot.active_symbols
        if scan_fn is None or bars is None:
            logger.debug(
                "[universe] Tier-3 (%s): no scan_fn, returning all %d active",
                strategy_name,
                len(active),
            )
            return active

        candidates = []
        for symbol in active:
            df = bars.get(symbol)
            if df is None or df.empty:
                continue
            try:
                if scan_fn(symbol, df):
                    candidates.append(symbol)
            except Exception as exc:
                logger.debug("[universe] scan error %s: %s", symbol, exc)

        logger.info(
            "[universe] Tier-3 (%s): %d candidates from %d active",
            strategy_name,
            len(candidates),
            len(active),
        )
        return candidates
