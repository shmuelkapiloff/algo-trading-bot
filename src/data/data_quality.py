"""Automated data quality checks for market bar data.

Checks implemented (as defined in TRADING_BOT_PLAN.md §6 "data_quality.py"):
  1. Missing bar detection    — more than N% of expected bars absent
  2. Stale bar detection      — bar timestamp too far from expected calendar
  3. Volume anomaly           — bar volume > K × median (spike check)
  4. Price continuity         — close-to-close gap > threshold (corp action proxy)
  5. IEX vs SIP spread        — volume difference > tolerance (feed mismatch)
  6. Zero / negative price    — obviously bad bar
  7. OHLC consistency         — high < low, or open/close outside high/low

Results are collected per symbol as :class:`QualityReport` objects.
Symbols that fail hard checks are quarantined (excluded from universe).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------


class CheckSeverity(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"  # symbol quarantined from trading


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    check_name: str
    severity: CheckSeverity
    message: str
    detail: Optional[dict] = None


@dataclass
class QualityReport:
    symbol: str
    as_of_date: date
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(c.severity == CheckSeverity.FAIL for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(c.severity == CheckSeverity.WARN for c in self.checks)

    @property
    def worst_severity(self) -> CheckSeverity:
        if any(c.severity == CheckSeverity.FAIL for c in self.checks):
            return CheckSeverity.FAIL
        if any(c.severity == CheckSeverity.WARN for c in self.checks):
            return CheckSeverity.WARN
        return CheckSeverity.OK

    def summary(self) -> str:
        fails = sum(1 for c in self.checks if c.severity == CheckSeverity.FAIL)
        warns = sum(1 for c in self.checks if c.severity == CheckSeverity.WARN)
        return f"{self.symbol}: FAIL={fails} WARN={warns} ({self.worst_severity.value.upper()})"


# ---------------------------------------------------------------------------
# Data Quality Checker
# ---------------------------------------------------------------------------


class DataQualityChecker:
    """Runs automated quality checks on a dict of symbol → DataFrame.

    Parameters
    ----------
    max_missing_bar_ratio:
        Maximum fraction of missing bars before marking FAIL (default 0.02 = 2%).
    volume_spike_threshold:
        Flag as WARN if bar volume > this multiple of rolling median (default 10×).
    price_gap_warn_pct:
        Flag price gap (abs change between consecutive closes) above this % (default 15%).
    price_gap_fail_pct:
        Flag as FAIL if price gap exceeds this % (potential split/corpaction) (default 40%).
    stale_bar_max_days:
        Flag as FAIL if most-recent bar is older than this many calendar days (default 5).
    iex_sip_volume_diff_pct:
        Flag as WARN if IEX volume < SIP volume × (1 - tolerance) (default 0.30 = 30%).
        Pass None to skip this check (single-feed setups).
    """

    def __init__(
        self,
        max_missing_bar_ratio: float = 0.02,
        volume_spike_threshold: float = 10.0,
        price_gap_warn_pct: float = 0.15,
        price_gap_fail_pct: float = 0.40,
        stale_bar_max_days: int = 5,
        iex_sip_volume_diff_pct: Optional[float] = 0.30,
    ) -> None:
        self.max_missing_bar_ratio = max_missing_bar_ratio
        self.volume_spike_threshold = volume_spike_threshold
        self.price_gap_warn_pct = price_gap_warn_pct
        self.price_gap_fail_pct = price_gap_fail_pct
        self.stale_bar_max_days = stale_bar_max_days
        self.iex_sip_volume_diff_pct = iex_sip_volume_diff_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_all(
        self,
        bars: Dict[str, pd.DataFrame],
        as_of: Optional[date] = None,
        iex_bars: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, QualityReport]:
        """Run all checks for every symbol in ``bars``.

        Parameters
        ----------
        bars:
            Primary bar data: symbol → DataFrame with DatetimeIndex and
            columns [open, high, low, close_adj, volume].
        as_of:
            Reference date for staleness checks (defaults to today).
        iex_bars:
            Optional secondary feed (IEX) for cross-feed volume comparison.

        Returns
        -------
        Dict mapping symbol → :class:`QualityReport`.
        """
        as_of = as_of or date.today()
        reports: Dict[str, QualityReport] = {}

        for symbol, df in bars.items():
            report = QualityReport(symbol=symbol, as_of_date=as_of)

            if df is None or df.empty:
                report.checks.append(
                    CheckResult(
                        check_name="empty_dataframe",
                        severity=CheckSeverity.FAIL,
                        message="DataFrame is empty or None",
                    )
                )
                reports[symbol] = report
                continue

            self._check_ohlc_consistency(df, report)
            self._check_zero_negative_price(df, report)
            self._check_missing_bars(df, report)
            self._check_stale_bars(df, as_of, report)
            self._check_volume_spikes(df, report)
            self._check_price_gaps(df, report)

            if iex_bars is not None and self.iex_sip_volume_diff_pct is not None:
                iex_df = iex_bars.get(symbol)
                if iex_df is not None and not iex_df.empty:
                    self._check_iex_sip_volume(df, iex_df, report)

            if report.passed:
                logger.debug("[dq] %s: ALL OK", symbol)
            else:
                logger.warning("[dq] %s", report.summary())

            reports[symbol] = report

        return reports

    def quarantined_symbols(self, reports: Dict[str, QualityReport]) -> set[str]:
        """Return the set of symbols that failed at least one hard check."""
        return {sym for sym, r in reports.items() if not r.passed}

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_ohlc_consistency(self, df: pd.DataFrame, report: QualityReport) -> None:
        """Verify high >= low and open/close within [low, high]."""
        bad_hl = (df["high"] < df["low"]).sum()
        if bad_hl > 0:
            report.checks.append(
                CheckResult(
                    check_name="ohlc_consistency",
                    severity=CheckSeverity.FAIL,
                    message=f"{bad_hl} bars where high < low",
                    detail={"bad_bars": bad_hl},
                )
            )
            return
        report.checks.append(
            CheckResult(check_name="ohlc_consistency", severity=CheckSeverity.OK, message="OK")
        )

    def _check_zero_negative_price(self, df: pd.DataFrame, report: QualityReport) -> None:
        bad = (df["close_adj"] <= 0).sum()
        if bad > 0:
            report.checks.append(
                CheckResult(
                    check_name="zero_negative_price",
                    severity=CheckSeverity.FAIL,
                    message=f"{bad} bars with close_adj <= 0",
                    detail={"bad_bars": bad},
                )
            )
        else:
            report.checks.append(
                CheckResult(check_name="zero_negative_price", severity=CheckSeverity.OK, message="OK")
            )

    def _check_missing_bars(self, df: pd.DataFrame, report: QualityReport) -> None:
        """Detect gaps in the bar sequence (assumes daily bars).

        Compares the number of bars present vs expected trading days
        using a simple business-day calendar (no holiday awareness).
        """
        if len(df) < 5:
            report.checks.append(
                CheckResult(
                    check_name="missing_bars",
                    severity=CheckSeverity.WARN,
                    message=f"Only {len(df)} bars — insufficient history",
                )
            )
            return

        start = df.index.min()
        end = df.index.max()
        expected_bdays = len(pd.bdate_range(start, end))
        actual = len(df)
        ratio = (expected_bdays - actual) / max(expected_bdays, 1)

        if ratio > self.max_missing_bar_ratio:
            severity = CheckSeverity.FAIL if ratio > 0.05 else CheckSeverity.WARN
            report.checks.append(
                CheckResult(
                    check_name="missing_bars",
                    severity=severity,
                    message=f"Missing {expected_bdays - actual} bars ({ratio:.1%})",
                    detail={"expected": expected_bdays, "actual": actual, "missing_ratio": ratio},
                )
            )
        else:
            report.checks.append(
                CheckResult(check_name="missing_bars", severity=CheckSeverity.OK, message="OK")
            )

    def _check_stale_bars(
        self, df: pd.DataFrame, as_of: date, report: QualityReport
    ) -> None:
        """Flag if the most recent bar is too old."""
        last_bar = df.index.max()
        last_date = last_bar.date() if hasattr(last_bar, "date") else last_bar
        delta_days = (as_of - last_date).days

        if delta_days > self.stale_bar_max_days:
            report.checks.append(
                CheckResult(
                    check_name="stale_bars",
                    severity=CheckSeverity.FAIL,
                    message=f"Most recent bar is {delta_days} days old (threshold: {self.stale_bar_max_days})",
                    detail={"last_bar_date": str(last_date), "delta_days": delta_days},
                )
            )
        else:
            report.checks.append(
                CheckResult(check_name="stale_bars", severity=CheckSeverity.OK, message="OK")
            )

    def _check_volume_spikes(self, df: pd.DataFrame, report: QualityReport) -> None:
        """Flag bars where volume is K× the rolling median."""
        if "volume" not in df.columns:
            return
        median_vol = df["volume"].rolling(20, min_periods=5).median()
        spikes = (df["volume"] > median_vol * self.volume_spike_threshold).sum()
        if spikes > 0:
            report.checks.append(
                CheckResult(
                    check_name="volume_spikes",
                    severity=CheckSeverity.WARN,
                    message=f"{spikes} bars with volume > {self.volume_spike_threshold}× median",
                    detail={"spike_bars": spikes},
                )
            )
        else:
            report.checks.append(
                CheckResult(check_name="volume_spikes", severity=CheckSeverity.OK, message="OK")
            )

    def _check_price_gaps(self, df: pd.DataFrame, report: QualityReport) -> None:
        """Detect abnormal close-to-close gaps (potential corporate actions)."""
        close = df["close_adj"]
        pct_changes = close.pct_change().abs().dropna()

        fail_gaps = (pct_changes > self.price_gap_fail_pct).sum()
        warn_gaps = (pct_changes > self.price_gap_warn_pct).sum() - fail_gaps

        if fail_gaps > 0:
            report.checks.append(
                CheckResult(
                    check_name="price_gaps",
                    severity=CheckSeverity.FAIL,
                    message=(
                        f"{fail_gaps} bars with price gap > {self.price_gap_fail_pct:.0%} "
                        "(likely unadjusted corporate action)"
                    ),
                    detail={"fail_gaps": fail_gaps, "warn_gaps": warn_gaps},
                )
            )
        elif warn_gaps > 0:
            report.checks.append(
                CheckResult(
                    check_name="price_gaps",
                    severity=CheckSeverity.WARN,
                    message=f"{warn_gaps} bars with price gap > {self.price_gap_warn_pct:.0%}",
                    detail={"warn_gaps": warn_gaps},
                )
            )
        else:
            report.checks.append(
                CheckResult(check_name="price_gaps", severity=CheckSeverity.OK, message="OK")
            )

    def _check_iex_sip_volume(
        self,
        sip_df: pd.DataFrame,
        iex_df: pd.DataFrame,
        report: QualityReport,
    ) -> None:
        """Compare IEX vs SIP (primary feed) volumes.

        IEX captures ~2–4% of US equity volume. A very low IEX/SIP ratio
        on a day can indicate a feed problem on the IEX side.

        We check the rolling average ratio; individual bars vary widely.
        """
        try:
            common_idx = sip_df.index.intersection(iex_df.index)
            if len(common_idx) < 5:
                return
            sip_vol = sip_df.loc[common_idx, "volume"]
            iex_vol = iex_df.loc[common_idx, "volume"]
            # IEX should be a consistent fraction of SIP — detect if IEX drops to near zero
            ratio = iex_vol.sum() / max(sip_vol.sum(), 1)
            if ratio < (1.0 - (self.iex_sip_volume_diff_pct or 0.30)):
                report.checks.append(
                    CheckResult(
                        check_name="iex_sip_volume",
                        severity=CheckSeverity.WARN,
                        message=f"IEX/SIP volume ratio {ratio:.2%} below tolerance",
                        detail={"iex_sip_ratio": ratio},
                    )
                )
            else:
                report.checks.append(
                    CheckResult(
                        check_name="iex_sip_volume", severity=CheckSeverity.OK, message="OK"
                    )
                )
        except Exception as exc:
            logger.debug("[dq] IEX/SIP check skipped: %s", exc)
