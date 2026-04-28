"""Event-driven backtesting engine.

Design
------
The engine processes daily OHLCV bars in chronological order. For each
trading date it:

  1. Builds a "look-back slice" of all bars up to (and including) the date,
     so strategies only see past data — no look-ahead leakage.
  2. Calls each registered strategy's ``generate_signals()`` method.
  3. Runs signals through a simplified sizing model (fixed-fractional).
  4. Simulates fills using :class:`FillSimulator` with costs from
     :class:`CostModel`.
  5. Manages open positions: applies stop-loss, take-profit, and TTL exits.
  6. Records every completed trade in a ``BacktestTrade``.

The engine is synchronous and deterministic when used inside a
``deterministic_context(seed)`` block.

Limitations (by design)
------------------------
* Long-only, end-of-day bars only (no intraday tick simulation).
* Single broker / no broker failover.
* No PDT rule enforcement (paper trading assumption).
* For multi-strategy ranking / portfolio optimisation, use walk_forward.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, Iterator, List, Optional, Sequence

import pandas as pd

from .costs import CostModel
from .fill_simulator import FillResult, FillSimulator, FillStatus

logger = logging.getLogger(__name__)

# Type alias: symbol → DataFrame (columns: open, high, low, close_adj, volume)
BarDict = Dict[str, pd.DataFrame]


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


@dataclass
class BacktestTrade:
    """One completed round-trip trade."""

    symbol: str
    strategy: str
    entry_date: date
    exit_date: date
    qty: int
    entry_price: float
    exit_price: float
    cost_usd: float          # total round-trip transaction cost
    pnl_gross: float         # (exit - entry) × qty
    pnl_net: float           # pnl_gross - cost_usd
    pnl_pct: float           # pnl_net / (entry_price × qty)
    exit_reason: str         # "stop", "take_profit", "ttl", "eod"
    entry_fill_latency_ms: float = 0.0
    exit_fill_latency_ms: float = 0.0


@dataclass
class BacktestResult:
    """Aggregate results for one backtest run."""

    initial_capital: float
    final_capital: float
    trades: List[BacktestTrade] = field(default_factory=list)

    # Computed lazily via ``compute_metrics()``
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_pnl_per_trade: float = 0.0
    total_trades: int = 0
    total_cost_usd: float = 0.0
    equity_curve: List[float] = field(default_factory=list)

    def compute_metrics(self) -> "BacktestResult":
        """Populate derived statistics from the trades list."""
        if not self.trades:
            self.total_return = (self.final_capital - self.initial_capital) / max(
                self.initial_capital, 1.0
            )
            return self

        import math

        wins = [t for t in self.trades if t.pnl_net > 0]
        self.total_trades = len(self.trades)
        self.win_rate = len(wins) / max(self.total_trades, 1)
        self.avg_pnl_per_trade = sum(t.pnl_net for t in self.trades) / max(
            self.total_trades, 1
        )
        self.total_cost_usd = sum(t.cost_usd for t in self.trades)
        self.total_return = (self.final_capital - self.initial_capital) / max(
            self.initial_capital, 1.0
        )

        # CAGR — need date span
        if self.trades:
            start = min(t.entry_date for t in self.trades)
            end = max(t.exit_date for t in self.trades)
            years = max((end - start).days / 365.25, 1 / 252)
            if self.final_capital > 0 and self.initial_capital > 0:
                self.cagr = (self.final_capital / self.initial_capital) ** (
                    1.0 / years
                ) - 1.0

        # Max drawdown from equity curve
        if self.equity_curve:
            peak = self.equity_curve[0]
            max_dd = 0.0
            for v in self.equity_curve:
                peak = max(peak, v)
                dd = (peak - v) / max(peak, 1.0)
                max_dd = max(max_dd, dd)
            self.max_drawdown = max_dd

        # Sharpe (daily returns, annualised)
        if len(self.equity_curve) > 1:
            daily_rets = [
                (self.equity_curve[i] - self.equity_curve[i - 1])
                / max(self.equity_curve[i - 1], 1.0)
                for i in range(1, len(self.equity_curve))
            ]
            import statistics

            mu = statistics.mean(daily_rets)
            sigma = statistics.stdev(daily_rets) if len(daily_rets) > 1 else 1e-9
            self.sharpe = (mu / max(sigma, 1e-9)) * math.sqrt(252)

        return self

    def summary(self) -> str:
        return (
            f"BacktestResult: trades={self.total_trades}  "
            f"return={self.total_return:.2%}  CAGR={self.cagr:.2%}  "
            f"Sharpe={self.sharpe:.2f}  MaxDD={self.max_drawdown:.2%}  "
            f"WinRate={self.win_rate:.2%}  AvgPnL=${self.avg_pnl_per_trade:.2f}  "
            f"TotalCost=${self.total_cost_usd:.2f}"
        )


# ---------------------------------------------------------------------------
# Internal position tracker
# ---------------------------------------------------------------------------


@dataclass
class _Position:
    symbol: str
    strategy: str
    entry_date: date
    qty: int
    entry_price: float
    stop_price: float
    take_profit_price: float
    ttl_days: int
    cost_usd_entry: float
    entry_fill_latency_ms: float = 0.0
    days_held: int = 0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Event-driven daily backtesting engine.

    Parameters
    ----------
    initial_capital:
        Starting cash (USD).
    strategies:
        List of strategy instances (must implement ``generate_signals()``).
    cost_model:
        :class:`CostModel` instance (defaults to standard parameters).
    fill_simulator:
        :class:`FillSimulator` instance (defaults to standard parameters).
    risk_per_trade:
        Fraction of equity to risk per trade (default 0.01 = 1 %).
    max_open_positions:
        Maximum simultaneous open positions (default 10).
    adv_usd_default:
        Fallback ADV in USD when not available in bars (default $50M).
    spread_bps_default:
        Fallback spread in bps when not available in bars (default 5 bps).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        strategies: Optional[list] = None,
        cost_model: Optional[CostModel] = None,
        fill_simulator: Optional[FillSimulator] = None,
        risk_per_trade: float = 0.01,
        max_open_positions: int = 10,
        adv_usd_default: float = 50_000_000.0,
        spread_bps_default: float = 5.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.strategies = strategies or []
        self.cost_model = cost_model or CostModel()
        self.fill_simulator = fill_simulator or FillSimulator()
        self.risk_per_trade = risk_per_trade
        self.max_open_positions = max_open_positions
        self.adv_usd_default = adv_usd_default
        self.spread_bps_default = spread_bps_default

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        bars: BarDict,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        regime_series: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """Run the backtest over the provided bar data.

        Parameters
        ----------
        bars:
            Dict mapping symbol → DataFrame.  The DataFrame must have a
            DatetimeIndex and columns ``open``, ``high``, ``low``,
            ``close_adj``, ``volume``.
        start_date / end_date:
            Optional date range filter.  If omitted, uses full history.
        regime_series:
            Optional pandas Series indexed by date with values
            "bull" | "bear" | "sideways".  If omitted, defaults to "bull".
        """
        from src.analysis.market_regime import MarketRegime

        # ── Build sorted date index ────────────────────────────────────
        all_dates: set[date] = set()
        for df in bars.values():
            for idx in df.index:
                d = idx.date() if hasattr(idx, "date") else idx
                all_dates.add(d)

        dates = sorted(all_dates)
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]

        if not dates:
            logger.warning("BacktestEngine: no dates in range, returning empty result")
            return BacktestResult(
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital,
            )

        # ── State ─────────────────────────────────────────────────────
        cash = self.initial_capital
        open_positions: List[_Position] = []
        completed_trades: List[BacktestTrade] = []
        equity_curve: List[float] = []

        def _current_equity() -> float:
            pos_value = sum(
                p.qty * _get_close(p.symbol, current_date) for p in open_positions
            )
            return cash + pos_value

        def _get_close(symbol: str, d: date) -> float:
            df = bars.get(symbol)
            if df is None:
                return 0.0
            row = df[df.index.normalize().date == d] if hasattr(df.index, "normalize") else df
            # pandas DatetimeIndex
            try:
                day_str = pd.Timestamp(d)
                if day_str in df.index:
                    return float(df.loc[day_str, "close_adj"])
                # Try date match
                mask = df.index.date == d
                if mask.any():
                    return float(df.loc[mask, "close_adj"].iloc[-1])
            except Exception:
                pass
            return 0.0

        def _get_adv(symbol: str) -> float:
            df = bars.get(symbol)
            if df is None or "volume" not in df.columns:
                return self.adv_usd_default
            if "close_adj" not in df.columns:
                return self.adv_usd_default
            adv_shares = float(df["volume"].rolling(20).mean().dropna().iloc[-1]) if len(df) >= 20 else float(df["volume"].mean())
            avg_close = float(df["close_adj"].rolling(20).mean().dropna().iloc[-1]) if len(df) >= 20 else float(df["close_adj"].mean())
            return adv_shares * avg_close

        # ── Main loop ─────────────────────────────────────────────────
        for current_date in dates:
            # Determine regime for this date
            if regime_series is not None:
                try:
                    regime_val = regime_series.get(pd.Timestamp(current_date), "bull")
                except Exception:
                    regime_val = "bull"
            else:
                regime_val = "bull"

            try:
                regime = MarketRegime(regime_val)
            except ValueError:
                regime = MarketRegime.BULL

            # ── Build look-back bar slices (no look-ahead) ─────────────
            lookback: BarDict = {}
            for symbol, df in bars.items():
                mask = df.index.date <= current_date
                if mask.any():
                    lookback[symbol] = df.loc[mask]

            # ── Exit management (check stops / TP / TTL) ───────────────
            still_open = []
            for pos in open_positions:
                close_px = _get_close(pos.symbol, current_date)
                if close_px <= 0.0:
                    still_open.append(pos)
                    continue

                pos.days_held += 1
                exit_reason: Optional[str] = None

                # Stop-loss check (use bar low as intraday proxy)
                df = bars.get(pos.symbol)
                bar_low = close_px
                if df is not None:
                    try:
                        mask = df.index.date == current_date
                        if mask.any():
                            bar_low = float(df.loc[mask, "low"].iloc[-1])
                    except Exception:
                        pass

                if bar_low <= pos.stop_price:
                    exit_reason = "stop"
                    exit_price = pos.stop_price  # assume filled at stop
                elif close_px >= pos.take_profit_price:
                    exit_reason = "take_profit"
                    exit_price = pos.take_profit_price
                elif pos.days_held >= pos.ttl_days:
                    exit_reason = "ttl"
                    exit_price = close_px
                else:
                    still_open.append(pos)
                    continue

                # ── Simulate exit fill ─────────────────────────────────
                notional = exit_price * pos.qty
                adv_usd = _get_adv(pos.symbol)
                realistic_exit = self.cost_model.apply_fill_price(
                    exit_price, "sell", adv_usd, notional, self.spread_bps_default
                )
                bar_vol = 0.0
                if df is not None:
                    try:
                        mask2 = df.index.date == current_date
                        if mask2.any():
                            bar_vol = float(df.loc[mask2, "volume"].iloc[-1])
                    except Exception:
                        pass

                adv_shares = adv_usd / max(exit_price, 1.0)
                fill_res: FillResult = self.fill_simulator.simulate_fill(
                    order_qty=pos.qty,
                    fill_price=realistic_exit,
                    adv_shares=adv_shares,
                    bar_volume=max(bar_vol, 1.0),
                )

                exit_cost = self.cost_model.estimate_one_way(
                    "sell", notional, pos.qty, adv_usd, self.spread_bps_default
                )
                total_cost = pos.cost_usd_entry + exit_cost.total_usd

                pnl_gross = (fill_res.fill_price - pos.entry_price) * fill_res.filled_qty
                pnl_net = pnl_gross - total_cost

                cash += fill_res.filled_qty * fill_res.fill_price - exit_cost.total_usd

                completed_trades.append(
                    BacktestTrade(
                        symbol=pos.symbol,
                        strategy=pos.strategy,
                        entry_date=pos.entry_date,
                        exit_date=current_date,
                        qty=fill_res.filled_qty,
                        entry_price=pos.entry_price,
                        exit_price=fill_res.fill_price,
                        cost_usd=total_cost,
                        pnl_gross=pnl_gross,
                        pnl_net=pnl_net,
                        pnl_pct=pnl_net / max(pos.entry_price * fill_res.filled_qty, 1.0),
                        exit_reason=exit_reason,
                        entry_fill_latency_ms=pos.entry_fill_latency_ms,
                        exit_fill_latency_ms=fill_res.latency_ms,
                    )
                )

                # If partial fill on exit, keep remainder open
                if fill_res.remaining_qty > 0:
                    remaining = _Position(
                        symbol=pos.symbol,
                        strategy=pos.strategy,
                        entry_date=pos.entry_date,
                        qty=fill_res.remaining_qty,
                        entry_price=pos.entry_price,
                        stop_price=pos.stop_price,
                        take_profit_price=pos.take_profit_price,
                        ttl_days=pos.ttl_days,
                        cost_usd_entry=0.0,  # already charged
                        entry_fill_latency_ms=pos.entry_fill_latency_ms,
                        days_held=pos.days_held,
                    )
                    still_open.append(remaining)

            open_positions = still_open

            # ── Signal generation ──────────────────────────────────────
            if len(open_positions) < self.max_open_positions:
                open_symbols = {p.symbol for p in open_positions}
                for strategy in self.strategies:
                    try:
                        signals = list(strategy.generate_signals(lookback, regime))
                    except Exception as exc:
                        logger.warning("Strategy %s raised: %s", strategy.name, exc)
                        continue

                    for signal in signals:
                        if signal.symbol in open_symbols:
                            continue
                        if len(open_positions) >= self.max_open_positions:
                            break

                        close_px = _get_close(signal.symbol, current_date)
                        if close_px <= 0.0:
                            continue

                        equity = _current_equity()
                        stop_dist = signal.stop_distance_pct or 0.03
                        # Fixed-fractional sizing: risk_per_trade × equity / stop_dist
                        risk_amt = equity * self.risk_per_trade
                        qty = max(int(risk_amt / max(close_px * stop_dist, 1.0)), 1)
                        notional = qty * close_px

                        # Capacity guard: never exceed max_positions allocation
                        max_pos_notional = equity / self.max_open_positions
                        if notional > max_pos_notional:
                            qty = max(int(max_pos_notional / max(close_px, 1.0)), 1)
                            notional = qty * close_px

                        if cash < notional:
                            continue  # insufficient cash

                        adv_usd = _get_adv(signal.symbol)
                        realistic_entry = self.cost_model.apply_fill_price(
                            close_px, "buy", adv_usd, notional, self.spread_bps_default
                        )

                        df = bars.get(signal.symbol)
                        bar_vol = float(df["volume"].iloc[-1]) if df is not None and not df.empty else 1.0
                        adv_shares = adv_usd / max(realistic_entry, 1.0)

                        fill_res = self.fill_simulator.simulate_fill(
                            order_qty=qty,
                            fill_price=realistic_entry,
                            adv_shares=adv_shares,
                            bar_volume=max(bar_vol, 1.0),
                        )

                        if fill_res.filled_qty == 0:
                            continue

                        entry_cost = self.cost_model.estimate_one_way(
                            "buy",
                            fill_res.filled_qty * fill_res.fill_price,
                            fill_res.filled_qty,
                            adv_usd,
                            self.spread_bps_default,
                        )

                        cash -= fill_res.filled_qty * fill_res.fill_price + entry_cost.total_usd

                        ttl_days = max(signal.ttl_seconds // 86_400, 1)
                        stop_price = fill_res.fill_price * (1.0 - stop_dist)
                        take_profit_price = fill_res.fill_price * (1.0 + stop_dist * 2.0)  # 2:1 R/R

                        open_positions.append(
                            _Position(
                                symbol=signal.symbol,
                                strategy=strategy.name,
                                entry_date=current_date,
                                qty=fill_res.filled_qty,
                                entry_price=fill_res.fill_price,
                                stop_price=stop_price,
                                take_profit_price=take_profit_price,
                                ttl_days=ttl_days,
                                cost_usd_entry=entry_cost.total_usd,
                                entry_fill_latency_ms=fill_res.latency_ms,
                            )
                        )
                        open_symbols.add(signal.symbol)

                        logger.debug(
                            "[backtest] ENTRY %s date=%s qty=%d price=%.2f",
                            signal.symbol,
                            current_date,
                            fill_res.filled_qty,
                            fill_res.fill_price,
                        )

            # ── Record equity curve ────────────────────────────────────
            equity_curve.append(_current_equity())

        # ── Force-close remaining positions at last date ───────────────
        last_date = dates[-1]
        for pos in open_positions:
            close_px = _get_close(pos.symbol, last_date)
            if close_px <= 0.0:
                continue
            adv_usd = _get_adv(pos.symbol)
            exit_cost = self.cost_model.estimate_one_way(
                "sell", close_px * pos.qty, pos.qty, adv_usd, self.spread_bps_default
            )
            pnl_gross = (close_px - pos.entry_price) * pos.qty
            total_cost = pos.cost_usd_entry + exit_cost.total_usd
            pnl_net = pnl_gross - total_cost
            cash += pos.qty * close_px - exit_cost.total_usd
            completed_trades.append(
                BacktestTrade(
                    symbol=pos.symbol,
                    strategy=pos.strategy,
                    entry_date=pos.entry_date,
                    exit_date=last_date,
                    qty=pos.qty,
                    entry_price=pos.entry_price,
                    exit_price=close_px,
                    cost_usd=total_cost,
                    pnl_gross=pnl_gross,
                    pnl_net=pnl_net,
                    pnl_pct=pnl_net / max(pos.entry_price * pos.qty, 1.0),
                    exit_reason="eod",
                )
            )

        result = BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=cash,
            trades=completed_trades,
            equity_curve=equity_curve,
        )
        result.compute_metrics()
        logger.info("Backtest complete. %s", result.summary())
        return result
