"""
Position Sizing — Phase 1 (Fixed-Fractional, Lean).

Single responsibility: given a SignalIntent and current portfolio state,
return the dollar position size to risk.

Phase 1 algorithm (from TRADING_BOT_PLAN.md §1 decision #11)
-------------------------------------------------------------
  position_size = (portfolio_value × max_risk_per_trade) / stop_distance_pct
  capped at:    portfolio_value × absolute_max_position_pct

This is intentionally simple. Bayesian Kelly is deferred to Phase 2 once
we have 90+ real trades to calibrate win-rate / avg_win / avg_loss.

The function also:
  - Checks global open-risk budget
  - Converts dollar size to share quantity (rounded down to whole shares)
  - Respects PDT day-trade counter

Upgrade path
------------
To switch to Bayesian Kelly (Phase 2), replace calculate_position_size()
with the BayesianKellySizer class. The PortfolioManager interface does not
change — it still calls size_signal(signal, portfolio_state).
"""

from __future__ import annotations

import logging
import math

from ..signals.models import SignalIntent

logger = logging.getLogger(__name__)


def calculate_position_size(
    signal: SignalIntent,
    portfolio_value: float,
    current_open_risk: float,
    max_risk_per_trade: float,
    absolute_max_position_pct: float,
    max_global_open_risk: float,
    stop_loss_floor_pct: float,
    last_price: float,
) -> int:
    """
    Compute the number of shares to buy for a given signal.

    Parameters
    ----------
    signal                    : The SignalIntent being sized.
    portfolio_value           : Current portfolio equity (USD).
    current_open_risk         : Sum of current open positions' risk as fraction
                                of portfolio (e.g. 0.012 = 1.2% at risk).
    max_risk_per_trade        : Maximum fraction to risk per trade (e.g. 0.01 = 1%).
    absolute_max_position_pct : Hard cap on position size (e.g. 0.03 = 3%).
    max_global_open_risk      : Total portfolio risk budget (e.g. 0.02 = 2%).
    stop_loss_floor_pct       : Minimum stop distance if signal doesn't provide one.
    last_price                : Current ask/last price of the symbol (USD).

    Returns
    -------
    Number of shares (int >= 0). Returns 0 if sizing is rejected.
    """
    if portfolio_value <= 0 or last_price <= 0:
        logger.warning(
            "[sizing] Rejected %s: invalid portfolio_value=%.2f or last_price=%.2f",
            signal.symbol,
            portfolio_value,
            last_price,
        )
        return 0

    stop_pct = signal.stop_distance_pct or stop_loss_floor_pct
    stop_pct = max(stop_pct, 0.005)  # hard floor: 0.5% to avoid absurd sizes

    # ── Fixed-Fractional base size ────────────────────────────────────
    risk_dollars = portfolio_value * max_risk_per_trade
    position_dollars = risk_dollars / stop_pct

    # ── Hard absolute cap ─────────────────────────────────────────────
    max_dollars = portfolio_value * absolute_max_position_pct
    position_dollars = min(position_dollars, max_dollars)

    # ── Global risk budget check ──────────────────────────────────────
    remaining_risk_budget = max_global_open_risk - current_open_risk
    if remaining_risk_budget <= 0:
        logger.info(
            "[sizing] %s rejected: global risk budget exhausted "
            "(open=%.3f%% >= max=%.3f%%)",
            signal.symbol,
            current_open_risk * 100,
            max_global_open_risk * 100,
        )
        return 0

    # Scale down to fit remaining budget
    max_risk_this_trade = remaining_risk_budget * portfolio_value
    if position_dollars * stop_pct > max_risk_this_trade:
        position_dollars = max_risk_this_trade / stop_pct
        logger.debug(
            "[sizing] %s scaled to fit budget: position_dollars=%.2f",
            signal.symbol,
            position_dollars,
        )

    # ── Convert to shares (whole shares only) ────────────────────────
    shares = math.floor(position_dollars / last_price)

    if shares <= 0:
        logger.debug(
            "[sizing] %s: 0 shares after floor (position_dollars=%.2f price=%.2f)",
            signal.symbol,
            position_dollars,
            last_price,
        )
        return 0

    logger.info(
        "[sizing] %s: %d shares @ $%.2f  (risk=%.2f%% stop=%.2f%%)",
        signal.symbol,
        shares,
        last_price,
        (shares * last_price * stop_pct / portfolio_value) * 100,
        stop_pct * 100,
    )
    return shares
