"""
P&L Attribution — Fama-French 3-Factor Model (Phase 2+ only).

Decomposes portfolio returns into:
  1. Market factor (β × Rm - Rf)
  2. Size factor (SMB — Small Minus Big)
  3. Value factor (HML — High Minus Low book-to-market)
  4. Alpha (residual return unexplained by the three factors)

Usage::

    attr = FamaFrenchAttribution()
    result = attr.run(portfolio_returns, factor_data)
    print(result.alpha, result.beta_market)

Note: Phase 1 does NOT use this module. Attribution is deferred to Phase 2
when we have sufficient live return history (≥ 252 trading days recommended).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """
    Fama-French 3-factor regression output.

    Attributes
    ----------
    alpha           : Annualized Jensen's alpha
    beta_market     : Market beta (sensitivity to Rm - Rf)
    beta_smb        : Size factor loading
    beta_hml        : Value factor loading
    r_squared       : Explanatory power of the 3-factor model
    tracking_error  : Annualized residual volatility
    information_ratio: alpha / tracking_error
    n_obs           : Number of observations used in regression
    """
    alpha: float
    beta_market: float
    beta_smb: float
    beta_hml: float
    r_squared: float
    tracking_error: float
    information_ratio: float
    n_obs: int
    residuals: Optional[pd.Series] = None


class FamaFrenchAttribution:
    """
    Fama-French 3-factor P&L attribution.

    Parameters
    ----------
    risk_free_rate_annual : Annual risk-free rate (default 5% = US T-bill proxy)
    trading_days_per_year : Used to annualize daily estimates (default 252)
    """

    def __init__(
        self,
        risk_free_rate_annual: float = 0.05,
        trading_days_per_year: int = 252,
    ) -> None:
        self.rf_annual = risk_free_rate_annual
        self.trading_days = trading_days_per_year
        self._rf_daily = (1 + risk_free_rate_annual) ** (1 / trading_days_per_year) - 1

    def run(
        self,
        portfolio_returns: pd.Series,
        factor_data: pd.DataFrame,
    ) -> AttributionResult:
        """
        Run Fama-French 3-factor OLS regression.

        Parameters
        ----------
        portfolio_returns : Daily portfolio returns (index = date)
        factor_data       : DataFrame with columns: mkt_rf, smb, hml
                            (factor returns already in excess of risk-free)
                            Index must align with portfolio_returns.

        Returns
        -------
        AttributionResult
        """
        # Align on common dates
        common_idx = portfolio_returns.index.intersection(factor_data.index)
        if len(common_idx) < 30:
            raise ValueError(
                f"Insufficient overlapping observations for attribution: {len(common_idx)} < 30"
            )

        port = portfolio_returns.loc[common_idx]
        factors = factor_data.loc[common_idx, ["mkt_rf", "smb", "hml"]]

        # Excess returns
        excess = port - self._rf_daily

        # OLS: excess_r = alpha + b_mkt * mkt_rf + b_smb * smb + b_hml * hml + e
        X = np.column_stack([
            np.ones(len(factors)),
            factors["mkt_rf"].values,
            factors["smb"].values,
            factors["hml"].values,
        ])
        y = excess.values

        try:
            coeffs, residuals_arr, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"Attribution regression failed: {exc}") from exc

        alpha_daily, beta_mkt, beta_smb, beta_hml = coeffs
        alpha_annual = (1 + alpha_daily) ** self.trading_days - 1

        # Fitted values and residuals
        y_hat = X @ coeffs
        residuals = pd.Series(y - y_hat, index=common_idx)

        # R²
        ss_res = float(np.sum(residuals.values ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Tracking error (annualized residual vol)
        tracking_error = float(residuals.std()) * (self.trading_days ** 0.5)

        information_ratio = alpha_annual / tracking_error if tracking_error > 0 else 0.0

        logger.info(
            "attribution.result alpha=%.4f beta_mkt=%.2f r2=%.3f te=%.4f ir=%.2f",
            alpha_annual, beta_mkt, r_squared, tracking_error, information_ratio,
        )

        return AttributionResult(
            alpha=alpha_annual,
            beta_market=float(beta_mkt),
            beta_smb=float(beta_smb),
            beta_hml=float(beta_hml),
            r_squared=float(r_squared),
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            n_obs=len(common_idx),
            residuals=residuals,
        )
