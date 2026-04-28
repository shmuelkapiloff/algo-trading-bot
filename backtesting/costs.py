"""Transaction Cost Model for backtesting.

Models three cost components:
1. Regulatory fees  : SEC Section 31 fee + FINRA TAF (sell-side only)
2. Spread cost      : half-spread × order notional (taker pays both entry & exit)
3. Market impact    : tiered slippage by order_notional / ADV_USD ratio

Round-trip cost estimate in bps:
    total_bps ≈ 10–25 bps for a typical S&P 500 stock at < 0.5 % ADV

References:
    SEC Section 31 fee rate: $0.0000278 per $1 notional (2024 schedule)
    FINRA TAF: $0.000166 per share sold, max $8.30 per trade
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Regulatory fee constants (2024 rates)
# ---------------------------------------------------------------------------

#: SEC Section 31 fee — applied to the sell-side notional only.
SEC_FEE_RATE: float = 0.0000278  # $0.0000278 per $1 notional

#: FINRA Trading Activity Fee (TAF) — applied per share sold.
FINRA_TAF_PER_SHARE: float = 0.000166  # $0.000166 per share

#: FINRA TAF cap per trade.
FINRA_TAF_MAX: float = 8.30  # $8.30 per trade


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TradeCosts:
    """Detailed cost breakdown for a trade (one-way or round-trip)."""

    sec_fee_usd: float = 0.0
    finra_taf_usd: float = 0.0
    spread_cost_usd: float = 0.0
    slippage_usd: float = 0.0
    total_usd: float = 0.0
    total_bps: float = 0.0

    def __add__(self, other: "TradeCosts") -> "TradeCosts":
        total = self.total_usd + other.total_usd
        # notional unknown here — caller uses estimate_round_trip instead
        return TradeCosts(
            sec_fee_usd=self.sec_fee_usd + other.sec_fee_usd,
            finra_taf_usd=self.finra_taf_usd + other.finra_taf_usd,
            spread_cost_usd=self.spread_cost_usd + other.spread_cost_usd,
            slippage_usd=self.slippage_usd + other.slippage_usd,
            total_usd=total,
            total_bps=0.0,  # must recompute with notional
        )


# ---------------------------------------------------------------------------
# Cost Model
# ---------------------------------------------------------------------------


class CostModel:
    """Realistic transaction cost model.

    Parameters
    ----------
    spread_bps_default:
        Default half-spread in bps when caller does not supply one (5 bps).
    slippage_bps_base:
        Base slippage for orders < 0.5 % ADV (3 bps).
    adv_impact_scaling:
        Multiplier for ADV-based market impact tier breaks (unused directly,
        available for subclass customisation).
    """

    def __init__(
        self,
        spread_bps_default: float = 5.0,
        slippage_bps_base: float = 3.0,
        adv_impact_scaling: float = 0.1,
    ) -> None:
        self.spread_bps_default = spread_bps_default
        self.slippage_bps_base = slippage_bps_base
        self.adv_impact_scaling = adv_impact_scaling

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _impact_bps(self, adv_ratio: float) -> float:
        """Tiered market-impact slippage based on order size / ADV."""
        if adv_ratio < 0.005:
            return self.slippage_bps_base  # < 0.5 % ADV
        if adv_ratio < 0.010:
            return self.slippage_bps_base + 2.0  # 0.5 – 1 %
        if adv_ratio < 0.020:
            return self.slippage_bps_base + 5.0  # 1 – 2 %
        return self.slippage_bps_base + 10.0  # > 2 %

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_one_way(
        self,
        side: str,  # "buy" | "sell"
        notional_usd: float,
        shares: float,
        adv_usd: float,
        spread_bps: float | None = None,
    ) -> TradeCosts:
        """Estimate one-way transaction cost (entry OR exit).

        Parameters
        ----------
        side:         "buy" or "sell"
        notional_usd: order notional in USD (price × qty)
        shares:       number of shares in the order
        adv_usd:      average daily volume in USD for the symbol
        spread_bps:   bid/ask full-spread in bps (uses default if None)
        """
        spread = spread_bps if spread_bps is not None else self.spread_bps_default
        notional_usd = max(notional_usd, 1e-9)

        # ── Regulatory fees (sell only) ────────────────────────────
        sec_fee = 0.0
        finra_taf = 0.0
        if side == "sell":
            sec_fee = notional_usd * SEC_FEE_RATE
            finra_taf = min(shares * FINRA_TAF_PER_SHARE, FINRA_TAF_MAX)

        # ── Spread cost (half-spread, market taker model) ──────────
        spread_cost = notional_usd * (spread / 2.0) / 10_000.0

        # ── Market impact (tiered by ADV ratio) ───────────────────
        adv_safe = max(adv_usd, 1.0)
        adv_ratio = notional_usd / adv_safe
        impact_bps = self._impact_bps(adv_ratio)
        slippage_cost = notional_usd * impact_bps / 10_000.0

        total = sec_fee + finra_taf + spread_cost + slippage_cost
        total_bps = total / notional_usd * 10_000.0

        return TradeCosts(
            sec_fee_usd=sec_fee,
            finra_taf_usd=finra_taf,
            spread_cost_usd=spread_cost,
            slippage_usd=slippage_cost,
            total_usd=total,
            total_bps=total_bps,
        )

    def estimate_round_trip(
        self,
        notional_usd: float,
        shares: float,
        adv_usd: float,
        spread_bps: float | None = None,
    ) -> TradeCosts:
        """Full round-trip cost (entry + exit).

        Regulatory fees are applied on exit (sell) side only.
        """
        entry = self.estimate_one_way("buy", notional_usd, shares, adv_usd, spread_bps)
        exit_ = self.estimate_one_way("sell", notional_usd, shares, adv_usd, spread_bps)

        total = entry.total_usd + exit_.total_usd
        total_bps = total / max(notional_usd, 1e-9) * 10_000.0

        return TradeCosts(
            sec_fee_usd=exit_.sec_fee_usd,
            finra_taf_usd=exit_.finra_taf_usd,
            spread_cost_usd=entry.spread_cost_usd + exit_.spread_cost_usd,
            slippage_usd=entry.slippage_usd + exit_.slippage_usd,
            total_usd=total,
            total_bps=total_bps,
        )

    def apply_fill_price(
        self,
        nominal_price: float,
        side: str,
        adv_usd: float,
        order_notional: float,
        spread_bps: float | None = None,
    ) -> float:
        """Return the realistic fill price after spread + slippage adjustment.

        BUY:  fill price > nominal (we pay more)
        SELL: fill price < nominal (we receive less)
        """
        spread = spread_bps if spread_bps is not None else self.spread_bps_default
        adv_safe = max(adv_usd, 1.0)
        adv_ratio = order_notional / adv_safe
        impact_bps = self._impact_bps(adv_ratio)

        # Total adverse fill adjustment in bps
        adjustment_bps = (spread / 2.0) + impact_bps
        adjustment = nominal_price * adjustment_bps / 10_000.0

        if side == "buy":
            return nominal_price + adjustment
        return nominal_price - adjustment
