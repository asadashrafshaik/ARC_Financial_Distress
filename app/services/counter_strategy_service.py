"""
Counter-Strategy Detection Service
14 detectors covering every known financial engineering tactic
that companies use to manipulate distress scores.

SCORE-LOWERING (company looks worse than it is):
  1.  Buyback Masquerade       — equity shrinks from buybacks, not real distress
  2.  Negative Equity Trap     — retained earnings deficit from decades of buybacks
  3.  Spin-off Asset Drain     — assets drop after spin-off, not real deterioration
  4.  R&D Heavy Company        — tech/pharma expensing R&D inflates apparent losses

SCORE-RAISING (company hiding real risk):
  5.  Cash Burn Spiral         — negative OCF + growing liabilities YoY
  6.  Zombie Liquidity         — current ratio looks fine but OCF is negative
  7.  Debt Rollover Risk       — very high debt + declining revenue trend
  8.  Earnings Management      — OCF << Net Income (accrual manipulation)
  9.  Asset Inflation          — sudden asset jump without revenue support
  10. Goodwill Gorging         — large M&A goodwill masking weak core operations
  11. Margin Collapse          — sustained margin deterioration trend
  12. Aggressive Revenue Recog — revenue growth without proportional cash flow
  13. Revenue Smoothing        — suspiciously flat revenue (channel stuffing)

INFORMATIONAL:
  14. Lease Capitalization     — IFRS 16/ASC 842 distorting ratios
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class CounterStrategySignal:
    strategy_detected: str
    description: str
    adjustment_applied: bool
    raw_score: float
    adjusted_score: float
    direction: str   # "lower" | "raise" | "informational"
    severity: str    # "low" | "medium" | "high"


# ── SCORE-LOWERING ────────────────────────────────────────────────────────────

def detect_buyback_masquerade(row: pd.Series, raw: float) -> Optional[CounterStrategySignal]:
    """Equity shrinks via buybacks while revenue/earnings grow — Z-Score falsely signals distress."""
    buyback = int(row.get("buyback_flag", 0)) == 1
    growth  = int(row.get("is_growth_company", 0)) == 1
    ni_pos  = int(row.get("net_income_positive", 0)) == 1
    roa     = float(row.get("return_on_assets", 0) or 0)
    curr_r  = float(row.get("current_ratio", 0) or 0)
    cash_q  = float(row.get("cash_quality_score", 0) or 0)

    if (buyback or growth) and ni_pos and roa > 0.08 and curr_r > 0.5 and cash_q >= 0 and raw > 25:
        adj = min(raw, 20.0)
        return CounterStrategySignal(
            strategy_detected="Buyback Masquerade",
            description=(
                f"Equity is shrinking due to aggressive share buybacks, not operational distress. "
                f"Revenue is growing and ROA is {roa*100:.1f}% — the company is profitable and healthy. "
                f"Traditional debt/equity and Z-Score metrics are distorted by capital returns to shareholders. "
                f"Raw score {raw:.1f}% capped at 20%."
            ),
            adjustment_applied=True, raw_score=raw, adjusted_score=adj,
            direction="lower", severity="high")
    return None


def detect_negative_equity_trap(row: pd.Series, raw: float) -> Optional[CounterStrategySignal]:
    """Negative book equity from decades of buybacks, but genuinely profitable and cash-generative."""
    equity  = float(row.get("equity_ratio", 0) or 0)
    roa     = float(row.get("return_on_assets", 0) or 0)
    ocf_r   = float(row.get("operating_cf_ratio", 0) or 0)
    ni_pos  = int(row.get("net_income_positive", 0)) == 1

    if equity < 0 and ni_pos and roa > 0.05 and ocf_r > 0 and raw > 30:
        adj = min(raw, 30.0)
        return CounterStrategySignal(
            strategy_detected="Negative Equity Trap",
            description=(
                f"Book equity is negative ({equity*100:.1f}% of assets) due to accumulated buybacks "
                f"or large dividends, not operational losses. The company is profitable (ROA {roa*100:.1f}%) "
                f"with positive operating cash flow. This is a known accounting artefact for mature, "
                f"capital-returning businesses like McDonald's and Starbucks. Score capped at 30%."
            ),
            adjustment_applied=True, raw_score=raw, adjusted_score=adj,
            direction="lower", severity="medium")
    return None


def detect_rd_heavy_distortion(row: pd.Series, raw: float) -> Optional[CounterStrategySignal]:
    """Tech/pharma expensing R&D depresses net income even when the business is healthy."""
    ni_pos  = int(row.get("net_income_positive", 0)) == 1
    margin  = float(row.get("net_profit_margin", 0) or 0)
    rev_pct = float(row.get("Revenues_pct1", 0) or 0)
    ocf_r   = float(row.get("operating_cf_ratio", 0) or 0)
    at      = float(row.get("asset_turnover", 0) or 0)

    if not ni_pos and -0.15 < margin < 0.02 and rev_pct > 0.15 and ocf_r > 0.05 and at > 0.3 and raw > 30:
        adj = min(raw, 35.0)
        return CounterStrategySignal(
            strategy_detected="R&D Expensing Distortion",
            description=(
                f"Net margin is thin ({margin*100:.1f}%) but revenue is growing {rev_pct*100:.1f}% YoY "
                f"with positive operating cash flow ({ocf_r*100:.1f}% of assets). Heavy R&D expensing "
                f"under US GAAP is likely depressing reported earnings without reflecting real economic value. "
                f"Common in biotech, SaaS, and semiconductor companies. Score capped at 35%."
            ),
            adjustment_applied=True, raw_score=raw, adjusted_score=adj,
            direction="lower", severity="medium")
    return None


def detect_spinoff_asset_drain(row: pd.Series, raw: float) -> Optional[CounterStrategySignal]:
    """After a spin-off, parent assets drop sharply but remaining business may be healthier."""
    asset_pct = float(row.get("Assets_pct1", 0) or 0)
    rev_pct   = float(row.get("Revenues_pct1", 0) or 0)
    roa       = float(row.get("return_on_assets", 0) or 0)
    ni_pos    = int(row.get("net_income_positive", 0)) == 1

    if asset_pct < -0.25 and rev_pct > asset_pct and roa > 0.05 and ni_pos and raw > 25:
        adj = min(raw, 30.0)
        return CounterStrategySignal(
            strategy_detected="Spin-off / Divestiture Effect",
            description=(
                f"Total assets dropped {abs(asset_pct)*100:.1f}% YoY — consistent with a major spin-off "
                f"or divestiture. Revenue decline was smaller ({rev_pct*100:.1f}%), and the remaining "
                f"business is profitable (ROA {roa*100:.1f}%). Asset-based distress metrics are temporarily "
                f"distorted by the structural change. Score capped at 30%."
            ),
            adjustment_applied=True, raw_score=raw, adjusted_score=adj,
            direction="lower", severity="low")
    return None


# ── SCORE-RAISING ─────────────────────────────────────────────────────────────

def detect_cash_burn_spiral(df: pd.DataFrame, row: pd.Series, raw: float) -> Optional[CounterStrategySignal]:
    """Negative OCF + growing liabilities = funding operations with debt (unsustainable)."""
    ocf_r    = float(row.get("operating_cf_ratio", 0) or 0)
    dr_chg   = float(row.get("debt_ratio_chg1", 0) or 0)
    dr_trend = float(row.get("debt_ratio_trend2", 0) or 0)

    if ocf_r < -0.03 and dr_chg > 0.03 and dr_trend > 0:
        adj = min(raw + 15.0, 95.0)
        return CounterStrategySignal(
            strategy_detected="Cash Burn Spiral",
            description=(
                f"Operating Cash Flow is negative ({ocf_r*100:.1f}% of assets) while the debt ratio "
                f"is rising (+{dr_chg*100:.1f}% YoY, trending upward). The company is funding operations "
                f"by taking on more debt — a pattern that accelerates toward insolvency. "
                f"Without a path to positive OCF, debt service will become unsustainable. Score +15 points."
            ),
            adjustment_applied=True, raw_score=raw, adjusted_score=adj,
            direction="raise", severity="high")
    return None


def detect_zombie_liquidity(row: pd.Series, raw: float) -> Optional[CounterStrategySignal]:
    """Current ratio looks fine but OCF is negative — burning cash behind a liquid facade."""
    curr_r = float(row.get("current_ratio", 0) or 0)
    ocf_r  = float(row.get("operating_cf_ratio", 0) or 0)
    dr     = float(row.get("debt_ratio", 0) or 0)

    if curr_r > 1.0 and ocf_r < -0.02 and dr > 0.5:
        adj = min(raw + 12.0, 95.0)
        return CounterStrategySignal(
            strategy_detected="Zombie Liquidity",
            description=(
                f"Current ratio of {curr_r:.2f}x appears healthy, but OCF is negative "
                f"({ocf_r*100:.1f}% of assets). Current assets (inventory, receivables) "
                f"inflate the ratio while the business burns cash from operations. "
                f"This is a classic pre-bankruptcy pattern. Score +12 points."
            ),
            adjustment_applied=True, raw_score=raw, adjusted_score=adj,
            direction="raise", severity="high")
    return None


def detect_debt_rollover_risk(df: pd.DataFrame, row: pd.Series, raw: float) -> Optional[CounterStrategySignal]:
    """Very high debt + declining revenue + rising leverage = dangerous refinancing cliff."""
    dr       = float(row.get("debt_ratio", 0) or 0)
    rev_chg  = float(row.get("Revenues_pct1", 0) or 0)
    dr_trend = float(row.get("debt_ratio_trend2", 0) or 0)

    if dr > 0.80 and rev_chg < -0.05 and dr_trend > 0:
        adj = min(raw + 10.0, 95.0)
        return CounterStrategySignal(
            strategy_detected="Debt Rollover Risk",
            description=(
                f"Debt ratio is very high ({dr*100:.1f}%), revenues declining ({rev_chg*100:.1f}% YoY), "
                f"and leverage still increasing. Lenders may demand higher rates or refuse to roll over "
                f"maturing debt, triggering a liquidity crisis even if current payments are met. Score +10 points."
            ),
            adjustment_applied=True, raw_score=raw, adjusted_score=adj,
            direction="raise", severity="high")
    return None


def detect_earnings_management(row: pd.Series, raw: float) -> Optional[CounterStrategySignal]:
    """OCF << Net Income — large accruals suggest overstated earnings (Enron pattern)."""
    cvi    = float(row.get("cash_vs_income_ratio", 1) or 1)
    ni_pos = int(row.get("net_income_positive", 0)) == 1

    if ni_pos and 0 < cvi < 0.4:
        pts = 12 if cvi < 0.2 else 8
        adj = min(raw + pts, 95.0)
        return CounterStrategySignal(
            strategy_detected="Earnings Management",
            description=(
                f"Operating Cash Flow is only {cvi:.2f}x Net Income (threshold: 0.4x). "
                f"Large positive accruals suggest reported earnings are overstated relative to "
                f"actual cash generation. Primary red flag in Enron, WorldCom, and Wirecard. Score +{pts} points."
            ),
            adjustment_applied=True, raw_score=raw, adjusted_score=adj,
            direction="raise", severity="high" if cvi < 0.2 else "medium")
    return None


def detect_asset_inflation(df: pd.DataFrame, row: pd.Series, raw: float) -> Optional[CounterStrategySignal]:
    """Assets jump >30% YoY without proportional revenue — M&A padding or goodwill inflation."""
    ap = float(row.get("Assets_pct1", 0) or 0)
    rp = float(row.get("Revenues_pct1", 0) or 0)

    if ap > 0.30 and rp < 0.05:
        adj = min(raw + 10.0, 95.0)
        return CounterStrategySignal(
            strategy_detected="Asset Inflation",
            description=(
                f"Total assets grew {ap*100:.1f}% YoY while revenues grew only {rp*100:.1f}%. "
                f"Indicates M&A-driven asset padding, goodwill from overpriced acquisitions, or balance "
                f"sheet engineering. Inflated assets artificially improve leverage ratios without "
                f"proportional economic value. Score +10 points."
            ),
            adjustment_applied=True, raw_score=raw, adjusted_score=adj,
            direction="raise", severity="medium")
    return None


def detect_goodwill_gorging(row: pd.Series, raw: float) -> Optional[CounterStrategySignal]:
    """Serial acquisitions accumulating unproductive goodwill that may be impaired but not written down."""
    asset_trend = float(row.get("Assets_pct1", 0) or 0)
    rev_trend   = float(row.get("Revenues_pct1", 0) or 0)
    dr          = float(row.get("debt_ratio", 0) or 0)
    at          = float(row.get("asset_turnover", 0) or 0)

    if asset_trend > 0.20 and rev_trend < asset_trend * 0.5 and dr > 0.55 and at < 0.5:
        adj = min(raw + 8.0, 95.0)
        return CounterStrategySignal(
            strategy_detected="Goodwill Gorging",
            description=(
                f"Assets growing {asset_trend*100:.1f}% YoY while revenue only grows {rev_trend*100:.1f}% "
                f"and asset turnover is low ({at:.2f}x). Consistent with serial acquisitions accumulating "
                f"unproductive goodwill. Impairments are often delayed, making the balance sheet look "
                f"stronger than it is. Score +8 points."
            ),
            adjustment_applied=True, raw_score=raw, adjusted_score=adj,
            direction="raise", severity="medium")
    return None


def detect_margin_collapse(df: pd.DataFrame, row: pd.Series, raw: float) -> Optional[CounterStrategySignal]:
    """Net margin declining for 3+ consecutive years — structural competitive pressure."""
    ticker = row.get("Ticker")
    if ticker is None or df is None: return None
    co = df[df["Ticker"] == ticker].sort_values("FiscalYear")
    margins = co["net_profit_margin"].dropna().tail(4)
    if len(margins) < 3: return None
    diffs = margins.diff().dropna()
    total_drop = float(margins.iloc[-1] - margins.iloc[0])

    if (diffs < 0).all() and total_drop < -0.08:
        adj = min(raw + 7.0, 95.0)
        return CounterStrategySignal(
            strategy_detected="Margin Collapse",
            description=(
                f"Net profit margin has declined consistently for 3+ consecutive years, "
                f"dropping {abs(total_drop)*100:.1f} percentage points total. "
                f"Sustained deterioration signals structural competitive pressure, cost inflation, "
                f"or pricing power loss that the balance sheet hasn't fully reflected yet. Score +7 points."
            ),
            adjustment_applied=True, raw_score=raw, adjusted_score=adj,
            direction="raise", severity="medium")
    return None


def detect_aggressive_revenue_recognition(row: pd.Series, raw: float) -> Optional[CounterStrategySignal]:
    """Fast revenue growth without proportional cash — related-party transactions or aggressive recognition."""
    rev_pct = float(row.get("Revenues_pct1", 0) or 0)
    cvi     = float(row.get("cash_vs_income_ratio", 1) or 1)
    ocf_r   = float(row.get("operating_cf_ratio", 0) or 0)
    margin  = float(row.get("net_profit_margin", 0) or 0)

    if rev_pct > 0.20 and 0 < cvi < 0.5 and ocf_r < 0.03 and 0 < margin < 0.05:
        adj = min(raw + 8.0, 95.0)
        return CounterStrategySignal(
            strategy_detected="Aggressive Revenue Recognition",
            description=(
                f"Revenue growing {rev_pct*100:.1f}% YoY but cash conversion is poor "
                f"(OCF/NI: {cvi:.2f}x) and margins are thin ({margin*100:.1f}%). "
                f"Fast revenue without proportional cash is associated with aggressive recognition, "
                f"channel stuffing, or related-party transactions with no real economic substance. Score +8 points."
            ),
            adjustment_applied=True, raw_score=raw, adjusted_score=adj,
            direction="raise", severity="medium")
    return None


def detect_revenue_smoothing(df: pd.DataFrame, row: pd.Series, raw: float) -> Optional[CounterStrategySignal]:
    """Revenue growth std dev < 1.5% over 5 years — suspiciously stable, possible channel stuffing."""
    ticker = row.get("Ticker")
    if ticker is None or df is None: return None
    co = df[df["Ticker"] == ticker].sort_values("FiscalYear")
    changes = co["Revenues_pct1"].dropna().tail(5)
    if len(changes) < 4: return None
    std, mean = float(changes.std()), float(changes.mean())

    if mean > 0.01 and std < 0.015:
        adj = min(raw + 5.0, 95.0)
        return CounterStrategySignal(
            strategy_detected="Revenue Smoothing",
            description=(
                f"Revenue growth std dev of {std*100:.2f}% over 5 years is suspiciously stable "
                f"(mean: {mean*100:.1f}%/yr). Real businesses in competitive markets fluctuate. "
                f"Associated with channel stuffing or bill-and-hold arrangements. Score +5 points."
            ),
            adjustment_applied=True, raw_score=raw, adjusted_score=adj,
            direction="raise", severity="low")
    return None


# ── INFORMATIONAL ─────────────────────────────────────────────────────────────

def detect_lease_capitalization(row: pd.Series, raw: float) -> Optional[CounterStrategySignal]:
    """IFRS 16/ASC 842 forces operating leases onto balance sheet — informational only."""
    dr = float(row.get("debt_ratio", 0) or 0)
    cr = float(row.get("current_ratio", 1) or 1)

    if dr > 0.70 and 0.85 < cr < 1.15:
        return CounterStrategySignal(
            strategy_detected="Lease Capitalization Effect",
            description=(
                f"High debt ratio ({dr*100:.1f}%) + current ratio near 1.0 ({cr:.2f}x) is consistent with "
                f"IFRS 16/ASC 842 lease capitalization. Operating leases appear as liabilities, inflating "
                f"debt ratio without representing traditional borrowings. Retailers, airlines, and restaurants "
                f"are most affected. No score adjustment — verify industry context."
            ),
            adjustment_applied=False, raw_score=raw, adjusted_score=raw,
            direction="informational", severity="low")
    return None


# ── ORCHESTRATOR ──────────────────────────────────────────────────────────────

def run_all_detectors(
    df: pd.DataFrame,
    row: pd.Series,
    raw: float,
) -> tuple[float, list[CounterStrategySignal]]:
    """
    Run all 14 detectors in priority order.
    Pass 1: Score-lowering (only most significant fires — no stacking).
    Pass 2: Score-raising (all stack additively).
    Pass 3: Informational only.
    """
    signals: list[CounterStrategySignal] = []
    score = raw

    # Pass 1 — lowering (first match wins)
    for fn in [
        lambda: detect_buyback_masquerade(row, score),
        lambda: detect_negative_equity_trap(row, score),
        lambda: detect_rd_heavy_distortion(row, score),
        lambda: detect_spinoff_asset_drain(row, score),
    ]:
        sig = fn()
        if sig:
            score = sig.adjusted_score
            signals.append(sig)
            break

    # Pass 2 — raising (all stack)
    for fn in [
        lambda: detect_cash_burn_spiral(df, row, score),
        lambda: detect_zombie_liquidity(row, score),
        lambda: detect_debt_rollover_risk(df, row, score),
        lambda: detect_earnings_management(row, score),
        lambda: detect_asset_inflation(df, row, score),
        lambda: detect_goodwill_gorging(row, score),
        lambda: detect_margin_collapse(df, row, score),
        lambda: detect_aggressive_revenue_recognition(row, score),
        lambda: detect_revenue_smoothing(df, row, score),
    ]:
        sig = fn()
        if sig:
            score = sig.adjusted_score
            signals.append(sig)

    # Pass 3 — informational
    sig = detect_lease_capitalization(row, score)
    if sig:
        signals.append(sig)

    return round(min(score, 99.0), 1), signals