import numpy as np
import pandas as pd
from typing import Optional

from app.core.model_store import model_store
from app.services.sec_service import build_raw_dataframe
from app.services.feature_service import build_features, get_feature_columns
from app.services.counter_strategy_service import (
    run_all_detectors, detect_buyback_masquerade
)
from app.services.shap_service import explain_ticker
from app.models.schemas import (
    AnalyzeResponse, YearlyScore, RiskTier, AltmanZone,
    CounterStrategyWarning, ShapExplanation, ShapContribution
)


def _tier(s: float) -> RiskTier:
    if s >= 75: return RiskTier.CRITICAL
    if s >= 50: return RiskTier.HIGH
    if s >= 25: return RiskTier.MODERATE
    return RiskTier.LOW


def _zone(z) -> AltmanZone:
    if z is None or (isinstance(z, float) and np.isnan(z)): return AltmanZone.GREY
    if z < 1.81: return AltmanZone.DISTRESS
    if z < 2.99: return AltmanZone.GREY
    return AltmanZone.SAFE


def _f(val) -> Optional[float]:
    try:
        v = float(val)
        return None if np.isnan(v) else round(v, 6)
    except:
        return None


def _score_df(df_feat: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if not model_store.is_loaded:
        raise RuntimeError("Model not loaded.")

    cols = list(model_store.feature_names) or get_feature_columns()

    # Snapshot display columns BEFORE imputation
    display_cols = ["altman_z", "debt_ratio", "return_on_assets",
                    "current_ratio", "net_profit_margin", "buyback_flag",
                    "is_growth_company", "cash_quality_score", "cash_vs_income_ratio"]
    snapshot = {c: df_feat[c].copy() for c in display_cols if c in df_feat.columns}

    for c in cols:
        if c not in df_feat.columns:
            df_feat[c] = np.nan

    med  = model_store.medians
    X_df = df_feat[cols].copy()
    for c in cols:
        fill_val = float(med[c]) if c in med.index else 0.0
        X_df[c]  = X_df[c].fillna(fill_val)

    X = X_df.values
    if model_store.scaler is not None:
        X = model_store.scaler.transform(X)

    # Use calibrated probabilities if available, else raw model
    if model_store.calibrator is not None:
        probs = model_store.calibrator.predict_proba(X)[:, 1]
    else:
        probs = model_store.model.predict_proba(X)[:, 1]

    df_feat = df_feat.copy()
    df_feat["risk_score_raw"] = (probs * 100).round(1)

    # Restore display columns
    for c, series in snapshot.items():
        df_feat[c] = series.values

    return df_feat, X


def analyze_ticker(ticker: str) -> AnalyzeResponse:
    df_raw, company_name, cik = build_raw_dataframe(ticker)
    if df_raw is None or len(df_raw) == 0:
        raise ValueError(
            f"No 10-K data found for '{ticker.upper()}'. Must be a US-listed company on SEC EDGAR."
        )

    df_feat, X_all = _score_df(build_features(df_raw.copy()))

    latest    = df_feat.iloc[-1]
    raw_score = float(latest["risk_score_raw"])
    final, cs_signals = run_all_detectors(df_feat, latest, raw_score)

    history = []
    for _, row in df_feat.iterrows():
        raw = float(row["risk_score_raw"])
        bb  = detect_buyback_masquerade(row, raw)
        adj = bb.adjusted_score if bb else raw
        history.append(YearlyScore(
            fiscal_year=int(row["FiscalYear"]),
            risk_score=round(adj, 1),
            risk_score_raw=round(raw, 1),
            altman_z=_f(row.get("altman_z")),
            debt_ratio=_f(row.get("debt_ratio")),
            return_on_assets=_f(row.get("return_on_assets")),
            current_ratio=_f(row.get("current_ratio")),
            net_profit_margin=_f(row.get("net_profit_margin")),
            buyback_flag=int(row.get("buyback_flag", 0)),
            is_growth_company=int(row.get("is_growth_company", 0)),
            cash_quality_score=float(row.get("cash_quality_score", 0) or 0),
            override_fired=bb is not None,
            override_reason=bb.description if bb else "",
            risk_tier=_tier(adj),
        ))

    cs_warnings = [
        CounterStrategyWarning(
            strategy_detected=s.strategy_detected,
            description=s.description,
            adjustment_applied=s.adjustment_applied,
            raw_score=s.raw_score,
            adjusted_score=s.adjusted_score,
        ) for s in cs_signals
    ]

    shap_result      = explain_ticker(ticker, df_feat)
    shap_explanation = None
    if "contributions" in shap_result and shap_result["contributions"]:
        shap_explanation = ShapExplanation(
            ticker=shap_result["ticker"],
            fiscal_year=shap_result["fiscal_year"],
            base_value=shap_result["base_value"],
            contributions=[ShapContribution(**c) for c in shap_result["contributions"]],
            raise_drivers=[ShapContribution(**c) for c in shap_result["raise_drivers"]],
            lower_drivers=[ShapContribution(**c) for c in shap_result["lower_drivers"]],
        )

    az = _f(latest.get("altman_z"))
    return AnalyzeResponse(
        ticker=ticker.upper(),
        company_name=company_name,
        cik=cik,
        fiscal_year_range=f"{int(df_raw['FiscalYear'].min())}–{int(df_raw['FiscalYear'].max())}",
        total_filings=len(df_raw),
        latest_risk_score=final,
        latest_risk_tier=_tier(final),
        latest_altman_z=az,
        latest_altman_zone=_zone(az),
        latest_debt_ratio=_f(latest.get("debt_ratio")),
        latest_roa=_f(latest.get("return_on_assets")),
        latest_current_ratio=_f(latest.get("current_ratio")),
        latest_net_profit_margin=_f(latest.get("net_profit_margin")),
        counter_strategies=cs_warnings,
        history=history,
        shap_explanation=shap_explanation,
        model_features=len(model_store.feature_names),
    )