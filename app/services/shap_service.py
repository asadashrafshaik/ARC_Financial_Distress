"""
SHAP Explainability Service
Supports TreeExplainer (LightGBM/XGBoost/RF) and LinearExplainer (LogisticRegression).
"""

import numpy as np
import pandas as pd
import shap

from app.core.model_store import model_store
from app.services.feature_service import get_feature_columns

FEATURE_LABELS = {
    "debt_ratio":                    "Debt Ratio",
    "equity_ratio":                  "Equity Ratio",
    "net_profit_margin":             "Net Profit Margin",
    "return_on_assets":              "Return on Assets",
    "current_ratio":                 "Current Ratio",
    "asset_turnover":                "Asset Turnover",
    "operating_cf_ratio":            "Operating Cash Flow Ratio",
    "retained_to_assets":            "Retained Earnings / Assets",
    "liab_to_equity":                "Liabilities / Equity",
    "net_income_positive":           "Net Income Positive",
    "equity_multiplier":             "Equity Multiplier",
    "Z_X1":                          "Altman Z — Working Capital",
    "Z_X2":                          "Altman Z — Retained Earnings",
    "Z_X3":                          "Altman Z — EBIT / Assets",
    "Z_X4":                          "Altman Z — Equity / Liabilities",
    "Z_X5":                          "Altman Z — Revenue / Assets",
    "altman_z":                      "Altman Z-Score",
    "debt_ratio_chg1":               "Debt Ratio Change (1yr)",
    "debt_ratio_pct1":               "Debt Ratio Change % (1yr)",
    "return_on_assets_chg1":         "ROA Change (1yr)",
    "return_on_assets_pct1":         "ROA Change % (1yr)",
    "net_profit_margin_chg1":        "Profit Margin Change (1yr)",
    "current_ratio_chg1":            "Current Ratio Change (1yr)",
    "altman_z_chg1":                 "Altman Z Change (1yr)",
    "altman_z_pct1":                 "Altman Z Change % (1yr)",
    "Assets_pct1":                   "Asset Growth (1yr)",
    "Revenues_pct1":                 "Revenue Growth (1yr)",
    "debt_ratio_trend2":             "Debt Ratio Trend (2yr)",
    "return_on_assets_trend2":       "ROA Trend (2yr)",
    "buyback_flag":                  "Buyback Detected",
    "z_score_adjusted":              "Adjusted Z-Score",
    "is_growth_company":             "Growth Company Flag",
    "debt_ratio_growth_adjusted":    "Debt Ratio (Growth Adjusted)",
    "cash_quality_score":            "Cash Quality Score",
    "cash_vs_income_ratio":          "Cash vs Income Ratio",
}

_explainer_cache = None
_explainer_type  = None  # "tree" | "linear" | "permutation"


def _get_explainer():
    global _explainer_cache, _explainer_type
    if _explainer_cache is not None:
        return _explainer_cache, _explainer_type
    if not model_store.is_loaded:
        return None, None

    model     = model_store.model
    model_cls = type(model).__name__

    # Tree-based models
    tree_models = {"LGBMClassifier", "XGBClassifier", "RandomForestClassifier",
                   "GradientBoostingClassifier", "DecisionTreeClassifier",
                   "ExtraTreesClassifier"}
    # Linear models
    linear_models = {"LogisticRegression", "LinearSVC", "SGDClassifier",
                     "RidgeClassifier"}

    try:
        if model_cls in tree_models:
            _explainer_cache = shap.TreeExplainer(model)
            _explainer_type  = "tree"
            print(f"✅ SHAP TreeExplainer initialized for {model_cls}")
        elif model_cls in linear_models:
            # LinearExplainer needs a background dataset — use zeros as proxy
            n_features = len(model_store.feature_names) or len(get_feature_columns())
            background = np.zeros((1, n_features))
            _explainer_cache = shap.LinearExplainer(model, background, feature_perturbation="interventional")
            _explainer_type  = "linear"
            print(f"✅ SHAP LinearExplainer initialized for {model_cls}")
        else:
            # Fallback: KernelExplainer (slow but universal)
            n_features = len(model_store.feature_names) or len(get_feature_columns())
            background = np.zeros((1, n_features))
            _explainer_cache = shap.KernelExplainer(model.predict_proba, background)
            _explainer_type  = "kernel"
            print(f"✅ SHAP KernelExplainer initialized for {model_cls}")
        return _explainer_cache, _explainer_type
    except Exception as e:
        print(f"SHAP explainer init failed: {e}")
        return None, None


def compute_shap(X_row: np.ndarray, feature_names: list[str], top_n: int = 12) -> list[dict]:
    explainer, etype = _get_explainer()
    if explainer is None:
        return []
    try:
        shap_values = explainer.shap_values(X_row)

        # Extract class-1 (distressed) SHAP values
        if isinstance(shap_values, list) and len(shap_values) == 2:
            sv = shap_values[1][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv = shap_values[0, :, 1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            sv = shap_values[0]
        else:
            sv = np.array(shap_values).flatten()

        results = []
        for i, feat in enumerate(feature_names):
            if i >= len(sv): break
            sv_val   = float(sv[i])
            feat_val = float(X_row[0, i])
            results.append({
                "feature":    feat,
                "label":      FEATURE_LABELS.get(feat, feat.replace("_", " ").title()),
                "value":      round(feat_val, 4),
                "shap_value": round(sv_val * 100, 2),
                "direction":  "raise" if sv_val > 0 else "lower",
            })

        results.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return results[:top_n]
    except Exception as e:
        print(f"SHAP compute failed: {e}")
        return []


def explain_ticker(ticker: str, df_feat: pd.DataFrame) -> dict:
    if not model_store.is_loaded:
        return {"error": "Model not loaded"}

    cols   = list(model_store.feature_names) or get_feature_columns()
    med    = model_store.medians
    latest = df_feat.iloc[-1].copy()
    X_df   = pd.DataFrame([latest[cols]])
    for c in cols:
        fill = float(med[c]) if c in med.index else 0.0
        X_df[c] = X_df[c].fillna(fill)

    X = X_df.values
    if model_store.scaler is not None:
        X = model_store.scaler.transform(X)

    contributions = compute_shap(X, cols, top_n=12)

    explainer, etype = _get_explainer()
    base_value = 0.0
    if explainer is not None and etype == "tree":
        try:
            ev = explainer.expected_value
            base_value = round(float(ev[1] if isinstance(ev, (list, np.ndarray)) else ev) * 100, 2)
        except Exception:
            pass

    return {
        "ticker":        ticker.upper(),
        "fiscal_year":   int(latest.get("FiscalYear", 0)),
        "base_value":    base_value,
        "contributions": contributions,
        "raise_drivers": [c for c in contributions if c["direction"] == "raise"][:5],
        "lower_drivers": [c for c in contributions if c["direction"] == "lower"][:5],
    }