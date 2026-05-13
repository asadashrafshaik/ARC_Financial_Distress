import numpy as np
import pandas as pd


def _sdiv(a, b):
    return np.where(b != 0, a / b, np.nan)


FEATURE_COLS = [
    "debt_ratio","equity_ratio","net_profit_margin","return_on_assets",
    "current_ratio","asset_turnover","operating_cf_ratio",
    "retained_to_assets","liab_to_equity","net_income_positive","equity_multiplier",
    "Z_X1","Z_X2","Z_X3","Z_X4","Z_X5","altman_z",
    "debt_ratio_chg1","debt_ratio_pct1",
    "return_on_assets_chg1","return_on_assets_pct1",
    "net_profit_margin_chg1","current_ratio_chg1",
    "altman_z_chg1","altman_z_pct1",
    "Assets_pct1","Revenues_pct1",
    "debt_ratio_trend2","return_on_assets_trend2",
    "buyback_flag","z_score_adjusted",
    "is_growth_company","debt_ratio_growth_adjusted",
    "cash_quality_score","cash_vs_income_ratio",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Ticker", "FiscalYear"]).reset_index(drop=True)

    a, l, eq = df["Assets"], df["Liabilities"], df["Equity"]
    rv, ni   = df["Revenues"], df["NetIncome"]
    ca, cl   = df["CurrentAssets"], df["CurrentLiabilities"]
    re, eb   = df["RetainedEarnings"], df["EBIT"]
    oc       = df["OperatingCashFlow"]

    # Group 1 — Base ratios
    df["debt_ratio"]          = _sdiv(l, a)
    df["equity_ratio"]        = _sdiv(eq, a)
    df["net_profit_margin"]   = _sdiv(ni, rv)
    df["return_on_assets"]    = _sdiv(ni, a)
    df["current_ratio"]       = _sdiv(ca, cl)
    df["asset_turnover"]      = _sdiv(rv, a)
    df["operating_cf_ratio"]  = _sdiv(oc, a)
    df["equity_multiplier"]   = _sdiv(a, eq)
    df["retained_to_assets"]  = _sdiv(re, a)
    df["liab_to_equity"]      = _sdiv(l, eq)
    df["net_income_positive"] = (ni > 0).astype(int)

    # Group 2 — Altman Z
    wc = ca - cl
    df["Z_X1"]     = _sdiv(pd.Series(wc.values, index=df.index), a)
    df["Z_X2"]     = _sdiv(re, a)
    df["Z_X3"]     = _sdiv(eb, a)
    df["Z_X4"]     = _sdiv(eq, l)
    df["Z_X5"]     = _sdiv(rv, a)
    df["altman_z"] = (1.2*df["Z_X1"] + 1.4*df["Z_X2"] +
                      3.3*df["Z_X3"] + 0.6*df["Z_X4"] + 1.0*df["Z_X5"])

    # Group 3 — Trend/Lag
    for col in ["debt_ratio","return_on_assets","net_profit_margin",
                "current_ratio","Assets","Revenues","altman_z"]:
        df[f"{col}_lag1"] = df.groupby("Ticker")[col].shift(1)
        df[f"{col}_chg1"] = df[col] - df[f"{col}_lag1"]
        df[f"{col}_pct1"] = _sdiv(df[f"{col}_chg1"],
                                   df[f"{col}_lag1"].abs().replace(0, np.nan))
    for col in ["debt_ratio","return_on_assets","altman_z"]:
        df[f"{col}_lag2"]   = df.groupby("Ticker")[col].shift(2)
        df[f"{col}_trend2"] = _sdiv(df[col] - df[f"{col}_lag2"],
                                     pd.Series(2, index=df.index))

    # Fix 1 — Buyback
    eq_pct  = df.groupby("Ticker")["Equity"].pct_change()
    rev_pct = df.groupby("Ticker")["Revenues"].pct_change()
    df["equity_shrinking"]           = (eq_pct < -0.10).astype(int)
    df["revenue_growing"]            = (rev_pct > 0.05).astype(int)
    df["buyback_flag"]               = ((df["equity_shrinking"]==1) & (df["revenue_growing"]==1)).astype(int)
    df["z_score_adjusted"]           = df["altman_z"] + (df["buyback_flag"] * 1.5)

    # Fix 2 — Growth
    df["is_growth_company"]          = ((rev_pct > 0.10) & (df["asset_turnover"] > 0.50) &
                                        (df["net_income_positive"]==1)).astype(int)
    df["debt_ratio_growth_adjusted"] = df["debt_ratio"] - (df["is_growth_company"] * 0.15)

    # Fix 3 — Cash quality
    ni_abs = ni.abs().replace(0, np.nan)
    df["cash_vs_income_ratio"] = _sdiv(oc, pd.Series(ni_abs.values, index=df.index))
    df["high_cash_quality"]    = (df["cash_vs_income_ratio"] > 1.2).astype(int)
    df["low_cash_quality"]     = (df["cash_vs_income_ratio"] < 0.5).astype(int)
    df["cash_quality_score"]   = df["high_cash_quality"] - df["low_cash_quality"]

    # Clip outliers
    if len(df) >= 20:
        for col in FEATURE_COLS:
            if col in df.columns:
                df[col] = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))

    return df


def get_feature_columns() -> list[str]:
    return FEATURE_COLS
