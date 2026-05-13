import os, json, time, requests
from typing import Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from app.core.config import settings

FACTS_URL  = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
TICKER_URL = "https://www.sec.gov/include/ticker.txt"
CACHE_TTL  = 24  # hours

TAGS = {
    "Assets": [
        "Assets",
    ],
    "Liabilities": [
        "Liabilities",
    ],
    "Equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "StockholdersEquityAttributableToParent",
        "CommonStockholdersEquity",
        "TotalEquityGrossOfTax",
        "PartnersCapital",
        "MembersEquity",
        "LimitedPartnersCapital",
        "EquityAttributableToOwnersOfParent",
        "TotalEquity",
    ],
    "Revenues": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
        "TelecommunicationsRevenue",
        "OperatingRevenue",
        "NetSales",
        "PremiumsEarnedNet",
        "InterestAndFeeIncomeLoansAndLeases",
        "NetRevenues",
        "TotalRevenues",
        "RevenueNet",
    ],
    "NetIncome": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
        "NetIncomeLossAttributableToParent",
        "IncomeLossFromContinuingOperations",
        "NetIncomeLossIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "CurrentAssets": [
        "AssetsCurrent",
        "CurrentAssets",
    ],
    "CurrentLiabilities": [
        "LiabilitiesCurrent",
        "CurrentLiabilities",
    ],
    "RetainedEarnings": [
        "RetainedEarningsAccumulatedDeficit",
        "RetainedEarnings",
        "AccumulatedOtherComprehensiveIncomeLossNetOfTax",
    ],
    "EBIT": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
        "OperatingIncome",
        "EarningsBeforeInterestAndTaxes",
        "GrossProfit",
    ],
    "OperatingCashFlow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        "CashGeneratedFromOperations",
        "OperatingCashFlow",
    ],
}


def _cache(name):
    os.makedirs(settings.cache_dir, exist_ok=True)
    return os.path.join(settings.cache_dir, name)


def _stale(path):
    if not os.path.exists(path): return True
    return datetime.now() - datetime.fromtimestamp(os.path.getmtime(path)) > timedelta(hours=CACHE_TTL)


def load_ticker_map() -> dict[str, str]:
    cache = _cache("ticker_cik_map.txt")
    if not _stale(cache):
        raw = open(cache, encoding="utf-8").read()
    else:
        try:
            r = requests.get(TICKER_URL, headers={"User-Agent": settings.sec_user_agent}, timeout=30)
            r.raise_for_status()
            raw = r.text
            open(cache, "w", encoding="utf-8").write(raw)
        except Exception as e:
            if os.path.exists(cache):
                raw = open(cache, encoding="utf-8").read()
            else:
                raise RuntimeError(f"Cannot load SEC ticker map: {e}")
    mapping = {}
    for line in raw.strip().splitlines():
        p = line.strip().split()
        if len(p) >= 2:
            mapping[p[0].upper()] = p[1].zfill(10)
    return mapping


def fetch_company_facts(cik: str) -> Optional[dict]:
    cache = _cache(f"companyfacts_{cik}.json")
    if not _stale(cache):
        try:
            return json.load(open(cache, encoding="utf-8"))
        except json.JSONDecodeError:
            os.remove(cache)
    for attempt in range(1, settings.sec_retries + 1):
        time.sleep(settings.sec_delay)
        try:
            r = requests.get(FACTS_URL.format(cik=cik),
                             headers={"User-Agent": settings.sec_user_agent}, timeout=30)
            if r.status_code == 200:
                data = r.json()
                json.dump(data, open(cache, "w", encoding="utf-8"))
                return data
            if r.status_code == 404: return None
            if r.status_code == 429:
                time.sleep(settings.sec_backoff * 2 ** (attempt - 1))
                continue
            time.sleep(settings.sec_backoff * attempt)
        except Exception:
            time.sleep(settings.sec_backoff * attempt)
    return None


def _extract(us_gaap: dict, synonyms: list[str]) -> dict[int, float]:
    """
    Try each tag synonym in order.
    For each fiscal year, keep the entry with:
      1. Latest filed date
      2. On a tie — largest absolute value (most comprehensive figure)
    Returns the first tag that yields any 10-K annual data.
    """
    for tag in synonyms:
        td = us_gaap.get(tag)
        if not td: continue
        usd = td.get("units", {}).get("USD", [])
        if not usd: continue
        fy_map: dict[int, tuple[float, str]] = {}
        for e in usd:
            if e.get("form") != "10-K": continue
            fy  = e.get("fy")
            val = e.get("val")
            filed = e.get("filed", "")
            if fy is None or val is None: continue
            fy  = int(fy)
            val = float(val)
            if fy not in fy_map:
                fy_map[fy] = (val, filed)
            else:
                existing_val, existing_filed = fy_map[fy]
                if filed > existing_filed:
                    # More recent filing wins
                    fy_map[fy] = (val, filed)
                elif filed == existing_filed and abs(val) > abs(existing_val):
                    # Same date — prefer larger absolute value (more complete figure)
                    fy_map[fy] = (val, filed)
        if fy_map:
            return {fy: v for fy, (v, _) in fy_map.items()}
    return {}


def build_raw_dataframe(ticker: str) -> tuple[Optional[pd.DataFrame], str, str]:
    try:
        tm = load_ticker_map()
    except RuntimeError:
        return None, "", ""

    cik = tm.get(ticker.upper())
    if not cik: return None, "", ""

    facts = fetch_company_facts(cik)
    if not facts: return None, "", ""

    name    = facts.get("entityName", ticker.upper())
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    series = {col: _extract(us_gaap, syns) for col, syns in TAGS.items()}

    # Fallback: derive Assets from Liabilities + Equity if missing
    if not series["Assets"]:
        liab = series.get("Liabilities", {})
        eq   = series.get("Equity", {})
        if liab and eq:
            common_fy = set(liab.keys()) & set(eq.keys())
            series["Assets"] = {fy: liab[fy] + eq[fy] for fy in common_fy}

    # Fallback: derive Liabilities from Assets - Equity if missing
    if not series["Liabilities"]:
        assets = series.get("Assets", {})
        eq     = series.get("Equity", {})
        if assets and eq:
            common_fy = set(assets.keys()) & set(eq.keys())
            series["Liabilities"] = {fy: assets[fy] - eq[fy] for fy in common_fy}

    all_fy: set[int] = set()
    for s in series.values(): all_fy.update(s.keys())
    if not all_fy: return None, name, cik

    rows = []
    for fy in sorted(all_fy):
        row: dict = {"Ticker": ticker.upper(), "Company": name, "CIK": cik, "FiscalYear": fy}
        for col in TAGS:
            row[col] = series[col].get(fy, float("nan"))
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in TAGS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["Assets"]).reset_index(drop=True), name, cik