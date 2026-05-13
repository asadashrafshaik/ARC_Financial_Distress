from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class RiskTier(str, Enum):
    LOW      = "LOW"
    MODERATE = "MODERATE"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class AltmanZone(str, Enum):
    DISTRESS = "Distress Zone"
    GREY     = "Grey Zone"
    SAFE     = "Safe Zone"


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., example="AAPL")


class YearlyScore(BaseModel):
    fiscal_year: int
    risk_score: float
    risk_score_raw: float
    altman_z: Optional[float]
    debt_ratio: Optional[float]
    return_on_assets: Optional[float]
    current_ratio: Optional[float]
    net_profit_margin: Optional[float]
    buyback_flag: int
    is_growth_company: int
    cash_quality_score: float
    override_fired: bool
    override_reason: str
    risk_tier: RiskTier


class CounterStrategyWarning(BaseModel):
    strategy_detected: str
    description: str
    adjustment_applied: bool
    raw_score: float
    adjusted_score: float


class ShapContribution(BaseModel):
    feature: str
    label: str
    value: float
    shap_value: float
    direction: str


class ShapExplanation(BaseModel):
    ticker: str
    fiscal_year: int
    base_value: float
    contributions: list[ShapContribution]
    raise_drivers: list[ShapContribution]
    lower_drivers: list[ShapContribution]


class AnalyzeResponse(BaseModel):
    ticker: str
    company_name: str
    cik: str
    fiscal_year_range: str
    total_filings: int
    latest_risk_score: float
    latest_risk_tier: RiskTier
    latest_altman_z: Optional[float]
    latest_altman_zone: AltmanZone
    latest_debt_ratio: Optional[float]
    latest_roa: Optional[float]
    latest_current_ratio: Optional[float]
    latest_net_profit_margin: Optional[float]
    counter_strategies: list[CounterStrategyWarning]
    history: list[YearlyScore]
    shap_explanation: Optional[ShapExplanation] = None
    model_features: int
    model_version: str = "LightGBM-v1-BuybackFix"