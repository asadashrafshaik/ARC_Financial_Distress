from fastapi import APIRouter
from app.core.model_store import model_store

router = APIRouter()


@router.get("/health", summary="Service health check")
async def health():
    baseline = model_store.altman_baseline
    return {
        "status":           "ok",
        "model_loaded":     model_store.is_loaded,
        "model_version":    "LightGBM-v3-Temporal-Calibrated",
        "model_features":   len(model_store.feature_names) if model_store.is_loaded else 0,
        "calibrated":       model_store.calibrator is not None,
        "altman_baseline":  baseline,
        "vs_altman_auc":    round(0.0, 4),  # populated after retrain
    }


@router.get("/model-info", summary="Model performance vs Altman Z baseline")
async def model_info():
    if not model_store.is_loaded:
        return {"error": "Model not loaded"}

    baseline = model_store.altman_baseline
    return {
        "model":           "LightGBM v3",
        "features":        len(model_store.feature_names),
        "calibrated":      model_store.calibrator is not None,
        "temporal_split":  "Pre-2020 train / 2020+ test",
        "altman_baseline": baseline,
        "improvements": [
            "Buyback masquerade detection",
            "14 counter-strategy detectors",
            "Platt scaling calibration",
            "SHAP explainability",
            "110 training companies",
            "Temporal out-of-sample validation",
        ]
    }