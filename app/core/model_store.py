import pickle, os
import numpy as np
import pandas as pd
from typing import Optional
from app.core.config import settings


class PlattCalibrator:
    """
    LightGBM + Platt LR wrapper.
    Defined here so pickle can always find it on load.
    """
    def __init__(self, base_model, lr):
        self.base_model = base_model
        self.lr         = lr

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1].reshape(-1, 1)
        cal = self.lr.predict_proba(raw)[:, 1]
        return np.column_stack([1 - cal, cal])


class ModelStore:
    def __init__(self):
        self.model             = None
        self.scaler            = None
        self.calibrator        = None
        self.feature_names: list[str] = []
        self.medians: pd.Series = pd.Series(dtype=float)
        self.altman_baseline: dict = {}
        self._loaded           = False

    def load(self, path: Optional[str] = None) -> bool:
        path = path or settings.model_path
        if not os.path.exists(path):
            print(f"⚠️  Model not found at: {path}")
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model           = data["model"]
            self.scaler          = data.get("scaler")
            self.calibrator      = data.get("calibrator")
            self.feature_names   = list(data.get("features", []))
            self.altman_baseline = data.get("altman_baseline", {})
            raw = data.get("medians", {})
            self.medians = raw if isinstance(raw, pd.Series) else pd.Series(raw)
            self._loaded = True
            return True
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None


model_store = ModelStore()