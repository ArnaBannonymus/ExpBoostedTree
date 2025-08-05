import os
import pickle
import numpy as np
import shap
from sklearn.ensemble import HistGradientBoostingRegressor
from .base import BaseExplainer

class SHAPExplainer(BaseExplainer):
    def __init__(self, model, feature_names=None, background_data=None, cache_dir=".shap_cache"):
        self.model = model
        self.feature_names = feature_names or [f"X{i}" for i in range(model.n_features_in_)]
        self.background = background_data
        self.cache_dir = cache_dir
        if isinstance(model, HistGradientBoostingRegressor):
            self._explainer = shap.TreeExplainer(model, data=background_data, algorithm="hist")
        else:
            self._explainer = shap.TreeExplainer(model, data=background_data)

    def _cache_path(self, key):
        return os.path.join(self.cache_dir, f"shap_{key}.pkl")

    def explain_global(self, kind="mean_abs", X=None, cache_key="X"):
        if kind=="gain" and hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        if X is None:
            X = self.background
        cache_file = self._cache_path(cache_key)
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                shap_vals = pickle.load(f)
        else:
            shap_vals = self._explainer.shap_values(X)
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(shap_vals, f)
        arr = np.array(shap_vals)
        if arr.ndim == 3:
            arr = arr.mean(axis=1)
        return np.mean(np.abs(arr), axis=0)

    def explain_local(self, X, **kwargs):
        sv = self._explainer.shap_values(X)
        base = self._explainer.expected_value
        return np.array(sv), np.array(base)
