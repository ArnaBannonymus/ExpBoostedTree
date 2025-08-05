import shap
from .base import BaseExplainer

class SHAPExplainer(BaseExplainer):
    def __init__(self, model, feature_names=None, background_data=None, cache_dir=".shap_cache"):
        self.model = model
        self.feature_names = feature_names or [f"X{i}" for i in range(model.n_features_in_)]
        self.background = background_data
        self.cache_dir = cache_dir
        # unified constructor, SHAP will auto-detect the best path
        self._explainer = shap.TreeExplainer(model, data=self.background)
