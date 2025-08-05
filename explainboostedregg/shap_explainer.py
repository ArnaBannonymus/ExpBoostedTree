# explainboostedregg/shap_explainer.py

import os
import pickle
import numpy as np
import shap
from sklearn.ensemble import HistGradientBoostingRegressor
from .base import BaseExplainer

class SHAPExplainer(BaseExplainer):
    """
    SHAP-based explainer for tree ensembles (GradientBoostingRegressor,
    HistGradientBoostingRegressor, XGBoost, etc.). Caches SHAP values on disk
    to avoid recomputation, and disables the additivity check for large models.
    """

    def __init__(
        self,
        model,
        feature_names=None,
        background_data=None,
        cache_dir=".shap_cache"
    ):
        """
        Parameters
        ----------
        model : fitted tree-based regressor
            e.g. sklearn.ensemble.GradientBoostingRegressor
        feature_names : list of str, optional
            Names of the features (length = n_features). If None, inferred as X0, X1, ...
        background_data : array-like of shape (n_background_samples, n_features), optional
            Data used by SHAP as the background distribution.
        cache_dir : str, default=".shap_cache"
            Directory in which to cache computed SHAP values.
        """
        self.model = model
        # Infer feature names if not provided
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            n_feats = getattr(model, "n_features_in_", None)
            if n_feats is None and hasattr(model, "feature_importances_"):
                n_feats = model.feature_importances_.shape[0]
            self.feature_names = [f"X{i}" for i in range(n_feats)]
        self.background = background_data
        self.cache_dir = cache_dir

        # Initialize SHAP TreeExplainer (auto-detects hist-based models)
        self._explainer = shap.TreeExplainer(model, data=self.background)

    def _cache_path(self, key: str) -> str:
        """
        Build a filesystem path for caching SHAP values for a given key.
        """
        return os.path.join(self.cache_dir, f"shap_{key}.pkl")

    def explain_global(
        self,
        *,
        kind: str = "mean_abs",
        X=None,
        cache_key: str = "X"
    ) -> np.ndarray:
        """
        Compute global feature importances.

        Parameters
        ----------
        kind : {"mean_abs", "gain"}, default="mean_abs"
            - "mean_abs": compute mean(|SHAP values|) over X.
            - "gain": if the model has `feature_importances_`, return that.
        X : array-like of shape (n_samples, n_features), optional
            Data on which to compute SHAP values. Defaults to background_data.
        cache_key : str, default="X"
            Identifier used to cache/load SHAP values on disk.

        Returns
        -------
        importances : ndarray of shape (n_features,)
        """
        # Fast path: use built-in feature_importances_
        if kind == "gain" and hasattr(self.model, "feature_importances_"):
            return np.array(self.model.feature_importances_)

        # Determine data for SHAP
        if X is None:
            if self.background is None:
                raise ValueError("No data provided for SHAP global explanation.")
            X = self.background

        # Attempt to load cached SHAP values
        cache_file = self._cache_path(cache_key)
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                shap_vals = pickle.load(f)
        else:
            # Compute SHAP values with additivity check disabled
            shap_vals = self._explainer.shap_values(X, check_additivity=False)
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(shap_vals, f)

        arr = np.array(shap_vals)
        # Handle multi-output case: shap_vals shape = (n_outputs, n_samples, n_features)
        if arr.ndim == 3:
            # Average across outputs
            arr = arr.mean(axis=1)

        # Compute mean absolute SHAP value per feature
        return np.mean(np.abs(arr), axis=0)

    def explain_local(
        self,
        X: np.ndarray,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute local SHAP contributions for each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to explain.

        Returns
        -------
        shap_values : ndarray
            If single-output: shape (n_samples, n_features).
            If multi-output: shape (n_outputs, n_samples, n_features).
        base_values : ndarray
            If single-output: shape (n_samples,) (all equal to the model's expected value).
            If multi-output: shape (n_outputs,).
        """
        # Compute SHAP values with additivity check disabled
        sv = self._explainer.shap_values(X, check_additivity=False)
        base = self._explainer.expected_value
        return np.array(sv), np.array(base)
