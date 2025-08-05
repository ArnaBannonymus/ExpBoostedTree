import numpy as np
from sklearn.base import is_regressor
from .shap_explainer import SHAPExplainer
from .permutation_explainer import PermutationExplainer
from .plots import bar, waterfall

class BoostedRegressorExplainer:
    def __init__(self, model, X_test, y_test, feature_names=None, use_shap=True, shap_kwargs={}, perm_kwargs={}):
        if not is_regressor(model):
            raise ValueError("Model must be a regressor.")
        self.feature_names = feature_names
        self.shap_exp = SHAPExplainer(model, feature_names, background_data=X_test, **shap_kwargs) if use_shap else None
        self.perm_exp = PermutationExplainer(model, X_test, y_test, feature_names, **perm_kwargs)

    def plot_global(self, kind="shap", top_n=10, **kwargs):
        if kind=="shap" and self.shap_exp:
            imp = self.shap_exp.explain_global(kind="mean_abs", X=self.shap_exp.background, cache_key="X_test")
        elif kind=="perm":
            imp = self.perm_exp.explain_global()
        else:
            raise ValueError("kind must be 'shap' or 'perm'")
        return bar(imp, self.feature_names, top_n=top_n, **kwargs)

    def plot_local(self, x, kind="shap", top_n=10, interactive=False, **kwargs):
        if kind=="shap" and self.shap_exp:
            contribs, base = self.shap_exp.explain_local(x.reshape(1, -1))
            contribs, base = contribs.flatten(), base[0] if np.ndim(base)>0 else base
        elif kind=="perm":
            contribs, base = self.perm_exp.explain_local(x.reshape(1, -1))
            contribs, base = contribs.flatten(), base[0]
        else:
            raise ValueError("kind must be 'shap' or 'perm'")
        idx = np.argsort(np.abs(contribs))[::-1][:top_n]
        if interactive:
            from .plots import waterfall_plotly
            return waterfall_plotly(contribs[idx], np.array(self.feature_names)[idx], base, **kwargs)
        return waterfall(contribs[idx], np.array(self.feature_names)[idx], base, **kwargs)
