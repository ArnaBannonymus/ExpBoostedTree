import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from explainboostedregg.shap_explainer import SHAPExplainer

def test_shap_global_shape():
    X = np.random.randn(30,4)
    y = X[:,0]*1.5 + np.random.randn(30)*0.1
    model = GradientBoostingRegressor().fit(X, y)
    expl = SHAPExplainer(model, feature_names=[f"f{i}" for i in range(4)], background_data=X, cache_dir=".")
    imp = expl.explain_global(kind="mean_abs", X=X)
    assert imp.shape == (4,)
