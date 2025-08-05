import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from explainboostedregg.boosted_explainer import BoostedRegressorExplainer

def test_global_shap_consistency():
    X = np.random.randn(50,5)
    y = X[:,0]*2 + np.random.randn(50)*0.1
    model = GradientBoostingRegressor().fit(X, y)
    expl = BoostedRegressorExplainer(model, X, y, feature_names=[f"f{i}" for i in range(5)])
    fig = expl.plot_global(kind="shap", top_n=3)
    assert fig is not None
