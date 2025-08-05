import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from explainboostedregg.permutation_explainer import PermutationExplainer

def test_perm_importances_nonnegative():
    X = np.random.randn(40,3)
    y = X[:,1]*3 + np.random.randn(40)*0.2
    model = GradientBoostingRegressor().fit(X, y)
    expl = PermutationExplainer(model, X, y, feature_names=[f"f{i}" for i in range(3)], n_repeats=3)
    imp = expl.explain_global()
    assert np.all(imp >= 0)
