import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from explainboostedregg.boosted_explainer import BoostedRegressorExplainer

X, y = fetch_california_housing(return_X_y=True)
feat_names = fetch_california_housing().feature_names
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=42)

model = HistGradientBoostingRegressor(random_state=42).fit(X_tr, y_tr)
expl = BoostedRegressorExplainer(model, X_te, y_te, feature_names=feat_names, use_shap=True, perm_kwargs={"n_repeats":5})

expl.plot_global(kind="shap", top_n=8).savefig("global_shap.png")
expl.plot_global(kind="perm", top_n=8).savefig("global_perm.png")
fig = expl.plot_local(X_te[5], kind="shap", top_n=8, interactive=False)
fig.savefig("local_shap.png")
