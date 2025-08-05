# explainboostedregg

Advanced explainers for tree-based regressors (GradientBoosting, HistGradientBoosting, XGBoost).

## Installation

```bash
pip install explainboostedregg
```

## Usage

```python
# demo_usage.py

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

from explainboostedregg.boosted_explainer import BoostedRegressorExplainer

# 1) Load data
X, y = fetch_california_housing(return_X_y=True)
feat_names = fetch_california_housing().feature_names

# 2) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3) Fit a boosted regressor
model = HistGradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# 4) Instantiate explainer (uses SHAP + permutation by default)
expl = BoostedRegressorExplainer(
    model,
    X_test,
    y_test,
    feature_names=feat_names,
    use_shap=True,
    shap_kwargs={"model_output": "raw"},
    perm_kwargs={"n_repeats": 5, "metric": "mse"}
)

# 5a) Plot and save global SHAP importances
fig_shap = expl.plot_global(kind="shap", top_n=8, title="Global SHAP Importances")
fig_shap.savefig("global_shap.png")

# 5b) Plot and save global permutation importances
fig_perm = expl.plot_global(kind="perm", top_n=8, title="Global Permutation Importances")
fig_perm.savefig("global_perm.png")

# 5c) Plot and save a local (sample) waterfall for test sample #10
fig_local = expl.plot_local(
    X_test[10],
    kind="shap",
    top_n=8,
    interactive=False,
    title="Sample Waterfall (SHAP)"
)
fig_local.savefig("local_waterfall.png")

```
```python
# In a notebook cell

import numpy as np
import plotly.io as pio
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from explainboostedregg.boosted_explainer import BoostedRegressorExplainer<img width="640" height="480" alt="global_perm" src="https://github.com/user-attachments/assets/e1927be6-0db8-4733-a468-884a3f54311c" />


# Load data
X, y = load_boston(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)

# Train
model = GradientBoostingRegressor(random_state=0).fit(X_tr, y_tr)

# Explain
expl = BoostedRegressorExplainer(model, X_te, y_te,
                                 feature_names=[f"f{i}" for i in range(X.shape[1])],
                                 use_shap=True,
                                 perm_kwargs={"n_repeats": 3})

# Show interactive waterfall for the first test sample
fig = expl.plot_local(X_te[0], kind="shap", top_n=5, interactive=True)
pio.show(fig)

```

## That will produce:

explanations/global_shap.png

explanations/global_perm.png

explanations/local_shap.png

## global_shap.png : 
<img width="640" height="480" alt="global_shap" src="https://github.com/user-attachments/assets/2465a007-feda-4840-904b-5ecd93751aee" />

## global_perm.png:
<img width="640" height="480" alt="global_perm" src="https://github.com/user-attachments/assets/058d100a-ba4e-4500-8a35-33a8231f1906" />

### Notes
Multi-output: If your regressor predicts multiple targets, you can pass output_index=<i> to plot_global/plot_local to select which output dimension to explain.

Caching SHAP: By default, SHAP values are cached in ./.shap_cache/ so rerunning against the same test set is instant.

Interactivity: Pass interactive=True to plot_local for a Plotly chart you can hover/zoom in Jupyter.
