import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from .base import BaseExplainer

class PermutationExplainer(BaseExplainer):
    def __init__(self, model, X_test, y_test, feature_names=None, metric="mse", n_repeats=10, random_state=None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names or [f"X{i}" for i in range(X_test.shape[1])]
        self.metric = metric
        self.n_repeats = n_repeats
        self.rng = np.random.RandomState(random_state)
        self._baseline = self._score(model.predict(X_test), y_test)

    def _score(self, y_pred, y_true):
        return mean_squared_error(y_true, y_pred) if self.metric=="mse" else r2_score(y_true, y_pred)

    def explain_global(self, **kwargs):
        n = self.X_test.shape[1]
        imps = np.zeros(n)
        for j in range(n):
            scores = []
            Xp = self.X_test.copy()
            for _ in range(self.n_repeats):
                self.rng.shuffle(Xp[:, j])
                scores.append(self._score(self.model.predict(Xp), self.y_test))
            imps[j] = np.mean(scores) - self._baseline
        return imps

    def explain_local(self, X, **kwargs):
        return np.zeros((X.shape[0], X.shape[1])), np.zeros(X.shape[0])
