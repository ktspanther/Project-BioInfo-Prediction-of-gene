"""Minimal xgboost stub for sandbox testing only.

This is NOT meant to be a real model. It exists solely so we can verify in
the offline build sandbox that the pipeline wiring is correct end-to-end.
The student should `pip install xgboost` on their machine and delete this
file before running anything for real.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression


class DMatrix:
    def __init__(self, X, label=None):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(label, dtype=int) if label is not None else None


class Booster:
    def __init__(self, lr: LogisticRegression, n_features: int):
        self.lr = lr
        self.n_features = n_features

    def predict(self, d: DMatrix, pred_contribs: bool = False):
        if pred_contribs:
            # Approximate per-feature contributions via weight * (x - mean)
            x = d.X
            mean = self.lr.coef_[0] * 0  # treat mean as 0 for simplicity
            coef = self.lr.coef_[0]
            contribs = x * coef[None, :]
            bias = np.full(x.shape[0], self.lr.intercept_[0])
            return np.column_stack([contribs, bias])
        return self.lr.predict_proba(d.X)[:, 1]


def train(params, dtrain: DMatrix, num_boost_round: int = 50):
    # Ignore most params; this is purely structural
    lr = LogisticRegression(max_iter=200, C=1.0)
    if dtrain.X.shape[1] == 0:
        # Degenerate: just constant predictor
        lr = LogisticRegression(max_iter=200)
        lr.classes_ = np.array([0, 1])
        # Add a dummy column
        X_aug = np.ones((dtrain.X.shape[0], 1))
        lr.fit(X_aug, dtrain.y)
        return Booster(lr, 0)
    lr.fit(dtrain.X, dtrain.y)
    return Booster(lr, dtrain.X.shape[1])
