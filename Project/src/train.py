"""
Hyperparameter tuning (Stage 4) and final model training & prediction (Stage 5).

The paper tunes XGBoost hyperparameters with PSO on the validation set.
We tune a small but impactful set:
    eta (learning rate), max_depth, subsample, colsample_bytree, min_child_weight,
    num_boost_round.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from pso import pso_maximize
from optimizer import get_optimizer


# Per-hyperparameter [lo, hi] bounds for PSO
HP_BOUNDS = np.array([
    [0.01, 0.30],   # eta
    [2, 10],        # max_depth
    [0.5, 1.0],     # subsample
    [0.5, 1.0],     # colsample_bytree
    [1, 10],        # min_child_weight
    [50, 500],      # num_boost_round
], dtype=float)


def _decode_hp(vec: np.ndarray) -> dict:
    return dict(
        eta=float(vec[0]),
        max_depth=int(round(vec[1])),
        subsample=float(vec[2]),
        colsample_bytree=float(vec[3]),
        min_child_weight=float(vec[4]),
        num_boost_round=int(round(vec[5])),
    )


def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         n_particles: int = 12, max_iter: int = 20,
                         patience: int = 6, seed: int = 0,
                         optimizer: str = "pso",
                         verbose: bool = True) -> dict:
    """Find good XGBoost hyperparameters using the chosen optimizer on validation AUC."""

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    def fn(vec):
        hp = _decode_hp(vec)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "verbosity": 0, "nthread": 0,
            "eta": hp["eta"],
            "max_depth": hp["max_depth"],
            "subsample": hp["subsample"],
            "colsample_bytree": hp["colsample_bytree"],
            "min_child_weight": hp["min_child_weight"],
        }
        bst = xgb.train(params, dtrain, num_boost_round=hp["num_boost_round"])
        return float(roc_auc_score(y_val, bst.predict(dval)))

    optimizer_fn = get_optimizer(optimizer)
    res = optimizer_fn(fn, HP_BOUNDS, n_particles=n_particles,
                       max_iter=max_iter, patience=patience,
                       seed=seed, verbose=verbose)
    hp = _decode_hp(res.best_x)
    if verbose:
        print(f"[HP] best val AUC = {res.best_score:.4f}; params = {hp}")
    return hp


def train_final(X_train: np.ndarray, y_train: np.ndarray, hp: dict,
                seed: int = 0) -> xgb.Booster:
    """Train the final classifier with tuned hyperparameters."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "verbosity": 0, "nthread": 0,
        "eta": hp["eta"],
        "max_depth": hp["max_depth"],
        "subsample": hp["subsample"],
        "colsample_bytree": hp["colsample_bytree"],
        "min_child_weight": hp["min_child_weight"],
        "seed": seed,
    }
    return xgb.train(params, dtrain, num_boost_round=hp["num_boost_round"])


def predict_and_score(booster: xgb.Booster, X: np.ndarray,
                      y: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (predicted probabilities, AUC) on a held-out set."""
    d = xgb.DMatrix(X, label=y)
    probs = booster.predict(d)
    auc = float(roc_auc_score(y, probs))
    return probs, auc
