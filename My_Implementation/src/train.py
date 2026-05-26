"""
Hyperparameter tuning (Stage 4) and final model training (Stage 5).

The paper tunes XGBoost hyperparameters with PSO on the validation set.
Bounds match the original R code's hyperparameter_lower / hyperparameter_upper:
  nrounds [300, 700], eta [0.005, 0.2], gamma [0, 10], max_depth [1, 10],
  lambda [1, 5], alpha [1, 10], min_child_weight [0, 10], subsample [0.1, 0.7]
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from pso import pso_maximize
from optimizer import get_optimizer


# Bounds match hyperparameter_lower and hyperparameter_upper in PatternChrome.R
HP_BOUNDS = np.array([
    [300,   700],   # nrounds
    [0.005, 0.2],   # eta
    [0,     10],    # gamma
    [1,     10],    # max_depth
    [1,      5],    # lambda (L2 regularization)
    [1,     10],    # alpha  (L1 regularization)
    [0,     10],    # min_child_weight
    [0.1,   0.7],   # subsample
], dtype=float)


def _decode_hp(vec: np.ndarray) -> dict:
    return dict(
        num_boost_round=int(round(vec[0])),
        eta=float(vec[1]),
        gamma=float(vec[2]),
        max_depth=int(round(vec[3])),
        reg_lambda=float(vec[4]),
        reg_alpha=float(vec[5]),
        min_child_weight=float(vec[6]),
        subsample=float(vec[7]),
    )


def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         n_particles: int = 20, max_iter: int = 20,
                         patience: int = 3, seed: int = 0,
                         optimizer: str = "pso",
                         verbose: bool = True) -> dict:
    """Use PSO to find good XGBoost hyperparameters on the validation set."""

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    def fn(vec):
        hp = _decode_hp(vec)
        params = {
            "objective":        "binary:logistic",
            "eval_metric":      "auc",
            "verbosity":        0,
            "nthread":          0,
            "eta":              hp["eta"],
            "gamma":            hp["gamma"],
            "max_depth":        hp["max_depth"],
            "lambda":           hp["reg_lambda"],
            "alpha":            hp["reg_alpha"],
            "min_child_weight": hp["min_child_weight"],
            "subsample":        hp["subsample"],
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
    """Train the final model with the tuned hyperparameters."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
        "verbosity":        0,
        "nthread":          0,
        "seed":             seed,
        "eta":              hp["eta"],
        "gamma":            hp["gamma"],
        "max_depth":        hp["max_depth"],
        "lambda":           hp["reg_lambda"],
        "alpha":            hp["reg_alpha"],
        "min_child_weight": hp["min_child_weight"],
        "subsample":        hp["subsample"],
    }
    return xgb.train(params, dtrain, num_boost_round=hp["num_boost_round"])


def predict_and_score(booster: xgb.Booster, X: np.ndarray,
                      y: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (predicted probabilities, AUC) on a held-out set."""
    d = xgb.DMatrix(X, label=y)
    probs = booster.predict(d)
    auc = float(roc_auc_score(y, probs))
    return probs, auc
