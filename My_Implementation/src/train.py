# Hyperparameter tuning (Stage 4) and final model training (Stage 5)
from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from pso import pso_maximize
from optimizer import get_optimizer


# Hyperparameter tuning parameters
# lower bounds of parameters in PSO for hyperparameter tuning
# upper bounds of parameters in PSO for hyperparameter tuning
HP_BOUNDS = np.array([
    [300,   700],   # nrounds
    [0.005, 0.2],   # eta
    [0,     10],    # gamma
    [1,     10],    # max_depth
    [1,      5],    # lambda
    [1,     10],    # alpha
    [0,     10],    # min_child_weight
    [0.1,   0.7],   # subsample
], dtype=float)


def _decode_hp(vec):
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


def tune_hyperparameters(X_train, y_train, X_val, y_val,
                         n_particles=20, max_iter=20, patience=3,
                         seed=0, optimizer="pso", verbose=True):
    # same structure as the hyperparameter_tuning function in PatternChrome.R
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


def train_final(X_train, y_train, hp, seed=0):
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


def predict_and_score(booster, X, y):
    d = xgb.DMatrix(X, label=y)
    probs = booster.predict(d)
    auc = float(roc_auc_score(y, probs))
    return probs, auc
