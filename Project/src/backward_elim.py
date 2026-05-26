"""
Backward elimination (Stage 3).

After greedy feature extraction we may have redundant patterns. Walk through
the feature list from last-added to first-added; try removing each one. If
validation AUC stays the same or improves, drop it permanently. Restart from
the end whenever we drop something. Stop when a full pass removes nothing.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from pattern import Pattern, build_feature_matrix


def _xgb_val_auc(X_tr: np.ndarray, y_tr: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 rounds: int = 50, eta: float = 0.2) -> float:
    if X_tr.shape[1] == 0:
        return 0.5
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {"objective": "binary:logistic", "eval_metric": "auc",
              "eta": eta, "verbosity": 0, "nthread": 0}
    bst = xgb.train(params, dtrain, num_boost_round=rounds)
    return float(roc_auc_score(y_val, bst.predict(dval)))


def backward_elimination(patterns: Sequence[Pattern],
                         X_train_signal: np.ndarray, y_train: np.ndarray,
                         X_val_signal: np.ndarray, y_val: np.ndarray,
                         verbose: bool = True) -> tuple[list[Pattern], np.ndarray]:
    """Drop patterns whose removal does not hurt validation AUC.

    Parameters
    ----------
    patterns : the feature-extraction output, in the order they were added
    X_*_signal : (n_genes, n_hm, n_bins) signal tensors

    Returns
    -------
    kept : list[Pattern]
    kept_train_feat : (n_train, len(kept)) feature matrix on the training set
    """
    kept = list(patterns)
    X_tr = build_feature_matrix(X_train_signal, kept)
    X_va = build_feature_matrix(X_val_signal, kept)
    best_auc = _xgb_val_auc(X_tr, y_train, X_va, y_val)
    if verbose:
        print(f"[BE] start: {len(kept)} patterns, val AUC = {best_auc:.4f}")

    progress = True
    while progress and len(kept) > 1:
        progress = False
        # walk from end to beginning
        for idx in range(len(kept) - 1, -1, -1):
            trial = kept[:idx] + kept[idx + 1:]
            X_tr_t = np.delete(X_tr, idx, axis=1)
            X_va_t = np.delete(X_va, idx, axis=1)
            auc = _xgb_val_auc(X_tr_t, y_train, X_va_t, y_val)
            if auc >= best_auc:
                kept = trial
                X_tr = X_tr_t
                X_va = X_va_t
                best_auc = auc
                progress = True
                if verbose:
                    print(f"  drop idx={idx}; remaining={len(kept)}; "
                          f"val AUC={best_auc:.4f}")
                break  # restart from end as per paper

    if verbose:
        print(f"[BE] done: {len(kept)} patterns kept, val AUC = {best_auc:.4f}")
    return kept, X_tr
