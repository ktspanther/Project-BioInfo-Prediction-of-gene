#### Backward elimination ####
from __future__ import annotations

from typing import Sequence
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from pattern import Pattern, build_feature_matrix


# XGBoost parameters during training and backward elimination
_ROUNDS = 50
_ETA    = 0.2


def _xgb_val_auc(X_tr, y_tr, X_val, y_val):
    if X_tr.shape[1] == 0:
        return 0.5
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_val, label=y_val)
    params = {"objective": "binary:logistic", "eval_metric": "auc",
              "eta": _ETA, "verbosity": 0, "nthread": 0}
    bst = xgb.train(params, dtrain, num_boost_round=_ROUNDS)
    return float(roc_auc_score(y_val, bst.predict(dval)))


def backward_elimination(patterns, X_train_signal, y_train,
                         X_val_signal, y_val, verbose=True):
    kept = list(patterns)
    X_tr = build_feature_matrix(X_train_signal, kept)
    X_va = build_feature_matrix(X_val_signal,   kept)
    best_auc = _xgb_val_auc(X_tr, y_train, X_va, y_val)
    if verbose:
        print(f"[BE] start: {len(kept)} patterns, val AUC = {best_auc:.4f}")

    progress = True
    while progress and len(kept) > 1:
        progress = False
        for idx in range(len(kept) - 1, -1, -1):
            trial  = kept[:idx] + kept[idx + 1:]
            X_tr_t = np.delete(X_tr, idx, axis=1)
            X_va_t = np.delete(X_va, idx, axis=1)
            auc = _xgb_val_auc(X_tr_t, y_train, X_va_t, y_val)
            if auc > best_auc:   # if(be_accuracy > validation_accuracy)
                kept     = trial
                X_tr     = X_tr_t
                X_va     = X_va_t
                best_auc = auc
                progress = True
                if verbose:
                    print(f"  drop idx={idx}; remaining={len(kept)}; "
                          f"val AUC={best_auc:.4f}")
                break  # restart from end

    if verbose:
        print(f"[BE] done: {len(kept)} patterns kept, val AUC = {best_auc:.4f}")
    return kept, X_tr
