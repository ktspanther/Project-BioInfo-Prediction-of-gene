"""
End-to-end PatternChrome pipeline.

Runs the five stages from the paper in order:
  1. Binning           (handled by the data loader)
  2. Feature extraction (PSO + XGBoost, greedy pattern discovery)
  3. Backward elimination (drop redundant patterns)
  4. Hyperparameter tuning (PSO on validation AUC)
  5. Final prediction (train final XGBoost, evaluate on test set)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import time
import numpy as np

from data import Dataset
from pattern import Pattern, build_feature_matrix
from feature_extraction import extract_features, FEConfig
from backward_elim import backward_elimination
from train import tune_hyperparameters, train_final, predict_and_score
from explain import shap_contributions


@dataclass
class PipelineResult:
    patterns: list[Pattern]
    hp: dict
    train_auc: float
    val_auc: float
    test_auc: float
    booster: object
    X_train_feat: np.ndarray
    X_val_feat: np.ndarray
    X_test_feat: np.ndarray
    test_probs: np.ndarray
    shap_test: np.ndarray = field(default_factory=lambda: np.empty(0))
    bias_test: np.ndarray = field(default_factory=lambda: np.empty(0))
    timings: dict = field(default_factory=dict)


def run_pipeline(ds: Dataset,
                 fe_cfg: FEConfig | None = None,
                 verbose: bool = True) -> PipelineResult:
    """Run all five stages and return everything needed for analysis."""
    t = {}

    t0 = time()
    if verbose: print("\n=== Stage 2: feature extraction ===")
    patterns, X_train_feat = extract_features(ds.X_train, ds.y_train,
                                              cfg=fe_cfg, verbose=verbose)
    t["feature_extraction"] = time() - t0

    t0 = time()
    if verbose: print("\n=== Stage 3: backward elimination ===")
    patterns, X_train_feat = backward_elimination(
        patterns, ds.X_train, ds.y_train, ds.X_val, ds.y_val, verbose=verbose
    )
    X_val_feat  = build_feature_matrix(ds.X_val,  patterns)
    X_test_feat = build_feature_matrix(ds.X_test, patterns)
    t["backward_elimination"] = time() - t0

    t0 = time()
    if verbose: print("\n=== Stage 4: hyperparameter tuning ===")
    hp = tune_hyperparameters(X_train_feat, ds.y_train,
                              X_val_feat, ds.y_val,
                              optimizer=fe_cfg.optimizer if fe_cfg else "pso",
                              verbose=verbose)
    t["hyperparameter_tuning"] = time() - t0

    t0 = time()
    if verbose: print("\n=== Stage 5: final training and test evaluation ===")
    booster = train_final(X_train_feat, ds.y_train, hp)
    _, train_auc  = predict_and_score(booster, X_train_feat, ds.y_train)
    _, val_auc    = predict_and_score(booster, X_val_feat,   ds.y_val)
    test_probs, test_auc = predict_and_score(booster, X_test_feat, ds.y_test)
    t["final_training"] = time() - t0
    if verbose:
        print(f"  train AUC = {train_auc:.4f}")
        print(f"  val   AUC = {val_auc:.4f}")
        print(f"  TEST  AUC = {test_auc:.4f}")

    shap_test, bias_test = shap_contributions(booster, X_test_feat)

    return PipelineResult(
        patterns=patterns, hp=hp,
        train_auc=train_auc, val_auc=val_auc, test_auc=test_auc,
        booster=booster,
        X_train_feat=X_train_feat,
        X_val_feat=X_val_feat,
        X_test_feat=X_test_feat,
        test_probs=test_probs,
        shap_test=shap_test, bias_test=bias_test,
        timings=t,
    )
