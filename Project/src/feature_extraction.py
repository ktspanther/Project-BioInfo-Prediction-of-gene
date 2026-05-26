"""
Feature extraction via PSO + XGBoost (Stage 2 of the pipeline).

Greedy forward construction of a feature set: each round we use PSO to
search the pattern parameter space for a single pattern whose addition
maximizes XGBoost AUC on a random training subset.

Pattern parameter encoding (continuous vector, fixed length):
    [width_real, h1, h2, ..., h_MAX_W, threshold, hm_index_real]
- `width` is rounded to an integer in [MIN_W, MAX_W]
- only the first `width` heights are used; the rest are ignored
- `threshold` is clipped to [THR_LO, THR_HI]
- `hm_index` is rounded to an integer in [0, n_hm - 1]

This keeps PSO's search space a fixed-dim continuous box even though the
pattern itself has a discrete width.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from pattern import Pattern, pattern_frequencies, build_feature_matrix
from optimizer import get_optimizer


# Search space settings (paper-inspired; widened where the paper is vague)
MIN_W = 3        # minimum pattern width in bins
MAX_W = 8        # maximum pattern width in bins
THR_LO = 0.30    # min correlation threshold
THR_HI = 0.95    # max correlation threshold
HEIGHT_LO = 0.0
HEIGHT_HI = 1.0


@dataclass
class FEConfig:
    """Tuning knobs for feature extraction. Defaults are demo-friendly."""
    n_patterns_max: int = 20      # paper extracts more; keep small for demo speed
    pso_particles: int = 15
    pso_iters: int = 25
    pso_patience: int = 6
    inner_subset_size: int = 2000  # genes used inside PSO scoring
    xgb_rounds: int = 50           # paper uses 50 rounds during feature extraction
    xgb_eta: float = 0.2           # paper uses 0.2 during feature extraction
    target_train_auc: float = 0.999  # paper: stop when training AUC hits 99.9%
    no_improve_patience: int = 4   # outer-loop early stop
    seed: int = 0
    optimizer: str = "pso"         # "pso", "de", or "random" — used for ablation studies


def decode(vec: np.ndarray, n_hm: int) -> Pattern:
    """Convert a continuous PSO position into a Pattern."""
    width = int(round(vec[0]))
    width = max(MIN_W, min(MAX_W, width))
    heights = np.clip(vec[1: 1 + width], HEIGHT_LO, HEIGHT_HI).astype(np.float32)
    threshold = float(np.clip(vec[1 + MAX_W], THR_LO, THR_HI))
    hm = int(round(vec[1 + MAX_W + 1]))
    hm = max(0, min(n_hm - 1, hm))
    return Pattern(heights=heights, threshold=threshold, hm_index=hm)


def bounds_for(n_hm: int) -> np.ndarray:
    """Per-dim [lo, hi] bounds for the PSO encoding."""
    rows = [[MIN_W, MAX_W]]                       # width
    rows += [[HEIGHT_LO, HEIGHT_HI]] * MAX_W       # heights (only first `width` used)
    rows += [[THR_LO, THR_HI]]                    # threshold
    rows += [[0, n_hm - 1]]                       # hm index
    return np.asarray(rows, dtype=float)


def _train_xgb_auc(X_tr: np.ndarray, y_tr: np.ndarray,
                   X_ev: np.ndarray, y_ev: np.ndarray,
                   cfg: FEConfig) -> float:
    """Train XGBoost with feature-extraction settings and return eval AUC."""
    if X_tr.shape[1] == 0:
        # No features → predict the prior; AUC undefined-ish, treat as 0.5
        return 0.5
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dev = xgb.DMatrix(X_ev, label=y_ev)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": cfg.xgb_eta,
        "verbosity": 0,
        "nthread": 0,
    }
    bst = xgb.train(params, dtrain, num_boost_round=cfg.xgb_rounds)
    preds = bst.predict(dev)
    return float(roc_auc_score(y_ev, preds))


def extract_features(X_train: np.ndarray, y_train: np.ndarray,
                     cfg: FEConfig | None = None,
                     verbose: bool = True) -> tuple[list[Pattern], np.ndarray]:
    """Greedy PSO-driven feature extraction.

    Parameters
    ----------
    X_train : (n_genes, n_hm, n_bins) signal tensor
    y_train : (n_genes,) 0/1 labels
    cfg     : FEConfig

    Returns
    -------
    patterns : list[Pattern]
    feature_matrix : (n_genes, n_patterns) integer counts for all training genes
    """
    cfg = cfg or FEConfig()
    rng = np.random.default_rng(cfg.seed)
    n_genes, n_hm, _ = X_train.shape

    patterns: list[Pattern] = []
    feat_cols: list[np.ndarray] = []  # frequencies on all training genes

    best_auc = 0.5
    stale = 0

    for round_idx in range(cfg.n_patterns_max):
        # Random subset for *inside* PSO scoring (the "random and changing
        # subset of 3000 genes" the paper describes — we use 2000 to be fast).
        subset_idx = rng.choice(n_genes, size=min(cfg.inner_subset_size, n_genes),
                                replace=False)
        Xs = X_train[subset_idx]
        ys = y_train[subset_idx]

        # Current feature matrix (already-accepted patterns) on the subset.
        if feat_cols:
            base = np.column_stack([col[subset_idx] for col in feat_cols])
        else:
            base = np.empty((len(subset_idx), 0), dtype=int)

        def objective(vec: np.ndarray) -> float:
            """AUC when we add the candidate pattern to the current set."""
            pat = decode(vec, n_hm)
            new_col = pattern_frequencies(Xs, pat).reshape(-1, 1)
            X_cand = np.hstack([base, new_col]) if base.size else new_col
            # Use a simple 50/50 in-subset eval split for a quick, fair score
            n = X_cand.shape[0]
            half = n // 2
            return _train_xgb_auc(X_cand[:half], ys[:half],
                                  X_cand[half:], ys[half:], cfg)

        optimizer_fn = get_optimizer(cfg.optimizer)
        result = optimizer_fn(
            objective,
            bounds=bounds_for(n_hm),
            n_particles=cfg.pso_particles,
            max_iter=cfg.pso_iters,
            patience=cfg.pso_patience,
            target_score=None,
            seed=cfg.seed + round_idx,
            verbose=False,
        )

        new_pat = decode(result.best_x, n_hm)
        new_col_full = pattern_frequencies(X_train, new_pat)
        feat_cols.append(new_col_full)
        patterns.append(new_pat)

        # Now check the full training AUC with all accepted patterns (paper's
        # stopping criterion is based on full training AUC, not the inner score).
        X_full = np.column_stack(feat_cols)
        half = len(y_train) // 2
        full_auc = _train_xgb_auc(X_full[:half], y_train[:half],
                                  X_full[half:], y_train[half:], cfg)

        improved = full_auc > best_auc + 1e-4
        if improved:
            best_auc = full_auc
            stale = 0
        else:
            stale += 1

        if verbose:
            print(f"[FE round {round_idx + 1:2d}/{cfg.n_patterns_max}] "
                  f"inner_auc={result.best_score:.4f}  "
                  f"full_train_auc={full_auc:.4f}  "
                  f"pat={new_pat}  stale={stale}")

        if full_auc >= cfg.target_train_auc:
            if verbose:
                print(f"  reached target train AUC {cfg.target_train_auc}; stop")
            break
        if stale >= cfg.no_improve_patience:
            if verbose:
                print(f"  no improvement for {cfg.no_improve_patience} rounds; stop")
            break

    feature_matrix = np.column_stack(feat_cols) if feat_cols else \
        np.empty((n_genes, 0), dtype=int)
    return patterns, feature_matrix
