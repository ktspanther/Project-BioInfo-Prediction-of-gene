"""
Feature extraction via PSO + XGBoost (Stage 2 of the pipeline).

We greedily add patterns one at a time. Each round, PSO searches the parameter
space to find the pattern that gives the biggest XGBoost AUC gain on a random
training subset. We stop when training AUC hits 99.9% or no improvement is seen.

Pattern parameter encoding (matches the original R code):
  vec[0]     : start_bin -- left bound of the genomic region to search (0-indexed)
  vec[1]     : end_bin   -- right bound of the region (0-indexed, inclusive)
  vec[2]     : hm_index  -- which histone mark (floored to integer)
  vec[3]     : threshold -- minimum Pearson r to count a match
  vec[4]     : n_pts     -- number of anchor points (pattern width)
  vec[5:5+NUM_CPS] : anchor point heights (only first n_pts are used)

A constraint filters out regions that are too small or too large:
  MIN_DIST <= (start - end)^2 <= max_dist(n_bins)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from pattern import Pattern, pattern_frequencies, build_feature_matrix
from optimizer import get_optimizer


# These match the R paper's settings
NUM_CPS  = 7     # max anchor points (num_cps in R)
MIN_CPS  = 3     # min anchor points (floor(num_cps/2) = 3)
THR_LO   = 0.25  # minimum threshold (MP_threshold lower bound)
THR_HI   = 0.75  # maximum threshold

# Distance constraint on (start - end)^2, same formula as the R code
MIN_DIST = (NUM_CPS * 3) ** 2   # (7*3)^2 = 441  → |start-end| >= 21 bins


def _max_dist(n_bins: int) -> int:
    # (floor(n_bins / 3))^2 → |start-end| <= n_bins//3
    return (n_bins // 3) ** 2


@dataclass
class FEConfig:
    """Knobs for feature extraction. Defaults match the paper's R settings."""
    n_patterns_max: int = 20       # cap to keep demo runs short; paper uses more
    pso_particles: int = 20        # swarm size (initial_swarm_size in R)
    pso_iters: int = 20            # max PSO iterations (initial_maxit in R)
    pso_patience: int = 3          # PSO inner stagnation (maxit.stagnate in R)
    inner_subset_size: int = 3000  # genes sampled per PSO call (num_sample_genes in R)
    xgb_rounds: int = 50           # nrounds during feature extraction
    xgb_eta: float = 0.2           # eta during feature extraction
    target_train_auc: float = 0.999  # stop when training AUC reaches this
    no_improve_patience: int = 4   # outer loop: stop after N consecutive misses
    seed: int = 0
    optimizer: str = "pso"         # "pso", "de", or "random"


def decode(vec: np.ndarray, n_hm: int, n_bins: int) -> Pattern:
    """Convert a PSO position vector into a Pattern object."""
    start = int(np.floor(vec[0]))
    end   = int(np.floor(vec[1]))
    start = max(0, min(n_bins - 1, start))
    end   = max(0, min(n_bins - 1, end))
    lo, hi = min(start, end), max(start, end)

    hm = int(np.floor(vec[2]))
    hm = max(0, min(n_hm - 1, hm))

    threshold = float(np.clip(vec[3], THR_LO, THR_HI))

    n_pts = int(np.floor(vec[4]))
    n_pts = max(MIN_CPS, min(NUM_CPS, n_pts))

    heights = np.clip(vec[5:5 + n_pts], 0.0, 1.0).astype(np.float32)

    return Pattern(heights=heights, threshold=threshold, hm_index=hm,
                   start_bin=lo, end_bin=hi)


def bounds_for(n_hm: int, n_bins: int) -> np.ndarray:
    """[lo, hi] bounds for each PSO dimension, given the data shape."""
    rows = [
        [0, n_bins - 1],        # start_bin
        [0, n_bins - 1],        # end_bin
        [0, n_hm - 0.001],      # hm_index (floored; -0.001 avoids exact upper hit)
        [THR_LO, THR_HI],       # threshold
        [MIN_CPS, NUM_CPS],     # number of anchor points
    ]
    rows += [[0.0, 1.0]] * NUM_CPS   # anchor heights (NUM_CPS slots, only n_pts used)
    return np.asarray(rows, dtype=float)


def _train_xgb_auc(X_tr: np.ndarray, y_tr: np.ndarray,
                   X_ev: np.ndarray, y_ev: np.ndarray,
                   cfg: FEConfig) -> float:
    if X_tr.shape[1] == 0:
        return 0.5
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    deval  = xgb.DMatrix(X_ev, label=y_ev)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": cfg.xgb_eta,
        "verbosity": 0,
        "nthread": 0,
    }
    bst = xgb.train(params, dtrain, num_boost_round=cfg.xgb_rounds)
    return float(roc_auc_score(y_ev, bst.predict(deval)))


def extract_features(X_train: np.ndarray, y_train: np.ndarray,
                     cfg: FEConfig | None = None,
                     verbose: bool = True) -> tuple[list[Pattern], np.ndarray]:
    """Greedy PSO-driven feature extraction.

    X_train: (n_genes, n_hm, n_bins) signal tensor
    y_train: (n_genes,) binary labels
    Returns: (list of patterns, feature matrix of shape (n_genes, n_patterns))
    """
    cfg = cfg or FEConfig()
    rng = np.random.default_rng(cfg.seed)
    n_genes, n_hm, n_bins = X_train.shape

    patterns: list[Pattern] = []
    feat_cols: list[np.ndarray] = []

    best_auc = 0.5
    stale = 0

    for round_idx in range(cfg.n_patterns_max):
        # Sample a random subset of genes for this PSO round (num_sample_genes in R)
        subset_idx = rng.choice(n_genes, size=min(cfg.inner_subset_size, n_genes),
                                replace=False)
        Xs = X_train[subset_idx]
        ys = y_train[subset_idx]

        if feat_cols:
            base = np.column_stack([col[subset_idx] for col in feat_cols])
        else:
            base = np.empty((len(subset_idx), 0), dtype=int)

        # Build n_bins-dependent constants once per round
        min_d = MIN_DIST
        max_d = _max_dist(n_bins)

        def objective(vec: np.ndarray) -> float:
            # Enforce the distance constraint (same as R: return -Inf equivalent)
            start = int(np.floor(vec[0]))
            end   = int(np.floor(vec[1]))
            dist_sq = (start - end) ** 2
            if not (min_d <= dist_sq <= max_d):
                return 0.5  # constraint violated; baseline AUC

            pat = decode(vec, n_hm, n_bins)
            new_col = pattern_frequencies(Xs, pat).reshape(-1, 1)
            X_cand = np.hstack([base, new_col]) if base.size else new_col
            n = X_cand.shape[0]
            half = n // 2
            return _train_xgb_auc(X_cand[:half], ys[:half],
                                  X_cand[half:], ys[half:], cfg)

        optimizer_fn = get_optimizer(cfg.optimizer)
        result = optimizer_fn(
            objective,
            bounds=bounds_for(n_hm, n_bins),
            n_particles=cfg.pso_particles,
            max_iter=cfg.pso_iters,
            patience=cfg.pso_patience,
            target_score=None,
            seed=cfg.seed + round_idx,
            verbose=False,
        )

        new_pat = decode(result.best_x, n_hm, n_bins)
        new_col_full = pattern_frequencies(X_train, new_pat)
        feat_cols.append(new_col_full)
        patterns.append(new_pat)

        # Check full training AUC to decide whether to stop (paper's stopping criterion)
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
