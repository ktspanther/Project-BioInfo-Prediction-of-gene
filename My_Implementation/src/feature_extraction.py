# Python port of the feature extraction loop from PatternChrome.R
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from pattern import Pattern, pattern_frequencies, build_feature_matrix
from optimizer import get_optimizer


# Training parameters
NUM_CPS  = 7      # maximal number of anchor points of pattern
MIN_CPS  = 3      # floor(num_cps / 2)
THR_LO   = 0.25   # lower bounds of parameters in PSO for feature extraction
THR_HI   = 0.75   # upper bounds of parameters in PSO for feature extraction

# minimum / maximum length of search area for patterns
MIN_DIST = (NUM_CPS * 3) ** 2   # (num_cps*3)^2


def _max_dist(n_bins: int) -> int:
    # (floor(10000/bin_size/3))^2  —  adapted for variable bin count
    return (n_bins // 3) ** 2


@dataclass
class FEConfig:
    # Training parameters (matching PatternChrome.R)
    n_patterns_max: int = 20       # cap for demo speed; paper runs until AUC target
    pso_particles: int = 20        # initial_swarm_size
    pso_iters: int = 20            # initial_maxit
    pso_patience: int = 3          # initial_maxit_stagnate
    inner_subset_size: int = 3000  # number of sample genes per objective function call
    xgb_rounds: int = 50           # nrounds  (XGBoost parameters during training and backward elimination)
    xgb_eta: float = 0.2           # eta
    target_train_auc: float = 0.999  # training accuracy where feature extraction step stops
    no_improve_patience: int = 4   # stop after this many rounds with no gain
    seed: int = 0
    optimizer: str = "pso"


def decode(vec: np.ndarray, hm_idx: int, n_bins: int) -> Pattern:
    # turn a PSO position vector into a Pattern object (HM is fixed, not part of vec)
    start = int(np.floor(vec[0]))
    end   = int(np.floor(vec[1]))
    start = max(0, min(n_bins - 1, start))
    end   = max(0, min(n_bins - 1, end))
    lo, hi = min(start, end), max(start, end)

    threshold = float(np.clip(vec[2], THR_LO, THR_HI))

    n_pts = int(np.floor(vec[3]))
    n_pts = max(MIN_CPS, min(NUM_CPS, n_pts))

    heights = np.clip(vec[4:4 + n_pts], 0.0, 1.0).astype(np.float32)

    return Pattern(heights=heights, threshold=threshold, hm_index=hm_idx,
                   start_bin=lo, end_bin=hi)


def bounds_for(n_bins: int) -> np.ndarray:
    # objective_lower / objective_upper from PatternChrome.R — HM handled separately
    rows = [
        [0, n_bins - 1],    # start bin  (R: 1 to floor(10000/bin_size))
        [0, n_bins - 1],    # end bin
        [THR_LO, THR_HI],   # threshold  (R: 0.25 to 0.75)
        [MIN_CPS, NUM_CPS], # number of anchor points  (R: floor(num_cps/2) to num_cps)
    ]
    rows += [[0.0, 1.0]] * NUM_CPS   # rep(0, num_cps) to rep(1, num_cps)
    return np.asarray(rows, dtype=float)


def _train_xgb_auc(X_tr, y_tr, X_ev, y_ev, cfg):
    if X_tr.shape[1] == 0:
        return 0.5
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    deval  = xgb.DMatrix(X_ev, label=y_ev)
    params = {"objective": "binary:logistic", "eval_metric": "auc",
              "eta": cfg.xgb_eta, "verbosity": 0, "nthread": 0}
    bst = xgb.train(params, dtrain, num_boost_round=cfg.xgb_rounds)
    return float(roc_auc_score(y_ev, bst.predict(deval)))


def extract_features(X_train, y_train, cfg=None, verbose=True):
    cfg = cfg or FEConfig()
    rng = np.random.default_rng(cfg.seed)
    n_genes, n_hm, n_bins = X_train.shape

    patterns = []
    feat_cols = []
    best_auc = 0.5
    stale = 0

    for round_idx in range(cfg.n_patterns_max):
        # train_genes <- sample(train_genes, num_sample_genes)
        subset_idx = rng.choice(n_genes, size=min(cfg.inner_subset_size, n_genes),
                                replace=False)
        Xs = X_train[subset_idx]
        ys = y_train[subset_idx]

        if feat_cols:
            base = np.column_stack([col[subset_idx] for col in feat_cols])
        else:
            base = np.empty((len(subset_idx), 0), dtype=int)

        min_d = MIN_DIST
        max_d = _max_dist(n_bins)
        optimizer_fn = get_optimizer(cfg.optimizer)

        # Run PSO once per histone mark, then pick the best pattern across all marks.
        # This replaces the joint HM+shape search that caused PSO to always lock onto
        # H3K4me3 and never explore the other 4 markers.
        best_hm_score  = -1.0
        best_hm_result = None
        best_hm_idx    = 0

        for hm_idx in range(n_hm):

            def objective(vec, _hm=hm_idx):
                # if(!between((floor(pars[1])-floor(pars[2]))^2, min_distance, max_distance)){return(-Inf)}
                start = int(np.floor(vec[0]))
                end   = int(np.floor(vec[1]))
                if not (min_d <= (start - end) ** 2 <= max_d):
                    return 0.5

                pat = decode(vec, _hm, n_bins)
                new_col = pattern_frequencies(Xs, pat).reshape(-1, 1)
                X_cand = np.hstack([base, new_col]) if base.size else new_col
                half = X_cand.shape[0] // 2
                return _train_xgb_auc(X_cand[:half], ys[:half],
                                      X_cand[half:], ys[half:], cfg)

            result = optimizer_fn(
                objective,
                bounds=bounds_for(n_bins),
                n_particles=cfg.pso_particles,
                max_iter=cfg.pso_iters,
                patience=cfg.pso_patience,
                target_score=None,
                seed=cfg.seed + round_idx * n_hm + hm_idx,
                verbose=False,
            )

            if result.best_score > best_hm_score:
                best_hm_score  = result.best_score
                best_hm_result = result
                best_hm_idx    = hm_idx

        new_pat = decode(best_hm_result.best_x, best_hm_idx, n_bins)
        new_col_full = pattern_frequencies(X_train, new_pat)
        feat_cols.append(new_col_full)
        patterns.append(new_pat)

        # check full training AUC — training accuracy where feature extraction step stops
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
                  f"best_hm={best_hm_idx}  inner_auc={best_hm_score:.4f}  "
                  f"full_train_auc={full_auc:.4f}  "
                  f"pat={new_pat}  stale={stale}")

        if full_auc >= cfg.target_train_auc:
            if verbose:
                print(f"  training accuracy reached {cfg.target_train_auc}; stop")
            break
        if stale >= cfg.no_improve_patience:
            if verbose:
                print(f"  no improvement for {cfg.no_improve_patience} rounds; stop")
            break

    feature_matrix = np.column_stack(feat_cols) if feat_cols else \
        np.empty((n_genes, 0), dtype=int)
    return patterns, feature_matrix
