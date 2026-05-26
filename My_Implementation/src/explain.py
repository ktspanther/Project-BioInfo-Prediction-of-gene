"""
Interpretation (the bit the paper sells as "explainability").

The paper uses the R `xgboostExplainer` package, which gives per-prediction
log-odds breakdowns. The Python ecosystem's equivalent is `shap`. To keep
this dependency-light, we use XGBoost's built-in per-prediction contributions
(`pred_contribs=True`), which return the exact tree-SHAP values for free.

This module produces:
  - per-gene waterfall: log-odds contributions of each feature
  - positional importance: feature importance projected onto the ±5000 bp window
  - per-HM correlation distribution: Pearson(pattern frequency, SHAP) per pattern
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb
from scipy.stats import pearsonr

from pattern import Pattern, pattern_frequencies


def shap_contributions(booster: xgb.Booster, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row, per-feature log-odds contributions.

    Returns
    -------
    contribs : (n_rows, n_features) — SHAP values in log-odds space
    bias     : (n_rows,) — the model's bias term (same for all rows)
    """
    d = xgb.DMatrix(X)
    raw = booster.predict(d, pred_contribs=True)
    # XGBoost returns shape (n_rows, n_features + 1); last column is the bias
    contribs = raw[:, :-1]
    bias = raw[:, -1]
    return contribs, bias


# --- positional importance -------------------------------------------------

def per_bin_per_hm_importance(patterns: list[Pattern],
                              feature_importance: np.ndarray,
                              n_bins: int, n_hm: int) -> np.ndarray:
    """Project a per-pattern importance vector onto the (n_hm, n_bins) grid.

    For each pattern p with importance v_p, distribute v_p uniformly across
    the bins it spans whenever it is matched: as a first-order approximation,
    we credit each bin within the pattern's footprint equally — but since
    patterns are spatially invariant (they can match at any position), here
    we instead distribute v_p uniformly across *all* bins for that HM. This
    matches the paper's "relative importance" framing where importance per
    pattern is shared across the full track for that HM.

    A finer-grained version would instead use the *match positions* per gene
    to localize importance — see `localized_importance` below.

    Returns
    -------
    (n_hm, n_bins) array.
    """
    out = np.zeros((n_hm, n_bins), dtype=float)
    for pat, v in zip(patterns, feature_importance):
        out[pat.hm_index] += v / n_bins
    return out


def localized_importance(patterns: list[Pattern],
                         feature_importance: np.ndarray,
                         X_signal: np.ndarray) -> np.ndarray:
    """Localize importance using *where* each pattern actually matches.

    For each gene, count how many times each pattern matches in each bin
    position, weighted by the pattern's feature importance, and average
    across genes. This is closer in spirit to Figure 5A in the paper
    (importance peaks at specific promoter positions).

    Returns
    -------
    (n_hm, n_bins) array of average importance per (HM, bin).
    """
    from pattern import sliding_pearson
    n_genes, n_hm, n_bins = X_signal.shape
    acc = np.zeros((n_hm, n_bins), dtype=float)
    for pat, v in zip(patterns, feature_importance):
        track = X_signal[:, pat.hm_index, :]  # (n_genes, n_bins)
        w = pat.width
        windows = np.lib.stride_tricks.sliding_window_view(track, w, axis=1)
        # centered windows for correlation
        w_means = windows.mean(axis=2, keepdims=True)
        w_std = windows.std(axis=2)
        t = pat.heights
        t_centered = t - t.mean()
        centered_dot = (windows - w_means).reshape(-1, w) @ t_centered
        centered_dot = centered_dot.reshape(n_genes, -1)
        denom = w_std * t.std() * w
        rs = np.zeros_like(centered_dot)
        mask = denom > 1e-12
        rs[mask] = centered_dot[mask] / denom[mask]
        matches = (rs > pat.threshold).astype(float)  # (n_genes, n_bins-w+1)
        # spread match credit across the bins the pattern covers
        for offset in range(w):
            acc[pat.hm_index, offset:offset + matches.shape[1]] += v * matches.mean(axis=0)
    return acc / max(len(patterns), 1)


# --- per-HM net effect (the violin plot from the paper) --------------------

def per_pattern_correlation(patterns: list[Pattern],
                            feat_matrix: np.ndarray,
                            shap_values: np.ndarray) -> list[tuple[int, float]]:
    """Pearson r between pattern frequency and its SHAP contribution.

    Positive r → pattern presence pushes prediction toward "expressed".
    Returns list of (hm_index, r) tuples, one per pattern.
    """
    out: list[tuple[int, float]] = []
    for i, pat in enumerate(patterns):
        x = feat_matrix[:, i].astype(float)
        s = shap_values[:, i]
        if x.std() < 1e-9 or s.std() < 1e-9:
            out.append((pat.hm_index, 0.0))
            continue
        r, _ = pearsonr(x, s)
        out.append((pat.hm_index, float(r)))
    return out
