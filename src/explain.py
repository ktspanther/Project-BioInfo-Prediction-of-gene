# Python equivalent of the xgboostExplainer logic from Custom_functions.R
# instead of rebuilding the tree manually, we use XGBoost's pred_contribs=True
from __future__ import annotations

import numpy as np
import xgboost as xgb
from scipy.stats import pearsonr

from pattern import Pattern, pattern_frequencies


def shap_contributions(booster, X):
    # same idea as explainPredictions() in Custom_functions.R
    d = xgb.DMatrix(X)
    raw = booster.predict(d, pred_contribs=True)
    contribs = raw[:, :-1]   # last column is the bias term
    bias     = raw[:, -1]
    return contribs, bias


def per_bin_per_hm_importance(patterns, feature_importance, n_bins, n_hm):
    # spread each pattern's importance across its HM track
    out = np.zeros((n_hm, n_bins), dtype=float)
    for pat, v in zip(patterns, feature_importance):
        out[pat.hm_index] += v / n_bins
    return out


def localized_importance(patterns, feature_importance, X_signal):
    # weight importance by where the pattern actually matches in each gene
    n_genes, n_hm, n_bins = X_signal.shape
    acc = np.zeros((n_hm, n_bins), dtype=float)
    for pat, v in zip(patterns, feature_importance):
        track = X_signal[:, pat.hm_index, :]
        w = pat.width
        windows = np.lib.stride_tricks.sliding_window_view(track, w, axis=1)
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
        matches = (rs > pat.threshold).astype(float)
        for offset in range(w):
            acc[pat.hm_index, offset:offset + matches.shape[1]] += v * matches.mean(axis=0)
    return acc / max(len(patterns), 1)


def per_pattern_correlation(patterns, feat_matrix, shap_values):
    # Pearson r between pattern frequency and its SHAP value — positive = promotes expression
    out = []
    for i, pat in enumerate(patterns):
        x = feat_matrix[:, i].astype(float)
        s = shap_values[:, i]
        if x.std() < 1e-9 or s.std() < 1e-9:
            out.append((pat.hm_index, 0.0))
            continue
        r, _ = pearsonr(x, s)
        out.append((pat.hm_index, float(r)))
    return out
