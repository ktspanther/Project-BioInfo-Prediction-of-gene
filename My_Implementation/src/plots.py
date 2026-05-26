"""
Plotting utilities that reproduce the paper's signature figures.

Each function returns a matplotlib Figure that the caller can save or
display. The aim is to give you ready-made slides for the class talk.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from data import HM_NAMES
from pattern import Pattern
from explain import (per_bin_per_hm_importance, localized_importance,
                     per_pattern_correlation)


def _bin_to_bp(bin_idx: int, n_bins: int, bp_per_bin: int) -> int:
    """Convert a bin index to bp offset from the TSS (TSS is at the centre)."""
    return (bin_idx - n_bins // 2) * bp_per_bin


def plot_pattern(pat: Pattern, ax=None):
    """Visualise a single pattern (the small line in Figure 2C of the paper)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 2.2))
    else:
        fig = ax.figure
    ax.plot(range(1, pat.width + 1), pat.heights, marker="o", color="#e76f51")
    ax.set_xlabel("Position in pattern")
    ax.set_ylabel("Height")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{HM_NAMES[pat.hm_index]} (thr={pat.threshold:.2f})")
    fig.tight_layout()
    return fig


def plot_positional_importance(patterns: Sequence[Pattern],
                               feat_importance: np.ndarray,
                               n_bins: int,
                               X_signal_sample: np.ndarray | None = None,
                               bp_per_bin: int = 100):
    """Figure 5A analogue: relative importance per bin and per HM around the TSS.

    If `X_signal_sample` is provided, we use it to compute localised importance
    (where patterns actually match), which matches the paper's figure more
    closely. Otherwise we fall back to uniform distribution across the track.
    """
    n_hm = len(HM_NAMES)
    if X_signal_sample is not None:
        imp = localized_importance(list(patterns), feat_importance, X_signal_sample)
    else:
        imp = per_bin_per_hm_importance(list(patterns), feat_importance, n_bins, n_hm)

    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.array([_bin_to_bp(i, n_bins, bp_per_bin) for i in range(n_bins)])
    palette = ["#d62728", "#bcbd22", "#2ca02c", "#1f77b4", "#e377c2"]
    for hm_idx, name in enumerate(HM_NAMES):
        ax.plot(xs, imp[hm_idx], label=name, color=palette[hm_idx], linewidth=1.8)
    # aggregated
    ax.plot(xs, imp.sum(axis=0), label="Aggregated", color="black",
            linewidth=2.2, linestyle="--")
    ax.axvline(0, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("Position relative to TSS (bp)")
    ax.set_ylabel("Relative importance")
    ax.set_title("Positional importance per histone modification")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_per_hm_correlation(patterns: Sequence[Pattern],
                            feat_matrix: np.ndarray,
                            shap_values: np.ndarray):
    """Figure 5B analogue: distribution of Pearson(freq, SHAP) per HM.

    Patterns whose frequency is positively correlated with positive SHAP push
    the prediction toward 'expressed'; negatively correlated → toward 'silent'.
    """
    corrs = per_pattern_correlation(list(patterns), feat_matrix, shap_values)
    by_hm: dict[int, list[float]] = {i: [] for i in range(len(HM_NAMES))}
    for hm_idx, r in corrs:
        by_hm[hm_idx].append(r)

    fig, ax = plt.subplots(figsize=(7, 4))
    data = [by_hm[i] for i in range(len(HM_NAMES))]
    labels = [f"{HM_NAMES[i]}\n(n={len(by_hm[i])})" for i in range(len(HM_NAMES))]
    parts = ax.violinplot(
        [d if len(d) > 0 else [0.0] for d in data],
        showmeans=False, showmedians=True
    )
    ax.set_xticks(range(1, len(HM_NAMES) + 1))
    ax.set_xticklabels(labels)
    ax.axhline(0, color="grey", linestyle=":", linewidth=1)
    ax.set_ylabel(r"Pearson $\rho$ between pattern frequency and SHAP")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Net effect of patterns, grouped by histone modification")
    fig.tight_layout()
    return fig


def plot_waterfall(booster: xgb.Booster, X_row: np.ndarray, patterns: Sequence[Pattern],
                   gene_label: str = "gene"):
    """Figure 6 analogue: per-gene contribution waterfall.

    `X_row` is the (n_features,) feature vector for one gene.
    """
    d = xgb.DMatrix(X_row.reshape(1, -1))
    raw = booster.predict(d, pred_contribs=True)[0]  # (n_features + 1,)
    contribs = raw[:-1]
    bias = raw[-1]

    # sort by absolute contribution, descending
    order = np.argsort(-np.abs(contribs))
    sorted_contribs = contribs[order]
    sorted_labels = [
        f"P{i + 1} ({HM_NAMES[patterns[i].hm_index]}) = {int(X_row[i])}"
        for i in order
    ]

    # cumulative log-odds → probability
    def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

    cum = bias + np.cumsum(sorted_contribs)
    cum = np.concatenate([[bias], cum])
    probs = sigmoid(cum)

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(contribs)), 4))
    xs = np.arange(len(sorted_contribs))
    colors = ["#1f77b4" if c > 0 else "#d62728" for c in sorted_contribs]
    # bars represent the per-step *change* in probability
    bottoms = probs[:-1]
    heights = probs[1:] - probs[:-1]
    ax.bar(xs, heights, bottom=bottoms, color=colors, edgecolor="black")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(sorted_labels, rotation=75, fontsize=8, ha="right")
    ax.set_ylabel("Probability of high expression")
    ax.set_title(f"Per-feature contribution waterfall — {gene_label} "
                 f"(final prob = {probs[-1]:.3f})")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def plot_auc_comparison(our_auc: float,
                        deepchrome_auc: float = 0.8008,
                        shallowchrome_auc: float = 0.8737):
    """Comparison bar chart against the two baselines mentioned in the paper."""
    labels = ["DeepChrome", "ShallowChrome", "PatternChrome (ours)"]
    values = [deepchrome_auc, shallowchrome_auc, our_auc]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=["#f4a261", "#2a9d8f", "#264653"])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Test AUC: this run vs published baselines")
    fig.tight_layout()
    return fig
