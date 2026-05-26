"""
Pattern representation and pattern-matching against histone modification signals.

A pattern is a short 1D template (a "shape") plus a correlation threshold and
the index of the histone modification (HM) it applies to. To compute the
pattern frequency in a gene's signal, slide the template along the chosen
HM track and count positions where Pearson correlation with the underlying
signal window exceeds the threshold.

This is the heart of PatternChrome: features are *shape* counts, not
*amount* sums. That makes them scale-invariant — a small bump or a tall
peak with the same shape both match.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Pattern:
    """A single histone modification shape template.

    Attributes
    ----------
    heights : np.ndarray
        1D array of length `w` (the pattern width), values in [0, 1].
    threshold : float
        Minimum Pearson correlation for a window to count as a match.
    hm_index : int
        Which histone modification (0..n_hm-1) this pattern applies to.
    """
    heights: np.ndarray
    threshold: float
    hm_index: int

    @property
    def width(self) -> int:
        return len(self.heights)

    def __repr__(self) -> str:
        return (f"Pattern(width={self.width}, hm={self.hm_index}, "
                f"thr={self.threshold:.2f})")


def sliding_pearson(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Pearson correlation between `template` and every length-w window of `signal`.

    Parameters
    ----------
    signal : (n_bins,) array
        The histone signal for one gene, one HM.
    template : (w,) array
        The pattern heights.

    Returns
    -------
    (n_bins - w + 1,) array of Pearson r values.

    Implementation
    --------------
    Pearson r between window x and template t is:
        r = sum((x - x̄)(t - t̄)) / (σ_x σ_t * w)

    We vectorize by computing the rolling mean and rolling std of x using
    cumulative sums, then the rolling dot product of x and t. Edge cases
    (zero variance windows) are returned as 0 (not NaN) so they're never
    counted as matches.
    """
    n = len(signal)
    w = len(template)
    if w > n:
        return np.empty(0, dtype=float)

    # Use stride tricks for the rolling windows (memory-cheap view).
    windows = np.lib.stride_tricks.sliding_window_view(signal, w)  # (n-w+1, w)

    # Mean / std per window
    w_means = windows.mean(axis=1)
    w_stds = windows.std(axis=1)
    t_mean = template.mean()
    t_std = template.std()

    # Centered dot product
    centered_dot = (windows - w_means[:, None]) @ (template - t_mean)
    denom = w_stds * t_std * w

    out = np.zeros_like(centered_dot)
    # Guard against zero-variance windows or zero-variance template
    mask = (denom > 1e-12)
    out[mask] = centered_dot[mask] / denom[mask]
    return out


def pattern_frequency(signal_matrix: np.ndarray, pat: Pattern) -> int:
    """How many windows in a single gene match the pattern.

    Parameters
    ----------
    signal_matrix : (n_hm, n_bins) array
        One gene's full HM signal data.
    pat : Pattern
    """
    track = signal_matrix[pat.hm_index]
    rs = sliding_pearson(track, pat.heights)
    return int(np.sum(rs > pat.threshold))


def pattern_frequencies(signal_tensor: np.ndarray, pat: Pattern) -> np.ndarray:
    """Vectorized pattern frequencies across many genes.

    Parameters
    ----------
    signal_tensor : (n_genes, n_hm, n_bins) array

    Returns
    -------
    (n_genes,) integer array of match counts.
    """
    tracks = signal_tensor[:, pat.hm_index, :]  # (n_genes, n_bins)
    n_genes, n_bins = tracks.shape
    w = pat.width
    if w > n_bins:
        return np.zeros(n_genes, dtype=int)

    # All sliding windows for all genes at once: (n_genes, n_bins-w+1, w)
    windows = np.lib.stride_tricks.sliding_window_view(tracks, w, axis=1)
    w_means = windows.mean(axis=2)
    w_stds = windows.std(axis=2)
    t_mean = pat.heights.mean()
    t_std = pat.heights.std()

    centered = windows - w_means[:, :, None]
    centered_dot = centered @ (pat.heights - t_mean)  # (n_genes, n_bins-w+1)
    denom = w_stds * t_std * w
    rs = np.zeros_like(centered_dot)
    mask = denom > 1e-12
    rs[mask] = centered_dot[mask] / denom[mask]
    return (rs > pat.threshold).sum(axis=1).astype(int)


def build_feature_matrix(signal_tensor: np.ndarray,
                         patterns: list[Pattern]) -> np.ndarray:
    """Stack per-pattern frequency columns into a feature matrix.

    Parameters
    ----------
    signal_tensor : (n_genes, n_hm, n_bins)
    patterns : list of Pattern

    Returns
    -------
    (n_genes, n_patterns) array of integer pattern counts.
    """
    if not patterns:
        return np.empty((signal_tensor.shape[0], 0), dtype=int)
    cols = [pattern_frequencies(signal_tensor, p) for p in patterns]
    return np.column_stack(cols)
