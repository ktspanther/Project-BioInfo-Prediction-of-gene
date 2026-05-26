"""
Pattern representation and sliding-window matching.

A Pattern is a short shape template (anchor points) plus:
  - a correlation threshold
  - the histone mark it applies to
  - the genomic region (start_bin, end_bin) to search within

The frequency of a pattern in a gene = number of windows inside [start_bin, end_bin]
where Pearson correlation with the template exceeds the threshold.

This matches the original R code: the PSO finds BOTH the shape AND the region.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Pattern:
    heights: np.ndarray   # anchor point heights, length = pattern width
    threshold: float      # minimum Pearson r to count a match
    hm_index: int         # which histone mark (0-indexed)
    start_bin: int = 0            # start of the region to search (inclusive, 0-indexed)
    end_bin: int | None = None    # end of the region (inclusive); None = full signal

    @property
    def width(self) -> int:
        return len(self.heights)

    def __repr__(self) -> str:
        return (f"Pattern(width={self.width}, hm={self.hm_index}, "
                f"thr={self.threshold:.2f}, region=[{self.start_bin},{self.end_bin}])")


def sliding_pearson(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Pearson r between `template` and every length-w window of `signal`.

    Returns an array of shape (len(signal) - len(template) + 1,).
    Windows with zero variance return 0 (not NaN) so they never count as matches.
    """
    n = len(signal)
    w = len(template)
    if w > n:
        return np.empty(0, dtype=float)

    windows = np.lib.stride_tricks.sliding_window_view(signal, w)  # (n-w+1, w)

    w_means = windows.mean(axis=1)
    w_stds  = windows.std(axis=1)
    t_mean  = template.mean()
    t_std   = template.std()

    centered_dot = (windows - w_means[:, None]) @ (template - t_mean)
    denom = w_stds * t_std * w

    out = np.zeros_like(centered_dot)
    mask = denom > 1e-12
    out[mask] = centered_dot[mask] / denom[mask]
    return out


def pattern_frequency(signal_matrix: np.ndarray, pat: Pattern) -> int:
    """Match count for a single gene.

    signal_matrix: (n_hm, n_bins)
    """
    n_bins = signal_matrix.shape[1]
    lo = pat.start_bin
    hi = (pat.end_bin + 1) if pat.end_bin is not None else n_bins
    track = signal_matrix[pat.hm_index, lo:hi]
    rs = sliding_pearson(track, pat.heights)
    return int(np.sum(rs > pat.threshold))


def pattern_frequencies(signal_tensor: np.ndarray, pat: Pattern) -> np.ndarray:
    """Match counts for all genes at once.

    signal_tensor: (n_genes, n_hm, n_bins)
    Returns: (n_genes,) integer array
    """
    n_genes, _, n_bins = signal_tensor.shape
    lo = pat.start_bin
    hi = (pat.end_bin + 1) if pat.end_bin is not None else n_bins

    tracks = signal_tensor[:, pat.hm_index, lo:hi]  # (n_genes, region_size)
    region_size = tracks.shape[1]
    w = pat.width
    if w > region_size:
        return np.zeros(n_genes, dtype=int)

    windows = np.lib.stride_tricks.sliding_window_view(tracks, w, axis=1)
    w_means = windows.mean(axis=2)
    w_stds  = windows.std(axis=2)
    t_mean  = pat.heights.mean()
    t_std   = pat.heights.std()

    centered = windows - w_means[:, :, None]
    centered_dot = centered @ (pat.heights - t_mean)
    denom = w_stds * t_std * w
    rs = np.zeros_like(centered_dot)
    mask = denom > 1e-12
    rs[mask] = centered_dot[mask] / denom[mask]
    return (rs > pat.threshold).sum(axis=1).astype(int)


def build_feature_matrix(signal_tensor: np.ndarray,
                         patterns: list[Pattern]) -> np.ndarray:
    """Stack pattern frequency columns into (n_genes, n_patterns)."""
    if not patterns:
        return np.empty((signal_tensor.shape[0], 0), dtype=int)
    cols = [pattern_frequencies(signal_tensor, p) for p in patterns]
    return np.column_stack(cols)
