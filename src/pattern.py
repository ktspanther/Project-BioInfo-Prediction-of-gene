# Pattern class and sliding-window matching
# matches the mp / cor() logic from PatternChrome.R's Objective function
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Pattern:
    heights: np.ndarray   # mp <- pars[6:(5+floor(pars[5]))]
    threshold: float      # pars[4]
    hm_index: int         # floor(pars[3]) — which histone mark
    start_bin: int = 0            # floor(pars[1])
    end_bin: int | None = None    # floor(pars[2])

    @property
    def width(self):
        return len(self.heights)

    def __repr__(self):
        return (f"Pattern(width={self.width}, hm={self.hm_index}, "
                f"thr={self.threshold:.2f}, region=[{self.start_bin},{self.end_bin}])")


def sliding_pearson(signal, template):
    # cor(mp, hm_train[g, pos:(pos+length(mp)-1)]) for all positions at once
    n = len(signal)
    w = len(template)
    if w > n:
        return np.empty(0, dtype=float)

    windows = np.lib.stride_tricks.sliding_window_view(signal, w)
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


def pattern_frequency(signal_matrix, pat):
    # sum(sapply(checked_positions, function(pos){ cor(...) > pars[4] }), na.rm=T)
    n_bins = signal_matrix.shape[1]
    lo = pat.start_bin
    hi = (pat.end_bin + 1) if pat.end_bin is not None else n_bins
    track = signal_matrix[pat.hm_index, lo:hi]
    rs = sliding_pearson(track, pat.heights)
    return int(np.sum(rs > pat.threshold))


def pattern_frequencies(signal_tensor, pat):
    # vectorised version of pattern_frequency over all genes
    n_genes, _, n_bins = signal_tensor.shape
    lo = pat.start_bin
    hi = (pat.end_bin + 1) if pat.end_bin is not None else n_bins

    # checked_positions <- 1:(ncol(hm_train) - floor(pars[5]) + 1)
    tracks = signal_tensor[:, pat.hm_index, lo:hi]
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


def build_feature_matrix(signal_tensor, patterns):
    if not patterns:
        return np.empty((signal_tensor.shape[0], 0), dtype=int)
    cols = [pattern_frequencies(signal_tensor, p) for p in patterns]
    return np.column_stack(cols)
