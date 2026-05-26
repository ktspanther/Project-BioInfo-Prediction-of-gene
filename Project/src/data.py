"""
Data loading.

Two paths:

1) Real DeepChrome data (recommended for the demo). Each file is a CSV with
   columns: GeneID, BinID, H3K27me3, H3K36me3, H3K4me1, H3K4me3, H3K9me3, Label.
   Each gene has 100 rows (100 bins × 100bp = ±5000bp around the TSS).
   Download e.g. from https://zenodo.org/record/2652278

   PatternChrome paper uses 200 bins × 50bp instead of 100 × 100bp. The pipeline
   logic doesn't change. To reproduce the paper exactly you'd need to regenerate
   bins from the raw .bam files (see DeepChrome README), which is overkill for
   a class demo.

2) Synthetic data with planted patterns. Useful when you don't have the data
   yet and want to verify the pipeline runs and recovers known signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

HM_NAMES = ["H3K27me3", "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3"]


@dataclass
class Dataset:
    """Container for one cell line's preprocessed data."""
    X_train: np.ndarray  # (n_train, n_hm, n_bins)
    y_train: np.ndarray  # (n_train,) 0/1
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    gene_ids_train: np.ndarray
    gene_ids_val: np.ndarray
    gene_ids_test: np.ndarray

    @property
    def n_bins(self) -> int:
        return self.X_train.shape[2]

    @property
    def n_hm(self) -> int:
        return self.X_train.shape[1]


def load_deepchrome_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one cell line CSV in DeepChrome format.

    Returns
    -------
    X : (n_genes, n_hm=5, n_bins=100) float32
    y : (n_genes,) 0/1 int
    gene_ids : (n_genes,) array
    """
    cols = ["GeneID", "BinID"] + HM_NAMES + ["Label"]
    df = pd.read_csv(path, header=None, names=cols)

    n_bins = df["BinID"].max()  # 100 in DeepChrome's prepared data
    # Sort to a known order, drop duplicate (GeneID, BinID) rows, then reshape
    df = (df.sort_values(["GeneID", "BinID"])
            .drop_duplicates(subset=["GeneID", "BinID"], keep="first")
            .reset_index(drop=True))

    gene_ids = df["GeneID"].unique()
    n_genes = len(gene_ids)

    # (n_genes, n_bins, n_hm)
    sig = df[HM_NAMES].to_numpy(dtype=np.float32).reshape(n_genes, n_bins, len(HM_NAMES))
    # Transpose to (n_genes, n_hm, n_bins)
    X = sig.transpose(0, 2, 1)

    # Label is per-gene (constant across the gene's bins). Take the first row of each.
    y = df.groupby("GeneID", sort=False)["Label"].first().to_numpy().astype(int)
    return X, y, gene_ids


def split_balanced(X: np.ndarray, y: np.ndarray, gene_ids: np.ndarray,
                   n_train: int = 6600, n_val: int = 5911, n_test: int = 5910,
                   seed: int = 42) -> Dataset:
    """Random class-balanced split mirroring the paper (6600 / 5911 / 5910).

    To keep classes balanced in each split, we sample half from y==1 and half
    from y==0. If the requested size is odd we sample one extra from class 1.
    """
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def take(needed):
        nonlocal pos, neg
        half = needed // 2
        extra = needed - 2 * half  # 0 or 1
        p_take, pos = pos[:half + extra], pos[half + extra:]
        n_take, neg = neg[:half], neg[half:]
        idx = np.concatenate([p_take, n_take])
        rng.shuffle(idx)
        return idx

    train_idx = take(n_train)
    val_idx = take(n_val)
    test_idx = take(n_test)

    return Dataset(
        X_train=X[train_idx], y_train=y[train_idx],
        X_val=X[val_idx], y_val=y[val_idx],
        X_test=X[test_idx], y_test=y[test_idx],
        gene_ids_train=gene_ids[train_idx],
        gene_ids_val=gene_ids[val_idx],
        gene_ids_test=gene_ids[test_idx],
    )


def load_presplit_csvs(train_path: str | Path,
                       valid_path: str | Path,
                       test_path:  str | Path) -> Dataset:
    """Load three pre-split CSVs — the format DeepChrome's Zenodo archive uses.

    The archive stores each cell line as:
        <cell>/classification/train.csv
        <cell>/classification/valid.csv
        <cell>/classification/test.csv

    No further splitting is performed. This is the correct way to load your
    data/E003/classification/ and data/E123/classification/ folders.
    """
    print(f"  train : {train_path}")
    X_tr, y_tr, gids_tr = load_deepchrome_csv(train_path)
    print(f"  valid : {valid_path}")
    X_va, y_va, gids_va = load_deepchrome_csv(valid_path)
    print(f"  test  : {test_path}")
    X_te, y_te, gids_te = load_deepchrome_csv(test_path)
    print(f"  shapes — train {X_tr.shape}  val {X_va.shape}  test {X_te.shape}")
    return Dataset(
        X_train=X_tr, y_train=y_tr,
        X_val=X_va,   y_val=y_va,
        X_test=X_te,  y_test=y_te,
        gene_ids_train=gids_tr,
        gene_ids_val=gids_va,
        gene_ids_test=gids_te,
    )


def make_synthetic(n_genes: int = 18000, n_bins: int = 100, n_hm: int = 5,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic dataset with planted patterns.

    Positive genes: a "bump" pattern (low-high-low) is added to HM index 3
    (mimicking H3K4me3) in the promoter region around bin 50.
    Negative genes: a "dip" pattern in the same place for HM index 0
    (mimicking H3K27me3).
    Gaussian noise everywhere. The pipeline should recover bumps near bin 50
    in HM 3 as the dominant feature.

    Use this when you don't yet have real REMC data, to validate that the
    pipeline runs end-to-end and learns something sensible.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 0.3, size=(n_genes, n_hm, n_bins)).astype(np.float32)
    y = rng.integers(0, 2, size=n_genes)

    bump = np.array([0.2, 0.6, 1.0, 0.6, 0.2], dtype=np.float32)
    dip = np.array([1.0, 0.6, 0.2, 0.6, 1.0], dtype=np.float32)
    width = bump.size

    # Plant in a noisy spot near the centre, with small location jitter.
    centre = n_bins // 2
    jitter = max(1, n_bins // 20)
    lo = max(0, centre - jitter)
    hi = max(lo + 1, min(n_bins - width, centre + jitter))
    for i in range(n_genes):
        offset = int(rng.integers(lo, hi + 1))
        if y[i] == 1:
            X[i, 3, offset:offset + width] += bump * rng.uniform(0.5, 1.5)
        else:
            X[i, 0, offset:offset + width] += dip * rng.uniform(0.5, 1.5)

    X = np.clip(X, 0, None)
    gene_ids = np.arange(n_genes)
    return X, y, gene_ids
