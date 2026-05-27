# data loading — real DeepChrome CSVs or synthetic data for testing
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

HM_NAMES = ["H3K27me3", "H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3"]


@dataclass
class Dataset:
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
    def n_bins(self):
        return self.X_train.shape[2]

    @property
    def n_hm(self):
        return self.X_train.shape[1]


def load_deepchrome_csv(path):
    # Load RNA_Seq data (binarize and sort it — same as R code)
    cols = ["GeneID", "BinID"] + HM_NAMES + ["Label"]
    df = pd.read_csv(path, header=None, names=cols)

    n_bins = df["BinID"].max()
    df = (df.sort_values(["GeneID", "BinID"])
            .drop_duplicates(subset=["GeneID", "BinID"], keep="first")
            .reset_index(drop=True))

    gene_ids = df["GeneID"].unique()
    n_genes = len(gene_ids)

    sig = df[HM_NAMES].to_numpy(dtype=np.float32).reshape(n_genes, n_bins, len(HM_NAMES))
    X = sig.transpose(0, 2, 1)  # → (n_genes, n_hm, n_bins)

    y = df.groupby("GeneID", sort=False)["Label"].first().to_numpy().astype(int)
    return X, y, gene_ids


def split_balanced(X, y, gene_ids, n_train=6600, n_val=5911, n_test=5910, seed=42):
    # Train, validation and test subsets — matches the R paper's split sizes
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def take(needed):
        nonlocal pos, neg
        half = needed // 2
        extra = needed - 2 * half
        p_take, pos = pos[:half + extra], pos[half + extra:]
        n_take, neg = neg[:half], neg[half:]
        idx = np.concatenate([p_take, n_take])
        rng.shuffle(idx)
        return idx

    train_idx = take(n_train)
    val_idx   = take(n_val)
    test_idx  = take(n_test)

    return Dataset(
        X_train=X[train_idx], y_train=y[train_idx],
        X_val=X[val_idx],     y_val=y[val_idx],
        X_test=X[test_idx],   y_test=y[test_idx],
        gene_ids_train=gene_ids[train_idx],
        gene_ids_val=gene_ids[val_idx],
        gene_ids_test=gene_ids[test_idx],
    )


def load_presplit_csvs(train_path, valid_path, test_path):
    # load pre-split CSVs from the DeepChrome Zenodo archive
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


def make_synthetic(n_genes=18000, n_bins=100, n_hm=5, seed=0):
    # not in the paper — I added this to test the pipeline without real data
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 0.3, size=(n_genes, n_hm, n_bins)).astype(np.float32)
    y = rng.integers(0, 2, size=n_genes)

    bump = np.array([0.2, 0.6, 1.0, 0.6, 0.2], dtype=np.float32)
    dip  = np.array([1.0, 0.6, 0.2, 0.6, 1.0], dtype=np.float32)
    width = bump.size

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
