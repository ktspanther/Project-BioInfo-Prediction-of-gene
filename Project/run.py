"""Run the full PatternChrome pipeline from the command line.

Three ways to call it
---------------------
# 1) Synthetic smoke test (no data needed, ~2 min):
python run.py --synthetic

# 2) Real data already split — matches data/E003/classification/ layout:
python run.py --train-csv data/E003/classification/train.csv \\
              --valid-csv data/E003/classification/valid.csv \\
              --test-csv  data/E003/classification/test.csv  \\
              --cell E003

# 3) Real data as a single CSV (will be split automatically):
python run.py --csv data/E003.csv --cell E003

# Tweak speed/quality:
python run.py --synthetic --n-patterns 12 --pso-iters 20 --pso-particles 12
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import (load_deepchrome_csv, load_presplit_csvs,
                  make_synthetic, split_balanced, Dataset)
from feature_extraction import FEConfig
from pipeline import run_pipeline
from plots import (plot_positional_importance, plot_per_hm_correlation,
                   plot_waterfall, plot_auc_comparison, plot_pattern)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PatternChrome pipeline runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true",
                     help="Generate synthetic data with planted patterns")
    src.add_argument("--csv", type=str, metavar="PATH",
                     help="Single DeepChrome-format CSV — auto-split")
    src.add_argument("--train-csv", type=str, metavar="PATH",
                     help="Pre-split train CSV (use with --valid-csv + --test-csv)")

    p.add_argument("--valid-csv", type=str, metavar="PATH")
    p.add_argument("--test-csv",  type=str, metavar="PATH")

    p.add_argument("--cell",  type=str, default="sample")
    p.add_argument("--out",   type=str, default="results")
    p.add_argument("--seed",  type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--n-patterns",    type=int, default=15)
    p.add_argument("--pso-particles", type=int, default=15)
    p.add_argument("--pso-iters",     type=int, default=25)
    p.add_argument("--inner-subset",  type=int, default=2000)
    return p


def load_data(args) -> Dataset:
    if args.synthetic:
        print("[data] synthetic dataset")
        X, y, gids = make_synthetic(n_genes=18000, n_bins=100, seed=args.seed)
        ds = split_balanced(X, y, gids, seed=args.seed)

    elif args.train_csv:
        if not args.valid_csv or not args.test_csv:
            raise ValueError("--train-csv requires --valid-csv and --test-csv")
        print(f"[data] loading pre-split CSVs for {args.cell}")
        ds = load_presplit_csvs(args.train_csv, args.valid_csv, args.test_csv)

    else:
        print(f"[data] loading {args.csv}")
        X, y, gids = load_deepchrome_csv(args.csv)
        ds = split_balanced(X, y, gids, seed=args.seed)

    print(f"[data] train={len(ds.y_train)}  val={len(ds.y_val)}  "
          f"test={len(ds.y_test)}  balance={ds.y_train.mean():.3f}")
    return ds


def save_plots(res, ds: Dataset, args) -> None:
    out, cell = args.out, args.cell
    print(f"\n[plots] writing to {out}/")

    fig = plot_auc_comparison(res.test_auc)
    fig.savefig(os.path.join(out, f"{cell}_auc_compare.png"), dpi=150)
    plt.close(fig)

    avg_abs_shap = np.abs(res.shap_test).mean(axis=0)
    fig = plot_positional_importance(
        res.patterns, avg_abs_shap, n_bins=ds.n_bins,
        X_signal_sample=ds.X_test[:200], bp_per_bin=10000 // ds.n_bins)
    fig.savefig(os.path.join(out, f"{cell}_positional_importance.png"), dpi=150)
    plt.close(fig)

    fig = plot_per_hm_correlation(res.patterns, res.X_test_feat, res.shap_test)
    fig.savefig(os.path.join(out, f"{cell}_per_hm_correlation.png"), dpi=150)
    plt.close(fig)

    pos_idx = int(np.where(ds.y_test == 1)[0][0])
    neg_idx = int(np.where(ds.y_test == 0)[0][0])
    for idx, label in [(pos_idx, "high"), (neg_idx, "low")]:
        fig = plot_waterfall(res.booster, res.X_test_feat[idx],
                             res.patterns, gene_label=f"{cell}: {label}-expr gene")
        fig.savefig(os.path.join(out, f"{cell}_waterfall_{label}.png"), dpi=150)
        plt.close(fig)

    n_pat = len(res.patterns)
    if n_pat > 0:
        cols = min(5, n_pat)
        rows = (n_pat + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(2.6 * cols, 2.0 * rows), squeeze=False)
        for i, pat in enumerate(res.patterns):
            plot_pattern(pat, axes[i // cols][i % cols])
        for j in range(n_pat, rows * cols):
            axes[j // cols][j % cols].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{cell}_patterns.png"), dpi=150)
        plt.close(fig)


def main():
    args = build_parser().parse_args()
    if args.train_csv and (not args.valid_csv or not args.test_csv):
        build_parser().error("--train-csv requires both --valid-csv and --test-csv")

    os.makedirs(args.out, exist_ok=True)
    ds = load_data(args)

    cfg = FEConfig(
        n_patterns_max=args.n_patterns,
        pso_particles=args.pso_particles,
        pso_iters=args.pso_iters,
        inner_subset_size=args.inner_subset,
        seed=args.seed,
    )

    t0 = time.time()
    res = run_pipeline(ds, fe_cfg=cfg, verbose=not args.quiet)
    elapsed = time.time() - t0

    print(f"\n{'─'*50}")
    print(f"  Cell line : {args.cell}")
    print(f"  Test AUC  : {res.test_auc:.4f}")
    print(f"  Patterns  : {len(res.patterns)}")
    print(f"  Wall time : {elapsed:.1f}s")
    print(f"{'─'*50}")

    save_plots(res, ds, args)
    print("[done]")


if __name__ == "__main__":
    main()
