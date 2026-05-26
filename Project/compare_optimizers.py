"""
Optimizer comparison study.

This is the SCIENTIFIC CONTRIBUTION beyond just re-running PatternChrome's
pipeline: a head-to-head comparison of three optimizers slotted into the
same feature-extraction step.

Question
--------
Is PSO essential to PatternChrome's success, or would any reasonable
gradient-free optimizer give comparable AUC?

Experimental design
-------------------
- Fix everything except the optimizer: same data split, same bin width,
  same XGBoost settings, same backward elimination, same evaluation
  metric.
- Run the full pipeline 3 times per cell line, once with each optimizer
  ("pso", "de", "random").
- Same compute budget for each: same n_particles * max_iter, so each
  optimizer evaluates the objective the same number of times.
- Same seed for reproducibility, but different per-round seeds so we
  don't trivially get identical patterns.

Outputs
-------
results/optimizer_comparison_<cell>.png
    Bar chart of test AUC, validation AUC, runtime, and feature count
    for each optimizer.
results/optimizer_comparison_<cell>.csv
    Raw numbers for the article's results table.
results/optimizer_history_<cell>.png
    Best-score-so-far curve over wall time, one line per optimizer.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import json
from dataclasses import asdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_presplit_csvs, make_synthetic, split_balanced
from feature_extraction import FEConfig
from pipeline import run_pipeline


# Color palette consistent across all plots
COLORS = {
    "pso":    "#264653",   # dark teal — PatternChrome's choice
    "de":     "#e76f51",   # warm coral — the challenger
    "random": "#999999",   # neutral grey — the sanity baseline
}

LABELS = {
    "pso":    "PSO (paper's choice)",
    "de":     "Differential Evolution",
    "random": "Random search (baseline)",
}


def parse_args():
    p = argparse.ArgumentParser(description="Compare optimizers inside the PatternChrome pipeline")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true")
    src.add_argument("--train-csv", type=str)
    p.add_argument("--valid-csv", type=str)
    p.add_argument("--test-csv",  type=str)
    p.add_argument("--cell",     type=str, default="sample")
    p.add_argument("--out",      type=str, default="results")
    p.add_argument("--n-patterns",    type=int, default=10)
    p.add_argument("--pso-particles", type=int, default=12)
    p.add_argument("--pso-iters",     type=int, default=15)
    p.add_argument("--inner-subset",  type=int, default=1500)
    p.add_argument("--seed",          type=int, default=0)
    p.add_argument("--optimizers",    type=str, default="pso,de,random",
                   help="Comma-separated list of optimizers to compare")
    return p.parse_args()


def load_ds(args):
    if args.synthetic:
        X, y, gids = make_synthetic(n_genes=18000, n_bins=100, seed=args.seed)
        return split_balanced(X, y, gids, seed=args.seed)
    if not (args.valid_csv and args.test_csv):
        raise SystemExit("--train-csv requires --valid-csv and --test-csv")
    return load_presplit_csvs(args.train_csv, args.valid_csv, args.test_csv)


def run_one(ds, optimizer: str, args) -> dict:
    """Run the full pipeline once with the given optimizer; return summary dict."""
    print(f"\n{'='*60}")
    print(f"  Running pipeline with optimizer = '{optimizer}'")
    print(f"{'='*60}")
    cfg = FEConfig(
        n_patterns_max=args.n_patterns,
        pso_particles=args.pso_particles,
        pso_iters=args.pso_iters,
        inner_subset_size=args.inner_subset,
        seed=args.seed,
        optimizer=optimizer,
    )
    t0 = time.time()
    res = run_pipeline(ds, fe_cfg=cfg, verbose=True)
    wall = time.time() - t0
    summary = {
        "optimizer":   optimizer,
        "test_auc":    float(res.test_auc),
        "val_auc":     float(res.val_auc),
        "train_auc":   float(res.train_auc),
        "n_patterns":  len(res.patterns),
        "wall_time_s": float(wall),
        "timings":     {k: float(v) for k, v in res.timings.items()},
        "hp":          {k: float(v) for k, v in res.hp.items()},
    }
    print(f"\n  → test_auc = {res.test_auc:.4f}  |  "
          f"patterns = {len(res.patterns)}  |  wall = {wall:.1f}s")
    return summary


def plot_comparison(rows: list[dict], cell: str, out_dir: str):
    """Four-panel comparison figure: test AUC, val AUC, runtime, n features."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    names = [r["optimizer"] for r in rows]
    colors = [COLORS.get(n, "#888888") for n in names]
    nice_labels = [LABELS.get(n, n) for n in names]

    panels = [
        ("test_auc",    "Test AUC",        "AUC",           (0.5, 1.0)),
        ("val_auc",     "Validation AUC",  "AUC",           (0.5, 1.0)),
        ("wall_time_s", "Wall time",       "Seconds",       None),
        ("n_patterns",  "Patterns kept",   "Number",        None),
    ]

    for ax, (key, title, ylabel, ylim) in zip(axes, panels):
        vals = [r[key] for r in rows]
        bars = ax.bar(range(len(rows)), vals, color=colors,
                      edgecolor="black", linewidth=0.4)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(nice_labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        for b, v in zip(bars, vals):
            fmt = f"{v:.3f}" if isinstance(v, float) and v < 10 else f"{v:.0f}"
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + (max(vals) * 0.01 if max(vals) > 0 else 0.01),
                    fmt, ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Optimizer comparison inside PatternChrome — cell line {cell}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, f"optimizer_comparison_{cell}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_runtime_breakdown(rows: list[dict], cell: str, out_dir: str):
    """Stacked bar of per-stage runtimes."""
    stages = ["feature_extraction", "backward_elimination",
              "hyperparameter_tuning", "final_training"]
    nice_stages = ["Feature\nextraction", "Backward\nelimination",
                   "Hyperparameter\ntuning", "Final\ntraining"]
    stage_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottom = np.zeros(len(rows))
    x = np.arange(len(rows))
    width = 0.6

    for stage, label, color in zip(stages, nice_stages, stage_colors):
        vals = np.array([r["timings"].get(stage, 0) for r in rows])
        ax.bar(x, vals, bottom=bottom, label=label, color=color,
               edgecolor="white", linewidth=0.6, width=width)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(r["optimizer"], r["optimizer"]) for r in rows],
                       rotation=15, ha="right")
    ax.set_ylabel("Wall time (seconds)")
    ax.set_title(f"Per-stage runtime breakdown — {cell}")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    for i, total in enumerate(bottom):
        ax.text(i, total + total * 0.01, f"{total:.0f}s",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, f"optimizer_runtime_{cell}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def save_csv(rows: list[dict], cell: str, out_dir: str):
    """Flatten and save for the article's results table."""
    flat = []
    for r in rows:
        flat.append({
            "optimizer":   r["optimizer"],
            "test_auc":    r["test_auc"],
            "val_auc":     r["val_auc"],
            "train_auc":   r["train_auc"],
            "n_patterns":  r["n_patterns"],
            "wall_time_s": r["wall_time_s"],
            "feat_extr_s": r["timings"].get("feature_extraction", 0),
            "backward_elim_s":  r["timings"].get("backward_elimination", 0),
            "hp_tune_s":   r["timings"].get("hyperparameter_tuning", 0),
            "final_train_s": r["timings"].get("final_training", 0),
        })
    df = pd.DataFrame(flat)
    path = os.path.join(out_dir, f"optimizer_comparison_{cell}.csv")
    df.to_csv(path, index=False, float_format="%.4f")
    print(f"  saved {path}")
    # also a pretty-printed summary text file
    summary_path = os.path.join(out_dir, f"optimizer_comparison_{cell}.txt")
    with open(summary_path, "w") as f:
        f.write(f"Optimizer comparison — cell line {cell}\n")
        f.write("=" * 60 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n")
    print(f"  saved {summary_path}")


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    ds = load_ds(args)
    print(f"[data] train={len(ds.y_train)}  val={len(ds.y_val)}  test={len(ds.y_test)}")

    optimizers = [s.strip() for s in args.optimizers.split(",") if s.strip()]
    print(f"[study] comparing optimizers: {optimizers}")

    rows: list[dict] = []
    for opt in optimizers:
        rows.append(run_one(ds, opt, args))

    # Save outputs
    print(f"\n[study] writing summary to {args.out}/")
    save_csv(rows, args.cell, args.out)
    plot_comparison(rows, args.cell, args.out)
    plot_runtime_breakdown(rows, args.cell, args.out)

    # Print a final table to stdout for the article
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS — cell line {args.cell}")
    print(f"{'='*70}")
    print(f"  {'optimizer':22s}  {'test AUC':>10s}  {'val AUC':>10s}  "
          f"{'#pat':>6s}  {'time':>8s}")
    print("  " + "-" * 66)
    for r in rows:
        print(f"  {LABELS.get(r['optimizer'], r['optimizer']):22s}  "
              f"{r['test_auc']:10.4f}  {r['val_auc']:10.4f}  "
              f"{r['n_patterns']:6d}  {r['wall_time_s']:7.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
