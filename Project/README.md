# PatternChrome — Python Re-Implementation

A Python re-implementation of [PatternChrome (Paul et al., *Bioinformatics* 2025)](https://doi.org/10.1093/bioinformatics/btaf033), built for a CS-student class presentation.

The original paper publishes R code at <https://gitlab.gwdg.de/MedBioinf/generegulation/patternchrome>. This is an independent re-implementation in Python from the paper's description. It mirrors the five-stage pipeline (binning → PSO feature extraction → backward elimination → hyperparameter tuning → final XGBoost) and reproduces the paper's signature plots.

## What's in the box

```
patternchrome/
├── README.md
├── requirements.txt
├── run.py                    # CLI entry point
├── src/
│   ├── data.py               # CSV loader + synthetic-data generator
│   ├── pattern.py            # Pattern class + sliding-Pearson matching
│   ├── pso.py                # Standalone PSO (SPSO2007-inspired)
│   ├── feature_extraction.py # Stage 2: PSO-driven greedy feature discovery
│   ├── backward_elim.py      # Stage 3: drop redundant patterns
│   ├── train.py              # Stage 4: hyperparameter tuning + Stage 5: final XGB
│   ├── explain.py            # SHAP contributions, positional importance
│   ├── plots.py              # Reproductions of Figures 4, 5, 6 from the paper
│   ├── pipeline.py           # Wires all five stages together
│   └── _xgboost_stub.py      # delete after install — used during development only
├── notebooks/                # space for your own demo notebook
├── data/                     # put the cell-line CSVs here
└── results/                  # plots and outputs land here
```

## Quick start

```bash
# 1. Create a fresh environment
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Smoke test on synthetic data (no download required, ~1-2 min)
python run.py --synthetic --cell SYN

# 4. Real run on one cell line (see below for how to get the CSV)
python run.py --csv data/E003.csv --cell E003
```

All figures are written to `results/`.

## Getting the data

The PatternChrome paper uses 56 cell lines from the [Roadmap Epigenomics (REMC) consortium](https://egg2.wustl.edu/roadmap/web_portal/). Each cell line has ChIP-seq tracks for 5 histone modifications (H3K27me3, H3K36me3, H3K4me1, H3K4me3, H3K9me3) and RNA-seq expression.

For a class demo you don't need to start from raw `.bam` files. The [DeepChrome dataset on Zenodo (record 2652278)](https://zenodo.org/record/2652278) provides the same 56 cell lines already binned and labelled in a friendly CSV format:

```
GeneID, BinID, H3K27me3, H3K36me3, H3K4me1, H3K4me3, H3K9me3, Label
```

Each gene has 100 rows (100 bins × 100bp = ±5000bp around the TSS).

**Important caveat:** PatternChrome uses **200 bins of 50bp**, DeepChrome's prepared data uses **100 bins of 100bp**. The pipeline logic is identical — you just lose some spatial resolution. For a class demo this is the right trade-off; reproducing the paper's exact numbers would require regenerating bins from raw BAMs (overkill).

Pick a couple of cell lines from the Zenodo archive (good choices: `E003` embryonic stem cell, `E123` K562 leukemia line — both highlighted in the paper) and drop them in `data/`.

## Running on real data

```bash
python run.py --csv data/E003.csv --cell E003

# A bigger run, closer to paper settings (~20-60 min depending on machine):
python run.py --csv data/E003.csv --cell E003 \
    --n-patterns 30 --pso-particles 30 --pso-iters 40 --inner-subset 3000
```

Expected runtimes on a modern laptop (4-core CPU):
- `--synthetic` smoke test: 1-3 min
- Default real-data run: 10-25 min per cell line
- Paper-comparable settings: 30-90 min per cell line

## The five pipeline stages

### 1. Binning (in `data.py`)

The ±5000 bp window around each TSS is split into N bins; the value of each bin is the mean ChIP-seq signal in that bin. With DeepChrome data, N=100 (100bp bins).

### 2. Feature extraction (in `feature_extraction.py`) — the paper's novel idea

A **pattern** is a short 1D template (e.g. `[0.2, 0.6, 1.0, 0.6, 0.2]`) plus a correlation threshold and a target HM. Its "frequency" in a gene = number of positions where Pearson correlation between the template and the underlying signal window exceeds the threshold.

We greedily build a feature set, one pattern at a time. Each round, PSO searches the continuous parameter space `[width, h1, ..., h_MAX_W, threshold, hm_index]` for the pattern whose addition gives the biggest XGBoost AUC bump on a random training subset.

### 3. Backward elimination (in `backward_elim.py`)

Walk through the feature list from last-added to first-added. Tentatively remove each pattern; if validation AUC stays ≥ previous, drop it permanently. Restart from the end whenever something is dropped.

### 4. Hyperparameter tuning (in `train.py`)

PSO again, this time over XGBoost's hyperparameters (`eta`, `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight`, `num_boost_round`). Objective = validation AUC.

### 5. Final prediction (in `train.py`)

Train one final XGBoost model with tuned hyperparameters on the training set. Report AUC on the held-out test set.

## Interpretation

After Stage 5, `explain.py` computes SHAP contributions for each test gene using XGBoost's native `pred_contribs=True`. This is the Python equivalent of the R `xgboostExplainer` package the paper uses. From there:

- `plot_positional_importance` → Figure 5A analogue (importance peak around the TSS)
- `plot_per_hm_correlation` → Figure 5B analogue (per-HM violin of pattern-vs-SHAP correlation)
- `plot_waterfall` → Figure 6 analogue (per-gene contribution breakdown)
- `plot_auc_comparison` → Figure 4 analogue (your AUC vs DeepChrome/ShallowChrome)
- `plot_pattern` → Figure 2C analogue (the actual shape of a learned pattern)

## Suggested presentation demo

A live demo on stage is tricky — runs are long and stochastic. Better plan:

1. **Pre-run** the pipeline ahead of time on one or two cell lines (e.g. E003 and E123). Save the plots into `results/`.
2. **On stage**, open the notebook or run `--synthetic --n-patterns 4 --pso-iters 5` as a fast live demonstration that the pipeline runs.
3. Use the pre-generated plots to walk through the results: positional importance + per-HM net effect + waterfall plot for one gene.
4. Have one slide showing your AUC vs DeepChrome (0.8008) and ShallowChrome (0.8737).

If you reach AUC > 0.85 on real data with default settings you've done well. AUC > 0.88 puts you near the paper's published numbers.

## Speed/quality knobs (in `feature_extraction.py: FEConfig`)

| Knob | Default | Paper-like | Effect |
|---|---|---|---|
| `n_patterns_max` | 20 | 30-50 | More patterns → better AUC, slower |
| `pso_particles` | 15 | 30 | Bigger swarm → better optima, slower |
| `pso_iters` | 25 | 40 | More iterations → tighter convergence |
| `inner_subset_size` | 2000 | 3000 | Bigger eval subset → less PSO noise |
| `target_train_auc` | 0.999 | 0.999 | Same as paper |

## Deviations from the paper (be honest about these in your talk)

1. **Bin width.** DeepChrome data is 100bp bins, paper uses 50bp bins. Doubling resolution would require regenerating bins from raw BAMs.
2. **PSO topology.** The paper uses SPSO2007 with K=3 random informants per particle. We use the simpler global-best (gbest) topology with the same recommended ω and c constants. On low-dim continuous problems the two perform similarly.
3. **XGBoost explainer.** The paper uses the R `xgboostExplainer` package. We use XGBoost's native `pred_contribs=True`, which gives exact TreeSHAP values. Mathematically equivalent for our purposes.
4. **Stopping budgets.** We use much smaller defaults for the class demo. Increase them via CLI flags for a paper-comparable run.
5. **The synthetic test mode.** Not in the paper — added for fast sandbox testing.

## Troubleshooting

- **`No module named xgboost`** — `pip install xgboost`. On Apple Silicon you may need `pip install xgboost --no-binary=:all:`.
- **`_xgboost_stub.py` still imported** — that file is for development only; you should never need to import it on a real machine. Feel free to delete it.
- **Pipeline says AUC ≈ 0.5** — almost always means the inner subset is too small or PSO budget too small; increase `--inner-subset` and `--pso-iters`.
- **`ValueError: shapes do not align`** — make sure your CSV is in DeepChrome format (8 columns, 100 rows per gene, no header). Sort by GeneID, BinID.
- **Out of memory** — drop `--inner-subset` or reduce `n_patterns_max`. The pattern-frequency computation builds a stride view that is memory-cheap, but XGBoost's `DMatrix` is not.

## References

- Paul N.B., Wolber J.C. et al. *PatternChrome: Prediction of gene expression using histone modification patterns extracted by Particle Swarm Optimization.* Bioinformatics 41(2), 2025.
- Singh R., Lanchantin J. et al. *DeepChrome.* Bioinformatics 32(17), 2016. — for the dataset format and baseline.
- Frasca F., Matteucci M. et al. *ShallowChrome.* BMC Bioinformatics 23, 2022. — for the other baseline.
- Clerc M. *Standard PSO 2007.* In *Innovations and Developments of Swarm Intelligence Applications*, IGI Global, 2012.


Every time you come back to work on the project: 
    conda activate patternchrome
    jupyter lab