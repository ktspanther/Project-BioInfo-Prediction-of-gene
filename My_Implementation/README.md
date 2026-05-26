# PatternChrome — Python re-implementation

Python port of [PatternChrome (Paul et al., Bioinformatics 2025)](https://doi.org/10.1093/bioinformatics/btaf033) done for a Master's class project.
The original code is in R at <https://gitlab.gwdg.de/MedBioinf/generegulation/patternchrome>.

## What the paper does

PatternChrome predicts whether a gene is expressed (high/low) from its histone modification ChIP-seq signal.
It has 5 stages:
1. Bin the ±5000 bp region around each TSS into N bins
2. Feature extraction: use PSO to find "patterns" (short shapes in the signal) that improve XGBoost AUC on a training subset — keep adding patterns until training AUC hits 99.9%
3. Backward elimination: drop patterns that don't help on the validation set
4. Hyperparameter tuning: use PSO again to tune XGBoost's hyperparameters
5. Final prediction: train the final model, evaluate on the test set

## Files

```
run.py                    # run from the command line
compare_optimizers.py     # our added experiment: PSO vs DE vs random search
src/
  data.py                 # load CSVs or generate synthetic data
  pattern.py              # Pattern class + sliding Pearson correlation
  pso.py                  # PSO matching the paper's R psoptim settings
  feature_extraction.py   # stage 2
  backward_elim.py        # stage 3
  train.py                # stages 4 and 5
  explain.py              # SHAP values for interpretation
  plots.py                # reproduce the paper's figures
  pipeline.py             # wires all stages together
  optimizer.py            # shared interface for PSO / DE / random
```

## Quick start

```bash
# 1. create the environment
conda activate base
make env

# 2. activate it
conda activate patternchrome

# 3. smoke test — no data needed, takes ~2 min
make test

# 4. real run on one cell line (see below for data)
make run-E003
```

## Getting the data

The paper uses 56 cell lines from [Roadmap Epigenomics](https://egg2.wustl.edu/roadmap/web_portal/).
The easiest way to get it is the [DeepChrome Zenodo archive](https://zenodo.org/record/2652278) which has the data already binned as CSVs:

```
GeneID, BinID, H3K27me3, H3K36me3, H3K4me1, H3K4me3, H3K9me3, Label
```

Download a cell line (e.g. E003), unzip, and put it in `data/E003/classification/`.

> Note: the paper uses 200 bins × 50 bp; DeepChrome uses 100 bins × 100 bp.
> The pipeline still works, you just have a bit less spatial resolution.

## Running

```bash
# with pre-split CSVs (DeepChrome format)
make run-E003

# or directly
python run.py --train-csv data/E003/classification/train.csv \
              --valid-csv data/E003/classification/valid.csv \
              --test-csv  data/E003/classification/test.csv \
              --cell E003

# bigger run, closer to paper settings
python run.py --csv data/E003.csv --cell E003 \
    --n-patterns 30 --pso-particles 30 --pso-iters 40 --inner-subset 3000
```

Plots are written to `results/`.

## Our added experiment

We compare PSO against two alternatives (Differential Evolution and random search) to check whether PSO is actually necessary:

```bash
make compare-E003   # runs the pipeline 3 times, once per optimizer
```

Output: `results/optimizer_comparison_E003.csv` and `results/optimizer_comparison_E003.png`.

## Differences from the original

1. **Bin width**: DeepChrome data is 100 bp bins, paper uses 50 bp. Regenerating from raw BAMs is too much work for a class project.
2. **PSO**: we use R's `psoptim` parameters (`w=c(0.8,0.4)`, `c.p=c.g=2.05`) but without the parallel cluster that the R code uses.
3. **Explainability**: paper uses R's `xgboostExplainer`; we use XGBoost's built-in `pred_contribs=True` (TreeSHAP). Similar idea, different implementation.
4. **n_patterns_max**: we cap at 20 by default to keep runs short; the paper just stops at 99.9% training AUC.

## References

- Paul N.B., Wolber J.C. et al. *PatternChrome: Prediction of gene expression using histone modification patterns extracted by Particle Swarm Optimization.* Bioinformatics 41(2), 2025.
- Singh R. et al. *DeepChrome.* Bioinformatics 32(17), 2016.


Every time you come back to work on the project:
    conda activate patternchrome
    jupyter lab
