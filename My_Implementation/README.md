# PatternChrome — Python re-implementation

Python port of [PatternChrome (Paul et al., Bioinformatics 2025)](https://doi.org/10.1093/bioinformatics/btaf033)
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


#We compare PSO against two alternatives (Differential Evolution and random search) to check whether PSO is actually necessary:

make compare-E003   # runs the pipeline 3 times, once per optimizer
```

Plots are written to `results/`.

