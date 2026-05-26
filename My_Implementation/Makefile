# =============================================================================
# PatternChrome — Makefile
#
# Directory layout expected:
#   data/
#     E003/classification/train.csv  valid.csv  test.csv
#     E123/classification/train.csv  valid.csv  test.csv
#   src/          (all .py modules)
#   notebooks/    (demo.ipynb)
#   results/      (plots land here — created automatically)
#
# Usage:
#   make help               — list all targets
#   make env                — create the conda environment
#   make test               — quick smoke test with synthetic data (~2 min)
#   make run-E003           — full pipeline on cell line E003 (PSO only)
#   make run-E123           — full pipeline on cell line E123 (PSO only)
#   make run-all            — both cell lines, PSO only
#   make compare-E003       — OPTIMIZER ABLATION on E003 (PSO vs DE vs random)
#   make compare-E123       — OPTIMIZER ABLATION on E123 (PSO vs DE vs random)
#   make compare-all        — ablation on both cell lines
#   make compare-test       — synthetic-data ablation (fast sanity check)
#   make notebook           — launch Jupyter Lab
#   make clean              — delete generated results
#   make clean-all          — delete results + pycache
# =============================================================================

# ── Config ────────────────────────────────────────────────────────────────────
ENV_NAME    := patternchrome
PYTHON      := conda run -n $(ENV_NAME) python
JUPYTER     := conda run -n $(ENV_NAME) jupyter

SRC_DIR     := src
DATA_DIR    := data
RESULTS_DIR := results
NOTEBOOK    := notebooks/demo.ipynb

# Pipeline parameters (override on the command line, e.g. make run-E003 N_PATTERNS=30)
N_PATTERNS     ?= 15
PSO_PARTICLES  ?= 15
PSO_ITERS      ?= 25
INNER_SUBSET   ?= 2000
SEED           ?= 0

# Optimizer-ablation parameters (override likewise)
ABL_PATTERNS    ?= 10
ABL_PARTICLES   ?= 12
ABL_ITERS       ?= 15
ABL_SUBSET      ?= 1500
ABL_OPTIMIZERS  ?= pso,de,random

# ── Phony targets (not real files) ───────────────────────────────────────────
.PHONY: help env env-check test check-data \
        run-E003 run-E123 run-all \
        compare-E003 compare-E123 compare-all compare-test \
        notebook clean clean-all lint

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  PatternChrome — available targets"
	@echo "  ─────────────────────────────────────────────────────────────"
	@echo "  Setup:"
	@echo "    make env              Create the conda environment"
	@echo "    make env-check        Verify all imports work"
	@echo "    make check-data       Confirm data files are present"
	@echo ""
	@echo "  Run pipeline (PSO only — the paper's setup):"
	@echo "    make test             Smoke test on synthetic data (~2 min)"
	@echo "    make run-E003         Run pipeline on E003 cell line"
	@echo "    make run-E123         Run pipeline on E123 cell line"
	@echo "    make run-all          Both cell lines"
	@echo ""
	@echo "  Optimizer ablation (PSO vs DE vs Random — your contribution):"
	@echo "    make compare-test     Ablation smoke test (~5 min)"
	@echo "    make compare-E003     Full ablation on E003 (~45 min)"
	@echo "    make compare-E123     Full ablation on E123 (~45 min)"
	@echo "    make compare-all      Ablation on both cell lines"
	@echo ""
	@echo "  Other:"
	@echo "    make notebook         Launch Jupyter Lab"
	@echo "    make clean            Delete results/ folder"
	@echo "    make clean-all        Delete results/ + all __pycache__"
	@echo ""
	@echo "  Parameters (pass on command line to override defaults):"
	@echo "    Pipeline:  N_PATTERNS=$(N_PATTERNS)  PSO_PARTICLES=$(PSO_PARTICLES)  PSO_ITERS=$(PSO_ITERS)"
	@echo "               INNER_SUBSET=$(INNER_SUBSET)  SEED=$(SEED)"
	@echo "    Ablation:  ABL_PATTERNS=$(ABL_PATTERNS)  ABL_PARTICLES=$(ABL_PARTICLES)  ABL_ITERS=$(ABL_ITERS)"
	@echo "               ABL_SUBSET=$(ABL_SUBSET)  ABL_OPTIMIZERS=$(ABL_OPTIMIZERS)"
	@echo ""
	@echo "  Examples:"
	@echo "    make run-E003 N_PATTERNS=30 PSO_PARTICLES=30 PSO_ITERS=40"
	@echo "    make compare-E003 ABL_OPTIMIZERS=pso,de"
	@echo ""

# ── Environment ───────────────────────────────────────────────────────────────
env:
	@echo "→ Creating conda environment '$(ENV_NAME)' with Python 3.11..."
	conda create -n $(ENV_NAME) python=3.11 -y
	conda run -n $(ENV_NAME) conda install -c conda-forge \
		numpy pandas scipy scikit-learn matplotlib seaborn \
		tqdm xgboost jupyterlab ipykernel -y
	conda run -n $(ENV_NAME) python -m ipykernel install \
		--user --name $(ENV_NAME) --display-name "Python ($(ENV_NAME))"
	@echo ""
	@echo "✓ Environment ready."
	@echo "  Activate with:  conda activate $(ENV_NAME)"
	@echo "  Then run:       make test"

env-check:
	@echo "→ Checking imports in environment '$(ENV_NAME)'..."
	$(PYTHON) -c "\
import numpy, pandas, scipy, sklearn, xgboost, matplotlib, seaborn; \
print('  numpy     ', numpy.__version__); \
print('  pandas    ', pandas.__version__); \
print('  sklearn   ', sklearn.__version__); \
print('  xgboost   ', xgboost.__version__); \
print('  scipy     ', scipy.__version__); \
print('✓ All imports OK')"
	@echo "→ Checking optimizer modules..."
	$(PYTHON) -c "\
import sys; sys.path.insert(0, '$(SRC_DIR)'); \
from optimizer import get_optimizer; \
[print(f'  {n:7s}  loaded from {get_optimizer(n).__module__}') for n in ['pso','de','random']]; \
print('✓ All optimizers available')"

# ── Data check ────────────────────────────────────────────────────────────────
check-data:
	@echo "→ Checking data files..."
	@for cell in E003 E123; do \
		for split in train valid test; do \
			f="$(DATA_DIR)/$$cell/classification/$$split.csv"; \
			if [ -f "$$f" ]; then \
				echo "  ✓ $$f"; \
			else \
				echo "  ✗ MISSING: $$f"; \
			fi; \
		done; \
	done
	@echo "Done."

# ── Results directory ─────────────────────────────────────────────────────────
$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

# ═════════════════════════════════════════════════════════════════════════════
# STANDARD PIPELINE RUNS (PSO only)
# ═════════════════════════════════════════════════════════════════════════════

test: $(RESULTS_DIR)
	@echo "→ Running smoke test on synthetic data..."
	$(PYTHON) run.py \
		--synthetic \
		--cell SYN \
		--out $(RESULTS_DIR) \
		--n-patterns 5 \
		--pso-particles 6 \
		--pso-iters 5 \
		--inner-subset 500 \
		--seed $(SEED)
	@echo "✓ Smoke test passed. Figures written to $(RESULTS_DIR)/"

run-E003: $(RESULTS_DIR) check-data
	@echo "→ Running pipeline on E003..."
	$(PYTHON) run.py \
		--train-csv  $(DATA_DIR)/E003/classification/train.csv \
		--valid-csv  $(DATA_DIR)/E003/classification/valid.csv \
		--test-csv   $(DATA_DIR)/E003/classification/test.csv \
		--cell       E003 \
		--out        $(RESULTS_DIR) \
		--n-patterns     $(N_PATTERNS) \
		--pso-particles  $(PSO_PARTICLES) \
		--pso-iters      $(PSO_ITERS) \
		--inner-subset   $(INNER_SUBSET) \
		--seed           $(SEED)
	@echo "✓ E003 done. Results in $(RESULTS_DIR)/"

run-E123: $(RESULTS_DIR) check-data
	@echo "→ Running pipeline on E123..."
	$(PYTHON) run.py \
		--train-csv  $(DATA_DIR)/E123/classification/train.csv \
		--valid-csv  $(DATA_DIR)/E123/classification/valid.csv \
		--test-csv   $(DATA_DIR)/E123/classification/test.csv \
		--cell       E123 \
		--out        $(RESULTS_DIR) \
		--n-patterns     $(N_PATTERNS) \
		--pso-particles  $(PSO_PARTICLES) \
		--pso-iters      $(PSO_ITERS) \
		--inner-subset   $(INNER_SUBSET) \
		--seed           $(SEED)
	@echo "✓ E123 done. Results in $(RESULTS_DIR)/"

run-all: run-E003 run-E123
	@echo "✓ All cell lines done. Results in $(RESULTS_DIR)/"

# ═════════════════════════════════════════════════════════════════════════════
# OPTIMIZER ABLATION (your scientific contribution)
# PSO vs Differential Evolution vs Random Search
# ═════════════════════════════════════════════════════════════════════════════

compare-test: $(RESULTS_DIR)
	@echo "→ Optimizer ablation on synthetic data..."
	@echo "  Optimizers: $(ABL_OPTIMIZERS)"
	$(PYTHON) compare_optimizers.py \
		--synthetic \
		--cell SYN \
		--out $(RESULTS_DIR) \
		--n-patterns 5 \
		--pso-particles 8 \
		--pso-iters 8 \
		--inner-subset 800 \
		--seed $(SEED) \
		--optimizers $(ABL_OPTIMIZERS)
	@echo "✓ Ablation smoke test passed."

compare-E003: $(RESULTS_DIR) check-data
	@echo "→ Optimizer ablation on E003..."
	@echo "  Optimizers: $(ABL_OPTIMIZERS)"
	@echo "  Runs the full pipeline 3 times (one per optimizer)."
	$(PYTHON) compare_optimizers.py \
		--train-csv  $(DATA_DIR)/E003/classification/train.csv \
		--valid-csv  $(DATA_DIR)/E003/classification/valid.csv \
		--test-csv   $(DATA_DIR)/E003/classification/test.csv \
		--cell       E003 \
		--out        $(RESULTS_DIR) \
		--n-patterns     $(ABL_PATTERNS) \
		--pso-particles  $(ABL_PARTICLES) \
		--pso-iters      $(ABL_ITERS) \
		--inner-subset   $(ABL_SUBSET) \
		--seed           $(SEED) \
		--optimizers     $(ABL_OPTIMIZERS)
	@echo "✓ E003 ablation done. See:"
	@echo "    $(RESULTS_DIR)/optimizer_comparison_E003.csv"
	@echo "    $(RESULTS_DIR)/optimizer_comparison_E003.png"
	@echo "    $(RESULTS_DIR)/optimizer_runtime_E003.png"

compare-E123: $(RESULTS_DIR) check-data
	@echo "→ Optimizer ablation on E123..."
	@echo "  Optimizers: $(ABL_OPTIMIZERS)"
	$(PYTHON) compare_optimizers.py \
		--train-csv  $(DATA_DIR)/E123/classification/train.csv \
		--valid-csv  $(DATA_DIR)/E123/classification/valid.csv \
		--test-csv   $(DATA_DIR)/E123/classification/test.csv \
		--cell       E123 \
		--out        $(RESULTS_DIR) \
		--n-patterns     $(ABL_PATTERNS) \
		--pso-particles  $(ABL_PARTICLES) \
		--pso-iters      $(ABL_ITERS) \
		--inner-subset   $(ABL_SUBSET) \
		--seed           $(SEED) \
		--optimizers     $(ABL_OPTIMIZERS)
	@echo "✓ E123 ablation done. See:"
	@echo "    $(RESULTS_DIR)/optimizer_comparison_E123.csv"
	@echo "    $(RESULTS_DIR)/optimizer_comparison_E123.png"
	@echo "    $(RESULTS_DIR)/optimizer_runtime_E123.png"

compare-all: compare-E003 compare-E123
	@echo ""
	@echo "✓ Full ablation done on both cell lines."
	@echo "  For your article: include the two CSVs as a results table,"
	@echo "  and the two comparison PNGs as figures."

# ═════════════════════════════════════════════════════════════════════════════
# OTHER TARGETS
# ═════════════════════════════════════════════════════════════════════════════

notebook:
	@echo "→ Launching Jupyter Lab..."
	$(JUPYTER) lab $(NOTEBOOK)

clean:
	@echo "→ Removing $(RESULTS_DIR)/..."
	rm -rf $(RESULTS_DIR)
	@echo "✓ Done."

clean-all: clean
	@echo "→ Removing __pycache__ directories..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Done."

lint:
	$(PYTHON) -m flake8 $(SRC_DIR)/ run.py compare_optimizers.py \
		--max-line-length=100 --ignore=E501,W503 || true