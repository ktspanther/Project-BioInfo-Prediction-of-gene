# Makefile for the PatternChrome Python re-implementation
# usage: make test | make run-E003 | make compare-E003

ENV_NAME  := patternchrome
PYTHON    := conda run -n $(ENV_NAME) python
JUPYTER   := conda run -n $(ENV_NAME) jupyter

DATA_DIR  := data
OUT       := results

# pipeline parameters — match PatternChrome.R defaults
# initial_swarm_size=20, initial_maxit=20, num_sample_genes=3000
N_PATTERNS    ?= 20
PSO_PARTICLES ?= 20
PSO_ITERS     ?= 20
INNER_SUBSET  ?= 3000
SEED          ?= 0

# ablation parameters (same PSO settings, fewer patterns to keep it faster)
ABL_PATTERNS  ?= 10
ABL_PARTICLES ?= 20
ABL_ITERS     ?= 20
ABL_SUBSET    ?= 3000
ABL_OPTS      ?= pso,de,random

.PHONY: env test run-E003 run-E123 run-all \
        compare-test compare-E003 compare-E123 compare-all \
        notebook clean

# create the conda environment
env:
	conda create -n $(ENV_NAME) python=3.11 -y
	conda run -n $(ENV_NAME) conda install -c conda-forge \
		numpy pandas scipy scikit-learn matplotlib seaborn \
		xgboost jupyterlab ipykernel -y
	conda run -n $(ENV_NAME) python -m ipykernel install \
		--user --name $(ENV_NAME) --display-name "Python ($(ENV_NAME))"

# quick smoke test with synthetic data, no download needed (~2 min)
test:
	mkdir -p $(OUT)
	$(PYTHON) run.py --synthetic --cell SYN --out $(OUT) \
		--n-patterns 5 --pso-particles 6 --pso-iters 5 \
		--inner-subset 500 --seed $(SEED)

# run the full pipeline on a real cell line
run-E003:
	mkdir -p $(OUT)
	$(PYTHON) run.py \
		--train-csv $(DATA_DIR)/E003/classification/train.csv \
		--valid-csv $(DATA_DIR)/E003/classification/valid.csv \
		--test-csv  $(DATA_DIR)/E003/classification/test.csv \
		--cell E003 --out $(OUT) \
		--n-patterns $(N_PATTERNS) --pso-particles $(PSO_PARTICLES) \
		--pso-iters $(PSO_ITERS) --inner-subset $(INNER_SUBSET) --seed $(SEED)

run-E123:
	mkdir -p $(OUT)
	$(PYTHON) run.py \
		--train-csv $(DATA_DIR)/E123/classification/train.csv \
		--valid-csv $(DATA_DIR)/E123/classification/valid.csv \
		--test-csv  $(DATA_DIR)/E123/classification/test.csv \
		--cell E123 --out $(OUT) \
		--n-patterns $(N_PATTERNS) --pso-particles $(PSO_PARTICLES) \
		--pso-iters $(PSO_ITERS) --inner-subset $(INNER_SUBSET) --seed $(SEED)

run-all: run-E003 run-E123

# optimizer comparison (PSO vs DE vs random) — our added experiment
compare-test:
	mkdir -p $(OUT)
	$(PYTHON) compare_optimizers.py --synthetic --cell SYN --out $(OUT) \
		--n-patterns 5 --pso-particles 8 --pso-iters 8 \
		--inner-subset 800 --seed $(SEED) --optimizers $(ABL_OPTS)

compare-E003:
	mkdir -p $(OUT)
	$(PYTHON) compare_optimizers.py \
		--train-csv $(DATA_DIR)/E003/classification/train.csv \
		--valid-csv $(DATA_DIR)/E003/classification/valid.csv \
		--test-csv  $(DATA_DIR)/E003/classification/test.csv \
		--cell E003 --out $(OUT) \
		--n-patterns $(ABL_PATTERNS) --pso-particles $(ABL_PARTICLES) \
		--pso-iters $(ABL_ITERS) --inner-subset $(ABL_SUBSET) \
		--seed $(SEED) --optimizers $(ABL_OPTS)

compare-E123:
	mkdir -p $(OUT)
	$(PYTHON) compare_optimizers.py \
		--train-csv $(DATA_DIR)/E123/classification/train.csv \
		--valid-csv $(DATA_DIR)/E123/classification/valid.csv \
		--test-csv  $(DATA_DIR)/E123/classification/test.csv \
		--cell E123 --out $(OUT) \
		--n-patterns $(ABL_PATTERNS) --pso-particles $(ABL_PARTICLES) \
		--pso-iters $(ABL_ITERS) --inner-subset $(ABL_SUBSET) \
		--seed $(SEED) --optimizers $(ABL_OPTS)

compare-all: compare-E003 compare-E123

clean:
	rm -rf $(OUT)
