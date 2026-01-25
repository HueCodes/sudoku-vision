# Sudoku Vision Makefile
# =====================

PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

# Directories
DATA_DIR := data
RAW_DIR := $(DATA_DIR)/raw
LABELED_DIR := $(DATA_DIR)/labeled
AUGMENTED_DIR := $(DATA_DIR)/augmented
TEST_IMAGES := $(DATA_DIR)/test_images
MODELS_DIR := models

# Default target
.PHONY: help
help:
	@echo "Sudoku Vision - Available targets:"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Create virtual environment and install dependencies"
	@echo ""
	@echo "Data Pipeline (Phase 1):"
	@echo "  make extract-cells  - Extract cells from test images"
	@echo "  make label          - Start interactive labeling tool"
	@echo "  make organize-data  - Organize labeled data into train/val/test splits"
	@echo "  make augment        - Augment training data"
	@echo "  make data-stats     - Show dataset statistics"
	@echo "  make gen-synthetic  - Generate synthetic training data"
	@echo ""
	@echo "Training (Phase 2):"
	@echo "  make train          - Train digit recognition model (v1)"
	@echo "  make train-v3       - Train improved model (v3) with augmentation"
	@echo "  make evaluate       - Evaluate model on test data"
	@echo "  make evaluate-full  - Full evaluation with plots"
	@echo ""
	@echo "Grid Detection (Phase 4):"
	@echo "  make test-grid      - Test grid detection on sample images"
	@echo "  make test-quality   - Test grid quality scoring"
	@echo ""
	@echo "Testing (Phase 5):"
	@echo "  make test           - Run all tests"
	@echo "  make test-unit      - Run unit tests"
	@echo "  make test-integration - Run integration tests"
	@echo ""
	@echo "Solver:"
	@echo "  make solver         - Build C solver"
	@echo "  make solver-wasm    - Build WebAssembly solver"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove generated files"
	@echo "  make clean-data     - Remove extracted/augmented data"

# ============================================================================
# Setup
# ============================================================================

.PHONY: setup
setup: $(VENV)/bin/activate
	$(PIP) install -r requirements.txt

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)

# ============================================================================
# Data Pipeline (Phase 1)
# ============================================================================

.PHONY: extract-cells
extract-cells:
	@echo "Extracting cells from test images..."
	$(PYTHON_VENV) tools/extract_cells.py $(TEST_IMAGES) \
		--output-dir $(RAW_DIR) \
		--create-labeling-manifest
	@echo "Done. Run 'make label' to start labeling."

.PHONY: extract-cells-from
extract-cells-from:
	@if [ -z "$(INPUT)" ]; then \
		echo "Usage: make extract-cells-from INPUT=<directory>"; \
		exit 1; \
	fi
	$(PYTHON_VENV) tools/extract_cells.py $(INPUT) \
		--output-dir $(RAW_DIR) \
		--create-labeling-manifest

.PHONY: label
label:
	@if [ -f "$(RAW_DIR)/to_label.json" ]; then \
		$(PYTHON_VENV) tools/label_cells.py $(RAW_DIR)/to_label.json; \
	elif [ -d "$(RAW_DIR)" ]; then \
		$(PYTHON_VENV) tools/label_cells.py $(RAW_DIR); \
	else \
		echo "No cells to label. Run 'make extract-cells' first."; \
		exit 1; \
	fi

.PHONY: organize-data
organize-data:
	@echo "Organizing labeled data..."
	$(PYTHON_VENV) tools/organize_dataset.py $(RAW_DIR)/labels.csv \
		--output $(LABELED_DIR) \
		--clean
	@echo "Done. Dataset organized in $(LABELED_DIR)"

.PHONY: organize-data-from
organize-data-from:
	@if [ -z "$(INPUT)" ]; then \
		echo "Usage: make organize-data-from INPUT=<csv or directory>"; \
		exit 1; \
	fi
	$(PYTHON_VENV) tools/organize_dataset.py $(INPUT) \
		--output $(LABELED_DIR)

.PHONY: augment
augment:
	@echo "Augmenting training data..."
	$(PYTHON_VENV) tools/augment_data.py $(LABELED_DIR)/train \
		--output $(AUGMENTED_DIR)/train \
		--multiplier 10 \
		--intensity medium
	@echo "Done. Augmented data in $(AUGMENTED_DIR)"

.PHONY: augment-preview
augment-preview:
	@if [ -z "$(IMAGE)" ]; then \
		echo "Usage: make augment-preview IMAGE=<path to image>"; \
		exit 1; \
	fi
	$(PYTHON_VENV) tools/augment_data.py . --preview $(IMAGE) \
		--output augmentation_preview.png

.PHONY: data-stats
data-stats:
	@echo "Dataset Statistics:"
	@echo ""
	@if [ -d "$(RAW_DIR)" ]; then \
		echo "=== Raw Data ==="; \
		$(PYTHON_VENV) tools/dataset_stats.py $(RAW_DIR); \
	fi
	@if [ -d "$(LABELED_DIR)" ]; then \
		echo ""; \
		echo "=== Labeled Data ==="; \
		$(PYTHON_VENV) tools/dataset_stats.py $(LABELED_DIR); \
	fi
	@if [ -d "$(AUGMENTED_DIR)" ]; then \
		echo ""; \
		echo "=== Augmented Data ==="; \
		$(PYTHON_VENV) tools/dataset_stats.py $(AUGMENTED_DIR); \
	fi

.PHONY: data-pipeline
data-pipeline: extract-cells
	@echo ""
	@echo "Cells extracted. Now run 'make label' to label them interactively."
	@echo "After labeling, run 'make organize-data' to create train/val/test splits."
	@echo "Finally, run 'make augment' to augment training data."

# ============================================================================
# Synthetic Data Generation
# ============================================================================

.PHONY: gen-synthetic
gen-synthetic:
	@echo "Generating synthetic training data..."
	$(PYTHON_VENV) ml/generate_synthetic_v2.py \
		--output $(DATA_DIR)/synthetic_v2 \
		--num-per-class 5000
	@echo "Done. Synthetic data in $(DATA_DIR)/synthetic_v2"

.PHONY: gen-synthetic-preview
gen-synthetic-preview:
	$(PYTHON_VENV) ml/generate_synthetic_v2.py --preview \
		--output $(DATA_DIR)/synthetic_v2

# ============================================================================
# Training (Phase 2)
# ============================================================================

.PHONY: train
train:
	$(PYTHON_VENV) ml/train.py \
		--dataset combined \
		--epochs 20 \
		--output $(MODELS_DIR)/digit_cnn.pt

.PHONY: train-v3
train-v3:
	@mkdir -p $(MODELS_DIR) logs
	$(PYTHON_VENV) ml/train_v2.py \
		--model v3 \
		--dataset mnist \
		--epochs 30 \
		--augmentation medium \
		--output $(MODELS_DIR)/digit_cnn_v3.pt \
		--calibrate

.PHONY: train-v3-real
train-v3-real:
	@mkdir -p $(MODELS_DIR) logs
	$(PYTHON_VENV) ml/train_v2.py \
		--model v3 \
		--dataset combined \
		--epochs 50 \
		--augmentation heavy \
		--pretrained $(MODELS_DIR)/digit_cnn_v3.pt \
		--output $(MODELS_DIR)/digit_cnn_v3_real.pt \
		--calibrate

.PHONY: train-mnist
train-mnist:
	$(PYTHON_VENV) ml/train.py \
		--dataset mnist \
		--epochs 10 \
		--output $(MODELS_DIR)/digit_cnn_mnist.pt

.PHONY: train-real
train-real:
	$(PYTHON_VENV) ml/train.py \
		--dataset real \
		--epochs 30 \
		--output $(MODELS_DIR)/digit_cnn_real.pt

.PHONY: evaluate
evaluate:
	@mkdir -p eval_results
	$(PYTHON_VENV) ml/evaluate_v2.py \
		--model $(MODELS_DIR)/digit_cnn_v3.pt \
		--data-dir $(DATA_DIR) \
		--output-dir eval_results

.PHONY: evaluate-full
evaluate-full:
	@mkdir -p eval_results
	$(PYTHON_VENV) ml/evaluate_v2.py \
		--model $(MODELS_DIR)/digit_cnn_v3.pt \
		--data-dir $(DATA_DIR) \
		--output-dir eval_results \
		--save-failures

# ============================================================================
# Grid Detection (Phase 4)
# ============================================================================

.PHONY: test-grid
test-grid:
	@echo "Testing grid detection on sample images..."
	@for img in $(TEST_IMAGES)/*.jpg; do \
		echo "Processing $$img..."; \
		$(PYTHON_VENV) cv/grid_v2.py "$$img" 2>/dev/null || echo "  Failed"; \
	done

.PHONY: test-quality
test-quality:
	@echo "Testing grid quality scoring..."
	@for img in $(TEST_IMAGES)/*.jpg; do \
		echo ""; \
		echo "=== $$img ==="; \
		$(PYTHON_VENV) cv/grid_quality.py "$$img" 2>/dev/null || echo "  Failed"; \
	done

.PHONY: test-preprocess
test-preprocess:
	@echo "Testing enhanced preprocessing..."
	$(PYTHON_VENV) cv/preprocess_v2.py $(TEST_IMAGES)/sample_1.jpg

# ============================================================================
# Testing
# ============================================================================

.PHONY: test
test: test-unit

.PHONY: test-unit
test-unit:
	$(PYTHON_VENV) -m pytest tests/ -v

.PHONY: test-solver
test-solver:
	$(PYTHON_VENV) -m pytest tests/test_solver.py -v

.PHONY: test-integration
test-integration:
	$(PYTHON_VENV) -m pytest tests/test_integration.py -v

# ============================================================================
# Solver
# ============================================================================

.PHONY: solver
solver:
	$(MAKE) -C solver

.PHONY: solver-wasm
solver-wasm:
	$(MAKE) -C solver wasm

.PHONY: solver-clean
solver-clean:
	$(MAKE) -C solver clean

# ============================================================================
# Cleanup
# ============================================================================

.PHONY: clean
clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache */.pytest_cache
	rm -rf *.pyc */*.pyc
	rm -f augmentation_preview.png
	rm -f preprocessed.png warped.png

.PHONY: clean-data
clean-data:
	rm -rf $(RAW_DIR)
	rm -rf $(AUGMENTED_DIR)
	@echo "Removed raw and augmented data. Labeled data preserved."

.PHONY: clean-all
clean-all: clean clean-data solver-clean
	rm -rf $(LABELED_DIR)
	@echo "All generated data removed."
