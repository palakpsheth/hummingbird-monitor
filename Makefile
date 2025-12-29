.DEFAULT_GOAL := help

UV ?= uv
UV_RUN ?= UV_INDEX_URL=$(PYTORCH_INDEX_URL) UV_EXTRA_INDEX_URL=https://pypi.org/simple $(UV) run
PYTEST ?= $(UV_RUN) pytest -n auto
RUFF ?= $(UV_RUN) ruff

DATA_DIR ?= data
MEDIA_DIR ?= $(DATA_DIR)/media
DB_PATH ?= $(DATA_DIR)/hbmon.sqlite
CONFIG_PATH ?= $(DATA_DIR)/config.json

PYTEST_UNIT_ARGS ?= -m "not integration"
PYTEST_INTEGRATION_ARGS ?= -m "integration"
PYTORCH_INDEX_URL ?= https://download.pytorch.org/whl/cpu
PYTORCH_GPU_INDEX_URL ?= https://download.pytorch.org/whl/cu121

.PHONY: help venv sync sync-gpu lint test test-unit test-integration pre-commit docker-build docker-up docker-build-gpu docker-up-gpu docker-down docker-ps clean-db clean-media clean-data

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_-]+:.*##/ {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

venv: ## Create a uv virtual environment (.venv)
	$(UV) venv

sync: ## Sync dev dependencies from pyproject.toml
	$(UV) pip install -e ".[dev]" --index-url $(PYTORCH_INDEX_URL) --extra-index-url https://pypi.org/simple

sync-gpu: ## Sync dev dependencies with CUDA-enabled PyTorch wheels
	$(UV) pip install -e ".[dev]" --index-url $(PYTORCH_GPU_INDEX_URL) --extra-index-url https://pypi.org/simple

lint: ## Run Ruff linting
	$(RUFF) check --fix .

test: ## Run full pytest suite with coverage
	$(PYTEST) --cov=hbmon --cov-report=term

test-unit: ## Run unit tests with coverage (marker: not integration)
	$(PYTEST) $(PYTEST_UNIT_ARGS) -vvv --cov=hbmon --cov-report=term

test-integration: ## Run integration/UI tests with coverage (marker: integration)
	$(PYTEST) $(PYTEST_INTEGRATION_ARGS) -vvv --cov=hbmon --cov-report=term

pre-commit: ## Run all pre-commit hooks
	$(UV_RUN) pre-commit run --all-files

docker-build: ## Build docker images
	docker compose build --build-arg PYTORCH_INDEX_URL=$(PYTORCH_INDEX_URL)

docker-up: ## Start docker compose (build if needed)
	docker compose build --build-arg PYTORCH_INDEX_URL=$(PYTORCH_INDEX_URL)
	docker compose up -d

docker-build-gpu: ## Build docker images with CUDA-enabled PyTorch
	docker compose build --build-arg PYTORCH_INDEX_URL=$(PYTORCH_GPU_INDEX_URL)

docker-up-gpu: ## Start docker compose with CUDA-enabled PyTorch (build if needed)
	docker compose build --build-arg PYTORCH_INDEX_URL=$(PYTORCH_GPU_INDEX_URL)
	docker compose up -d

docker-ps: ## Get docker compose status
	docker compose ps

docker-down: ## Stop docker compose
	docker compose down

clean-db: ## Remove the local database file only (defaults to ./data)
	rm -f $(DB_PATH)

clean-media: ## Remove local media files (defaults to ./data/media)
	rm -rf $(MEDIA_DIR)

clean-data: ## CAUTION: Remove all local data (defaults to ./data)
	rm -rf $(DATA_DIR)
