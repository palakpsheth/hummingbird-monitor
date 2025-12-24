.DEFAULT_GOAL := help

UV ?= uv
PYTEST ?= $(UV) run pytest
RUFF ?= $(UV) run ruff

DATA_DIR ?= data
MEDIA_DIR ?= $(DATA_DIR)/media
DB_PATH ?= $(DATA_DIR)/hbmon.sqlite
CONFIG_PATH ?= $(DATA_DIR)/config.json

PYTEST_UNIT_ARGS ?= -m "not integration"
PYTEST_INTEGRATION_ARGS ?= -m "integration"

.PHONY: help venv sync lint test test-unit test-integration pre-commit docker-build docker-up docker-down clean-db clean-media clean-data

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_-]+:.*##/ {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

venv: ## Create a uv virtual environment (.venv)
	$(UV) venv

sync: ## Sync dev dependencies from pyproject.toml
	$(UV) pip install -e ".[dev]"

lint: ## Run Ruff linting
	$(RUFF) check .

test: ## Run full pytest suite with coverage
	$(PYTEST) --cov=hbmon --cov-report=term

test-unit: ## Run unit tests with coverage (marker: not integration)
	$(PYTEST) $(PYTEST_UNIT_ARGS) --cov=hbmon --cov-report=term

test-integration: ## Run integration/UI tests with coverage (marker: integration)
	$(PYTEST) $(PYTEST_INTEGRATION_ARGS) --cov=hbmon --cov-report=term

pre-commit: ## Run all pre-commit hooks
	$(UV) run pre-commit run --all-files

docker-build: ## Build docker images
	docker compose build

docker-up: ## Start docker compose (build if needed)
	docker compose up -d --build

docker-down: ## Stop docker compose
	docker compose down

clean-db: ## Remove the local database file only (defaults to ./data)
	rm -f $(DB_PATH)

clean-media: ## Remove local media files (defaults to ./data/media)
	rm -rf $(MEDIA_DIR)

clean-data: ## Remove all local data (defaults to ./data)
	rm -rf $(DATA_DIR)
