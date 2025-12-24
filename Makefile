# Makefile for hbmon (hummingbird-monitor)
#
# This Makefile provides shortcuts for common development tasks including
# environment setup, linting, testing, and Docker operations.
#
# Usage: make <target>
# Run 'make help' to see all available targets.

.PHONY: help venv sync install lint lint-fix test test-unit test-cov \
        docker-build docker-up docker-down docker-logs docker-logs-worker docker-logs-web \
        pre-commit-install pre-commit-run run-web run-worker clean

# Default target
.DEFAULT_GOAL := help

# ============================================================================
# Environment Setup
# ============================================================================

venv: ## Create a uv virtual environment
	uv venv

sync: ## Sync dependencies from pyproject.toml using uv.lock
	uv sync

install: ## Install package in editable mode with dev dependencies
	uv pip install -e ".[dev]"

# ============================================================================
# Code Quality
# ============================================================================

lint: ## Run ruff linter
	uv run ruff check .

lint-fix: ## Run ruff linter with auto-fix
	uv run ruff check . --fix

# ============================================================================
# Testing
# ============================================================================

test: ## Run tests with coverage (default)
	uv run pytest --cov=hbmon --cov-report=term

test-unit: ## Run tests without coverage (faster)
	uv run pytest -q

test-cov: ## Run tests with full coverage reports (term + html + xml)
	uv run pytest --cov=hbmon --cov-report=term --cov-report=html --cov-report=xml

# ============================================================================
# Docker Operations
# ============================================================================

docker-build: ## Build Docker image
	docker build -t hbmon .

docker-up: ## Start all containers with Docker Compose (builds if needed)
	docker compose up -d --build

docker-down: ## Stop all containers
	docker compose down

docker-logs: ## Follow logs from all containers
	docker compose logs -f

docker-logs-worker: ## Follow logs from hbmon-worker container
	docker compose logs -f hbmon-worker

docker-logs-web: ## Follow logs from hbmon-web container
	docker compose logs -f hbmon-web

# ============================================================================
# Pre-commit Hooks
# ============================================================================

pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install

pre-commit-run: ## Run pre-commit on all files
	uv run pre-commit run --all-files

# ============================================================================
# Local Development
# ============================================================================

run-web: ## Run the web server locally (development mode with reload)
	uv run uvicorn hbmon.web:app --reload --host 0.0.0.0 --port 8000

run-worker: ## Run the worker locally (requires HBMON_RTSP_URL env var)
	uv run python -m hbmon.worker

# ============================================================================
# Cleanup
# ============================================================================

clean: ## Remove build artifacts and caches
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ============================================================================
# Help
# ============================================================================

help: ## Show this help message
	@echo "Usage: make <target>"
	@echo ""
	@echo "Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make venv install    # Set up development environment"
	@echo "  make lint test       # Run linter and tests"
	@echo "  make docker-up       # Start all containers"
