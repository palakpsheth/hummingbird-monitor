.DEFAULT_GOAL := help

UV ?= uv
UV_RUN ?= UV_INDEX_URL=$(PYTORCH_INDEX_URL) UV_EXTRA_INDEX_URL=https://pypi.org/simple $(UV) run
PYTEST ?= $(UV_RUN) pytest -n 2
RUFF ?= $(UV_RUN) ruff

DATA_DIR ?= data
MEDIA_DIR ?= $(DATA_DIR)/media
DB_PATH ?= $(DATA_DIR)/hbmon.sqlite
CONFIG_PATH ?= $(DATA_DIR)/config.json

PYTEST_UNIT_ARGS ?= -m "not integration"
PYTEST_INTEGRATION_ARGS ?= -m "integration"
PYTORCH_INDEX_URL ?= https://download.pytorch.org/whl/cpu
PYTORCH_GPU_INDEX_URL ?= https://download.pytorch.org/whl/cu121

.PHONY: help venv sync sync-gpu lint test test-unit test-integration pre-commit check-gpu docker-build docker-up docker-build-cuda docker-up-cuda docker-build-intel docker-up-intel docker-down docker-ps clean-db clean-media clean-data clean-openvino-cache test-stream-list test-stream-start test-detection check-health test-e2e

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_-]+:.*##/ {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

venv: ## Create a uv virtual environment (.venv)
	$(UV) venv

sync: ## Sync dev dependencies from pyproject.toml
	$(UV) pip install -e ".[dev]" --index-url $(PYTORCH_INDEX_URL) --extra-index-url https://pypi.org/simple
	$(UV_RUN) playwright install --with-deps chromium

sync-gpu: ## Sync dev dependencies with CUDA-enabled PyTorch wheels
	$(UV) pip install -e ".[dev]" --index-url $(PYTORCH_GPU_INDEX_URL) --extra-index-url https://pypi.org/simple
	$(UV_RUN) playwright install --with-deps chromium

lint: ## Run Ruff linting
	$(RUFF) check --fix .

test: ## Run full pytest suite with coverage
	$(PYTEST) --cov=hbmon --cov-report=term

test-unit: ## Run unit tests with coverage (marker: not integration)
	$(PYTEST) $(PYTEST_UNIT_ARGS) -vvv --cov=hbmon --cov-report=term

test-integration: ## Run integration/UI tests with coverage (marker: integration)
	$(PYTEST) $(PYTEST_INTEGRATION_ARGS) -vvv --cov=hbmon --cov-report=term

pre-commit: ## Run all pre-commit hooks
	$(UV_RUN) pre-commit install
	$(UV_RUN) pre-commit run --all-files

check-gpu: ## Detect available GPU hardware and recommend build
	@echo "=== GPU Detection ==="
	@echo ""
	@NVIDIA_FOUND=0; INTEL_FOUND=0; AMD_FOUND=0; \
	if lspci 2>/dev/null | grep -qi "vga.*nvidia\|3d.*nvidia"; then \
		NVIDIA_FOUND=1; \
		echo "✓ NVIDIA GPU detected:"; \
		lspci | grep -i "vga.*nvidia\|3d.*nvidia" | sed 's/^/  /'; \
		if command -v nvidia-smi >/dev/null 2>&1; then \
			echo "  nvidia-smi: available"; \
		else \
			echo "  nvidia-smi: NOT FOUND (install NVIDIA drivers + nvidia-container-toolkit)"; \
		fi; \
		echo ""; \
	fi; \
	if lspci 2>/dev/null | grep -qi "vga.*intel"; then \
		INTEL_FOUND=1; \
		echo "✓ Intel GPU detected:"; \
		lspci | grep -i "vga.*intel" | sed 's/^/  /'; \
		if [ -e /dev/dri/renderD128 ]; then \
			echo "  /dev/dri/renderD128: available"; \
			ls -la /dev/dri/renderD128 | sed 's/^/  /'; \
		else \
			echo "  /dev/dri/renderD128: NOT FOUND"; \
		fi; \
		echo ""; \
	fi; \
	if lspci 2>/dev/null | grep -qi "vga.*amd\|vga.*radeon\|3d.*amd\|3d.*radeon"; then \
		AMD_FOUND=1; \
		echo "✓ AMD GPU detected:"; \
		lspci | grep -i "vga.*amd\|vga.*radeon\|3d.*amd\|3d.*radeon" | sed 's/^/  /'; \
		echo ""; \
	fi; \
	if [ $$NVIDIA_FOUND -eq 0 ] && [ $$INTEL_FOUND -eq 0 ] && [ $$AMD_FOUND -eq 0 ]; then \
		echo "✗ No GPU detected (CPU only)"; \
		echo ""; \
	fi; \
	echo "=== Recommended Build ==="; \
	echo ""; \
	if [ $$NVIDIA_FOUND -eq 1 ]; then \
		echo "→ Use: make docker-up-cuda"; \
		echo "  (NVIDIA GPU with CUDA support)"; \
		echo "  Set HBMON_INFERENCE_BACKEND=cuda in .env"; \
		echo ""; \
	fi; \
	if [ $$INTEL_FOUND -eq 1 ]; then \
		echo "→ Use: make docker-up-intel"; \
		echo "  (Intel GPU support)"; \
		echo "  Set HBMON_INFERENCE_BACKEND=openvino-gpu in .env"; \
		echo ""; \
	fi; \
	if [ $$AMD_FOUND -eq 1 ]; then \
		echo "→ AMD GPU detected but not currently supported"; \
		echo "  Use: make docker-up (CPU-only build)"; \
		echo "  Note: ROCm support could be added in the future"; \
		echo ""; \
	fi; \
	if [ $$NVIDIA_FOUND -eq 0 ] && [ $$INTEL_FOUND -eq 0 ] && [ $$AMD_FOUND -eq 0 ]; then \
		echo "→ Use: make docker-up"; \
		echo "  (CPU-only build)"; \
		echo ""; \
	fi

docker-build: ## Build docker images
	docker compose build --build-arg PYTORCH_INDEX_URL=$(PYTORCH_INDEX_URL)

docker-up: ## Start docker compose (build if needed)
	docker compose build --build-arg PYTORCH_INDEX_URL=$(PYTORCH_INDEX_URL)
	docker compose up -d

docker-build-cuda: ## Build docker images with NVIDIA CUDA support
	docker compose build --build-arg PYTORCH_INDEX_URL=$(PYTORCH_GPU_INDEX_URL)

docker-up-cuda: ## Start docker compose with NVIDIA CUDA support (build if needed)
	docker compose build --build-arg PYTORCH_INDEX_URL=$(PYTORCH_GPU_INDEX_URL)
	docker compose up -d

docker-build-intel: ## Build docker images with Intel GPU support
	docker compose build --build-arg INSTALL_OPENVINO=1

docker-up-intel: ## Start docker compose with Intel GPU support (build if needed)
	docker compose build --build-arg INSTALL_OPENVINO=1
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

clean-openvino-cache: ## Clear OpenVINO model cache (forces model re-export on next startup)
	sudo rm -rf $(DATA_DIR)/openvino_cache
	@echo "OpenVINO cache cleared. Models will be re-exported on next worker startup."

test-stream-list: ## List observation videos available for test streaming
	$(UV_RUN) python scripts/extract_test_videos.py --limit 10

test-stream-start: ## Start RTSP test server with latest observation video
	@VIDEO=$$($(UV_RUN) python scripts/extract_test_videos.py --paths-only -f --limit 1 | head -1); \
	if [ -z "$$VIDEO" ]; then echo "No videos found"; exit 1; fi; \
	echo "Starting test stream with: $$VIDEO"; \
	$(UV_RUN) python scripts/rtsp_test_server.py "$$VIDEO"

test-detection: ## Test YOLO detection on observation snapshots with diagnostics
	$(UV_RUN) python scripts/test_detection.py --batch --limit 5

check-health: ## Check pipeline health for silent failures (DB, YOLO, CLIP, files)
	$(UV_RUN) python scripts/check_pipeline_health.py

test-e2e: ## End-to-end detection test (starts RTSP server, restarts worker, monitors)
	./scripts/run_detection_test.sh
