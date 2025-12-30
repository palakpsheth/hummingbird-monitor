# Agent Instructions: hummingbird-monitor (`hbmon`)

This document provides essential information for AI agents working on the `hbmon` repository. It adapts and summarizes key details from `.github/copilot-instructions.md`.

## Project Overview

`hbmon` (hummingbird-monitor) is a LAN-only hummingbird monitoring system using YOLO for detection, CLIP for species classification, and embedding-based re-identification for tracking individual birds.

- **Primary Language**: Python 3.11
- **Package Manager**: `uv` (required for local development)
- **Frameworks**: FastAPI + Jinja2 (Web UI), Gunicorn/Uvicorn
- **ML Stack**: PyTorch, Ultralytics YOLO, OpenCLIP
- **Database**: PostgreSQL (production), SQLite (testing/local)
- **Cache/Queue**: Redis
- **Deployment**: Docker Compose (multi-container)

## Critical Constraints & Mandatory Steps

### Before Every Commit/Submission
You **MUST** perform the following validation steps. Do not skip these, even for small changes.

1.  **Linting (Ruff)**:
    ```bash
    UV_INDEX_URL=https://download.pytorch.org/whl/cpu UV_EXTRA_INDEX_URL=https://pypi.org/simple uv run ruff check .
    ```
2.  **Full Test Suite**:
    ```bash
    UV_INDEX_URL=https://download.pytorch.org/whl/cpu UV_EXTRA_INDEX_URL=https://pypi.org/simple uv run pytest -n auto --verbose --cov=hbmon --cov-report=term
    ```
3.  **Verification**:
    - All linting errors must be fixed.
    - All tests must pass.
    - Coverage must be maintained or improved.

## Code Style & Conventions

- **Standard**: Follow **PEP 8**.
- **Linting**: **Ruff** is the source of truth (config in `pyproject.toml`).
- **Line Length**: **110 characters**.
- **Type Hints**: Use modern annotations (`from __future__ import annotations`).
- **Imports**: Group by standard library, third-party, and local. Prefer explicit imports.
- **Naming**: `snake_case` (functions/vars), `PascalCase` (classes), `UPPER_SNAKE_CASE` (constants).
- **Env Vars**: Prefix with `HBMON_` (e.g., `HBMON_RTSP_URL`).
- **Private Elements**: Use a single leading underscore (`_helper`).

## Repository Structure

- `src/hbmon/`: Core package source code.
- `tests/`: Pytest suite.
- `data/`: Local data, config, and SQLite DB (persisted via volumes in Docker).
- `media/`: Snapshots and video clips.
- `docker-compose.yml`: Multi-container orchestration.

## Development Workflow

### Local Setup (using `uv`)
```bash
uv venv
uv pip install -e ".[dev]" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
uv run pre-commit install
```

### Running Components
- **Web Server**: `uv run uvicorn hbmon.web:app --reload`
- **Worker**: `HBMON_RTSP_URL="..." uv run python -m hbmon.worker`
- **Docker**: `docker compose up -d --build`

## Testing Strategy

- **Framework**: `pytest` with `pytest-cov`.
- **Isolation**: Tests should work without heavy dependencies (SQLAlchemy, FastAPI, PyTorch) where possible. Use dataclass stubs or mocks.
- **Coverage**: Aim for broad coverage of non-ML code paths.

## Documentation Requirements

- **AGENTS.md / Copilot Instructions**: This file must be kept in sync with `.github/copilot-instructions.md`. When updating one, you **must** update the other to ensure consistency.
- **README**: Update whenever user-facing features, configuration (env vars), setup steps, or architecture changes.
- **Docstrings**: Triple-quoted docstrings for all modules, classes, and functions. Include usage examples where helpful.
- **Change Log**: Update when significant features or fixes are implemented.

## Architecture & Components

- **hbmon-worker**: Core processing pipeline (RTSP → YOLO → CLIP → Re-ID → DB).
- **hbmon-web**: FastAPI web interface and API.
- **hbmon-stream**: Dedicated MJPEG streaming service.
- **hbmon-db**: PostgreSQL for metadata and embeddings.
- **hbmon-redis**: Caching and state management.
- **wyze-bridge**: RTSP provider for Wyze cameras.
- **nginx (proxy)**: Reverse proxy for unified access.

## ML Pipeline Details

- **Detection**: Ultralytics YOLO (COCO 'bird' class) within configured ROI.
- **Classification**: OpenCLIP zero-shot species classification with custom prompts.
- **Re-identification**: Cosine similarity of CLIP embeddings; Prototypes updated via EMA.
- **Persistence**: PostgreSQL for metadata/embeddings; Local filesystem for media (JPEG/MP4).
- **GPU**: Optional. System must be optimized for CPU-only execution by default.
- **Security**: LAN-only design; no authentication by default. Do not expose to the public internet.
