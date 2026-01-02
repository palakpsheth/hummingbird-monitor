# Agent Instructions

Please follow the instructions in [`.github/copilot-instructions.md`](.github/copilot-instructions.md).
- **Package Manager**: `uv` (primary), `pip` (fallback)
- **Framework**: FastAPI + Jinja2 for web UI
- **ML Stack**: PyTorch, Ultralytics YOLO, OpenCLIP
- **Database**: PostgreSQL (production), SQLite (testing/local fallback)
- **Cache/Queue**: Redis
- **Deployment**: Docker Compose (multi-container: web, stream, worker, proxy, db, redis, wyze-bridge)

## Code Style & Conventions

### General Python Style

- Follow **PEP 8** conventions
- Use **Ruff** for linting (config in `pyproject.toml`)
- Line length: **110 characters** (configured in `[tool.ruff]`)
- Target Python version: **3.11**
- Type hints: Use modern type annotations (`from __future__ import annotations`)

### Docstrings

- Use triple-quoted docstrings for modules, classes, and functions
- Module docstrings should describe purpose, key classes/functions, and design goals
- Include usage examples in docstrings when helpful
- Document environment variables and configuration options in module docstrings

### Code Organization

- **Source code**: `src/hbmon/` (package structure)
- **Tests**: `tests/` (pytest-based, uses SQLite in-memory or file-based for isolation)
- **Configuration**: Environment variables + `/data/config.json` (persisted settings)
- **Media**: `/media` (snapshots + clips)

### Import Style

- Use `from __future__ import annotations` at the top of each module
- Group imports: standard library, third-party, local modules
- Prefer explicit imports over wildcard imports
- Handle optional dependencies gracefully (e.g., SQLAlchemy, FastAPI)

### Naming Conventions

- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Environment variables**: Prefix with `HBMON_` (e.g., `HBMON_RTSP_URL`, `HBMON_FPS_LIMIT`)
- **Private functions/methods**: Single leading underscore (`_helper_function`)

### Error Handling

- Use explicit exception types
- Provide clear error messages
- Gracefully handle missing optional dependencies with informative `RuntimeError`
- Log errors appropriately (use logging module when available)

## Dependencies & Optional Imports

The codebase is designed to allow importing core modules without requiring heavy dependencies (PyTorch, FastAPI, etc.).

### Required Dependencies

- `numpy>=1.26,<2.0`
- Standard library modules (json, os, pathlib, dataclasses)

### Optional Dependencies

- **SQLAlchemy / asyncpg**: For database models and PostgreSQL async connectivity
- **Redis**: For caching and state management
- **FastAPI/Uvicorn/Gunicorn**: For web UI and streaming (only needed when running web server)
- **PyTorch/YOLO/CLIP**: For ML inference (only needed when running worker)
- **OpenCV**: For video processing (only needed when running worker)

When adding code that uses optional dependencies:
- Check availability with try/except blocks
- Provide clear error messages when required at runtime
- Maintain dataclass stubs for core models to enable testing

## Testing

### Test Framework

- **pytest** with **pytest-cov** for coverage
- Tests are in `tests/` directory
- Run tests: `uv run pytest -n auto --verbose` or `pytest -n auto --verbose`
- Run with coverage: `uv run pytest -n auto --verbose --cov=hbmon --cov-report=term --cov-report=html`

### Test Principles

- Tests should work without heavy dependencies (no SQLAlchemy, FastAPI, PyTorch)
- Use dataclass stubs when database models aren't available
- Mock external services (RTSP streams, file I/O when appropriate)
- Test core logic, configuration parsing, helper functions, and edge cases
- Aim for broad coverage of non-ML code paths

### CRITICAL: Always Run Ruff and Full Pytest

**Before committing ANY changes to this repository, you MUST run Ruff and the full pytest suite.**
These checks can take several minutes; be patient and wait for them to complete.

The exact commands and options you are required to run are documented once in the
**“Before Every Commit”** section later in this file. Treat that checklist as the
single source of truth and follow it for **all** code changes, regardless of size.
### Pre-commit Hooks

The repository uses **pre-commit** to run checks before commits:
- Ruff linting (`ruff check`)
- Full test suite with coverage
- Install hooks: `uv run pre-commit install`
- Run manually: `uv run pre-commit run --all-files`

## Build & Development

### Using `uv` for Development

**IMPORTANT: Always use `uv` for local development and testing.**

This repository uses **`uv`** as the primary package manager and task runner.

Why `uv`?
- Faster than traditional `pip`
- Consistent with CI/CD pipeline
- Ensures reproducible environments
- Integrates with pre-commit hooks

**Do not use plain `pip` or system Python** unless `uv` is unavailable in your environment.

### Local Development Setup

```bash
# Install uv (if not already installed)
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Install dependencies (editable mode with dev dependencies)
uv pip install -e ".[dev]" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

# Run tests
uv run pytest -n auto --verbose -q

# Run web server (development mode)
uv run uvicorn hbmon.web:app --reload --host 0.0.0.0 --port 8000

# Run worker (requires RTSP access)
HBMON_RTSP_URL="rtsp://..." uv run python -m hbmon.worker
```

### Linting

```bash
# Run Ruff linter
uv run ruff check .

# Auto-fix issues (when safe)
uv run ruff check . --fix
```

### Docker Development

```bash
# Build and run with Docker Compose
docker compose up -d --build

# View logs
docker compose logs -f hbmon-worker
docker compose logs -f hbmon-web

# Stop containers
docker compose down
```

## Configuration

### Environment Variables

Key environment variables (with defaults):

**Infrastructure & Database**
- `HBMON_DB_ASYNC_URL`: PostgreSQL connection string (default: `postgresql+asyncpg://hbmon:hbmon@hbmon-db:5432/hbmon`)
- `HBMON_REDIS_URL`: Redis connection string (default: `redis://hbmon-redis:6379/0`)

**RTSP & Camera**
- `HBMON_RTSP_URL`: RTSP stream URL (required for worker)
- `HBMON_CAMERA_NAME`: Camera identifier (default: "hummingbirdcam")

**Detection (YOLO)**
- `HBMON_DETECT_CONF`: Detection confidence threshold (default: ~0.25-0.35)
- `HBMON_DETECT_IOU`: IoU threshold for NMS (default: ~0.45)
- `HBMON_MIN_BOX_AREA`: Minimum bounding box area in pixels (default: ~600)
- `HBMON_FPS_LIMIT`: Frame processing rate (default: ~8)

**Event Control**
- `HBMON_COOLDOWN_SECONDS`: Seconds between events (default: ~2-6)
- `HBMON_CLIP_SECONDS`: Video clip duration (default: ~2.0)

**Species Classification (CLIP)**
- `HBMON_MIN_SPECIES_PROB`: Minimum probability for species label (default: ~0.35)

**Re-identification**
- `HBMON_MATCH_THRESHOLD`: Cosine distance threshold for matching (default: ~0.25)
- `HBMON_EMA_ALPHA`: Exponential moving average weight (default: ~0.10)

**Paths**
- `HBMON_DATA_DIR`: Database and config directory (default: `/data`)
- `HBMON_MEDIA_DIR`: Media storage directory (default: `/media`)
- `HBMON_DEVICE`: PyTorch device (`cpu` or `cuda`, default: `cpu`)

### Persistent Configuration

User settings are stored in `/data/config.json` and include:
- ROI (region of interest) coordinates
- Tuned detection thresholds
- Other runtime-adjusted parameters

The config UI (`/config`) exposes the following grouped settings:

**Performance & Recording:**
- `fps_limit`: FPS limit for frame processing (1-60)
- `clip_seconds`: Duration of recorded clips (1-30 seconds)
- `cooldown_seconds`: Cooldown between triggers (0-120 seconds)

**Detection (YOLO):**
- `detect_conf`: Detection confidence threshold (0.05-0.95)
- `detect_iou`: IOU threshold for NMS (0.05-0.95)
- `min_box_area`: Minimum bounding box area (pixels)

**Classification (CLIP):**
- `min_species_prob`: Minimum species probability (0-1)
- `crop_padding`: Padding around bbox for CLIP crop (0-0.5)

**Re-Identification:**
- `match_threshold`: Cosine distance threshold for re-ID (0-1)
- `ema_alpha`: Prototype update weight (0-1)

**Background Subtraction:**
- `bg_subtraction_enabled`: Enable/disable motion filtering
- `bg_motion_threshold`: Pixel difference threshold (0-255)
- `bg_motion_blur`: Gaussian blur kernel size (odd integer)
- `bg_min_overlap`: Required motion overlap fraction (0-1)
- `bg_log_rejected`: Log motion-rejected candidates
- `bg_rejected_cooldown_seconds`: Cooldown for rejected logging (0-60)
- `bg_rejected_save_clip`: Save clips for rejected candidates
- `bg_save_masks`: Save motion masks for observations
- `bg_save_mask_overlay`: Save mask overlay images

**Display:**
- `timezone`: IANA timezone name or "local"

## Machine Learning Components

### YOLO Detection

- Uses Ultralytics YOLO for bird detection (COCO 'bird' class)
- Model weights cached automatically
- Processes frames within ROI if configured
- Returns bounding boxes with confidence scores

### CLIP Classification

- Uses OpenCLIP for zero-shot hummingbird species classification
- Embeddings are L2-normalized float32 vectors
- Supports multiple hummingbird species (Anna's, Allen's, Rufous, Costa's, etc.)
- Custom prompts for species classification

### Re-identification (Re-ID)

- Individual tracking via cosine similarity of CLIP embeddings
- Prototype embeddings updated with exponential moving average (EMA)
- Creates new individuals when embedding distance exceeds threshold
- Supports split review for incorrectly merged individuals

## Architecture Notes

### Web Application (`hbmon.web`)

- FastAPI app with Jinja2 templates
- No authentication (LAN-only by design)
- Routes: dashboard, observations, individuals, ROI calibration
- API endpoints for health checks, latest frame, ROI config
- Export endpoints for CSV and media bundles

### Worker (`hbmon.worker`)

- Continuously processes RTSP stream
- Detection → classification → re-ID → database write pipeline
- Respects cooldown periods to avoid duplicate events
- Records snapshots (JPEG) and clips (MP4)

### Database Schema

- **individuals**: Cluster of observations (one per identified bird)
- **observations**: Individual detection events with metadata
- **embeddings**: Per-observation embedding vectors (linked via observation UUID)

### Infrastructure Components

- **PostgreSQL**: Primary persistent storage for metadata, observations, and embeddings
- **Redis**: Fast state management, caching, and potentially future task queuing
- **Nginx (Proxy)**: Unified entry point for web and stream services
- **Wyze Bridge**: Local RTSP provider for Wyze cameras

## Documentation & README Updates

**ALWAYS keep `AGENTS.md` and `.github/copilot-instructions.md` in sync.** If you modify this file, you **must** update `AGENTS.md` (in the project root) to reflect the changes, and vice versa.

**ALWAYS update the README when making changes that affect:**

### User-Facing Changes
- New features or functionality that users will interact with
- Changes to web UI, API endpoints, or CLI commands
- New configuration options or environment variables
- Changes to deployment instructions or Docker setup
- New dependencies that affect installation
- Changes to data models or export formats

### Setup & Configuration
- New prerequisites or system requirements
- Changes to installation steps or Docker Compose configuration
- New tuning parameters or recommendations
- Changes to file paths, volumes, or persistent storage
- Updates to GPU acceleration setup
- New troubleshooting steps or common issues

### Architecture & Design
- New containers or services in Docker Compose
- Changes to database schema (if user-visible)
- New ML models or classification approaches
- Changes to the processing pipeline
- Performance improvements worth noting

### Examples of README Sections to Update
- **Quick Start**: If setup steps change
- **Configuration**: When adding/modifying environment variables
- **Tuning Guide**: When adding new tunable parameters
- **Architecture**: When adding/removing containers or changing data flow
- **Troubleshooting**: When fixing bugs or adding known issues
- **GPU Acceleration**: When changing CUDA/device requirements
- **Exports & Backups**: When changing export formats or locations
- **Directory Layout**: When adding new directories or restructuring

### When README Updates Are NOT Required
- Internal refactoring that doesn't change behavior
- Bug fixes that restore documented behavior
- Test-only changes
- CI/CD configuration changes (unless affecting user workflows)
- Minor code style improvements

**After updating the README:**
- Ensure consistency with the actual implementation
- Verify examples and commands are correct
- Check that version numbers and badges are current
- Maintain the existing structure and style

## Common Tasks

### Before Every Commit

**MANDATORY VALIDATION STEPS:**

1. **Run Ruff:**
   ```bash
   UV_INDEX_URL=https://download.pytorch.org/whl/cpu UV_EXTRA_INDEX_URL=https://pypi.org/simple uv run ruff check .
   ```

2. **Run Full Test Suite:**
   ```bash
   UV_INDEX_URL=https://download.pytorch.org/whl/cpu UV_EXTRA_INDEX_URL=https://pypi.org/simple uv run pytest -n auto --verbose --cov=hbmon --cov-report=term
   ```

3. **Verify:**
   - All linting errors are fixed
   - All tests pass
   - Coverage is maintained or improved

**Do not proceed with commit if either check fails.**

### Adding New Features

1. Ensure code follows existing style conventions
2. Add appropriate type hints
3. Handle optional dependencies gracefully
4. Write tests for new functionality
5. **Update documentation (README, docstrings)**
   - **README**: Update if the feature is user-facing or changes setup/configuration
   - **Docstrings**: Update for all new/modified functions, classes, and modules
   - See "Documentation & README Updates" section for detailed guidance
6. **Run ruff and full pytest before committing (MANDATORY)**

### Modifying ML Pipeline

1. Keep ML code in appropriate modules (`clip_model.py`, `clustering.py`, `worker.py`)
2. Document new hyperparameters as environment variables
3. **Add tuning guidance to README if needed**
   - Update "Tuning Guide" section with new parameters
   - Include default values and practical recommendations
   - Explain the impact of parameter changes
4. Ensure CPU-friendly defaults (GPU optional)
5. Test with and without GPU availability
6. **Update README if GPU requirements change**
7. **Run ruff and full pytest before committing (MANDATORY)**

### Database Changes

1. Update SQLAlchemy models in `models.py` (ensure compatibility with PostgreSQL)
2. Maintain dataclass stubs for testing (which may still use SQLite/memory)
3. **Update schema documentation**
   - Update README "Architecture" section (see "Persistent storage" and database schema details) if schema changes
   - Add a dedicated "Database Schema" subsection if more detailed documentation is needed
   - Document new tables, columns, or relationships
4. Handle PostgreSQL migrations (Alembic or similar if implemented)
5. **Update README if backup/export procedures change** (e.g., `pg_dump`)
6. **Run ruff and full pytest before committing (MANDATORY)**

### Docker & Deployment Changes

1. **Update README when modifying Docker setup:**
   - Changes to `docker-compose.yml` (new services, ports, volumes)
   - Changes to `Dockerfile` (new dependencies, build steps)
   - New environment variables in `.env.example`
   - Changes to nginx configuration
2. **Update "Quick Start" section** if deployment steps change
3. **Update "Architecture" section** if containers are added/removed
4. Ensure `.env.example` stays in sync with documented variables
5. Test the full Docker Compose setup before committing
6. **Run ruff and full pytest before committing (MANDATORY)**

### Adding Configuration Options

1. Add environment variable with `HBMON_` prefix
2. Ensure new environment variables can pass through `.env` to Docker (e.g., `docker-compose.yml`) and
   into the running application (web/worker) as appropriate.
3. Document in code module docstrings
4. **Update README in relevant sections:**
   - Add to "Configuration" > "Environment Variables"
   - Update tuning guide if it's a tunable parameter
   - Add to `.env.example` with helpful comments
5. Provide sensible defaults
6. **Run ruff and full pytest before committing (MANDATORY)**

## Security Considerations

- **No authentication**: This is intentional for LAN-only deployment
- **No external exposure**: Never expose ports to the internet
- **Input validation**: Validate user inputs in web forms and API endpoints
- **File paths**: Sanitize file paths to prevent directory traversal
- **SQL injection**: Use SQLAlchemy ORM to prevent SQL injection

## CI/CD

The repository uses **GitHub Actions** for continuous integration:

- Workflow file: `.github/workflows/ci.yml`
- Runs on: Ubuntu latest
- Python version: 3.11
- Steps:
  1. Install `uv`
  2. Create virtual environment
  3. Install dependencies (editable + dev)
  4. Run Ruff linter
  5. Run tests with coverage
  6. Post coverage report on PRs
  7. Generate and commit coverage badge
  8. Docker build smoke test

### Coverage Badge

- Coverage badge is auto-generated and committed to `coverage.svg`
- Displayed in README
- Updated on every CI run

## Additional Guidelines

### Performance

- Optimize for CPU by default (most users won't have GPU)
- Provide GPU acceleration as optional enhancement
- Use ROI to reduce processing overhead
- Tune FPS and detection thresholds for CPU constraints

### Maintainability

- Keep modules focused and cohesive
- Minimize dependencies between modules
- Document design decisions in module docstrings
- Prefer simple, inspectable algorithms over complex ones

### User Experience

- Optimize web UI for Android Chrome
- Keep setup simple (Docker Compose preferred)
- Provide clear tuning guidance
- Export capabilities for data portability

## Questions or Issues?

When in doubt:
- Follow existing code patterns in the repository
- Check the README for setup and usage guidance
- Review test files for examples
- **If you modify user-facing functionality, configuration, or setup: UPDATE THE README**
- **ALWAYS run `UV_INDEX_URL=https://download.pytorch.org/whl/cpu UV_EXTRA_INDEX_URL=https://pypi.org/simple uv run ruff check .` before committing**
- **ALWAYS run `UV_INDEX_URL=https://download.pytorch.org/whl/cpu UV_EXTRA_INDEX_URL=https://pypi.org/simple uv run pytest -n auto --verbose --cov=hbmon` before committing**
- Use these commands to ensure changes don't break tests or violate code style

## Video Storage and Streaming

### Architecture

Videos are stored **uncompressed** on disk to preserve pristine quality for ML training. When streaming to browsers, they are compressed **on-the-fly** using FFmpeg with intelligent caching:

1. **Storage**: Videos recorded by `recorder.py` in uncompressed format
2. **Streaming**: `/api/video/{obs_id}` endpoint compresses on first view
3. **Caching**: Compressed versions cached in `/media/.cache/compressed/`
4. **Cache Validation**: Auto-refreshes when source video changes (mtime check)

### Utilities (`observation_tools.py`)

The `observation_tools` module provides utilities for processing existing observations:

- `extract_video_metadata()` - Extract FPS, resolution, duration, codec from video files
- `update_observation_video_metadata()` - Update observation extra_json with video metadata
- `process_observations_batch()` - Batch process all observations to extract/update metadata
- `clean_compressed_cache()` - Clean up old cached compressed files (age + size limits)
- `validate_video_file()` - Check if video file is valid and readable

### When Adding Video-Related Features

1. **Test both uncompressed and compressed paths**
2. **Consider ML training use case** - avoid degrading uncompressed source
3. **Update cache management** if adding new compression parameters
4. **Add tests** for new video processing utilities
5. **Update README** with new configuration options
6. **Document performance impact** (compression time, cache size)

