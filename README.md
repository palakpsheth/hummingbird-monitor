# Hummingbird Monitor (`hbmon`)

<!-- The coverage badge uses shields.io with a JSON endpoint generated in CI.
     The GitHub Actions workflow writes `coverage-badge.json` at the repository
     root after tests run, and the badge URL points at that file. This keeps the
     badge up-to-date without relying on GitHub's cached image URLs. -->
![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/palakpsheth/hummingbird-monitor/main/coverage-badge.json&cacheSeconds=300)
[![CI](https://github.com/palakpsheth/hummingbird-monitor/actions/workflows/ci.yml/badge.svg)](https://github.com/palakpsheth/hummingbird-monitor/actions/workflows/ci.yml)
![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


LAN-only hummingbird monitoring system designed for a **Linux x64 mini PC**.

It ingests a **Wyze Cam v3** stream via **`wyze-bridge` (RTSP)**, detects birds using **YOLO**, records short clips, classifies hummingbird species using **CLIP/OpenCLIP**, and assigns sightings to **individual birds** via **embedding-based re-identification**.

The web UI is optimized for **Android Chrome** and is intentionally **no-login / no-password** (only safe on a trusted home LAN).

---

## Table of contents
- [What you get](#what-you-get)
- [Architecture](#architecture)
- [Quick start (Docker Compose)](#quick-start-docker-compose)
- [Database & cache configuration](#database--cache-configuration)
- [Developer shortcuts (Makefile)](#developer-shortcuts-makefile)
- [Recommended setup steps](#recommended-setup-steps)
- [Wyze Bridge configuration](#wyze-bridge-configuration)
- [Tuning guide (practical)](#tuning-guide-practical)
- [GPU acceleration (if available)](#gpu-acceleration-if-available)
- [Exports & backups](#exports--backups)
- [Local development (uv)](#local-development-uv)
- [GitHub Actions CI](#github-actions-ci)
- [License](#license)
- [Troubleshooting](#troubleshooting)
- [Current limitations (by design or early version)](#current-limitations-by-design-or-early-version)
- [Where this can be improved next (ideas)](#where-this-can-be-improved-next-ideas)
- [Testing & Coverage](#testing--coverage)
- [Pre‑commit hooks](#pre-commit-hooks)
- [Coverage badge (Shields.io)](#coverage-badge-shieldsio)
- [Directory layout](#directory-layout)

---

## What you get

### Core features
- **Automatic capture** of a snapshot + short video clip when a bird enters the feeder ROI
- **Species label + probability** logged for each event
- **Individual re-identification**: “bird A vs bird B” using image embeddings
- **Name individuals** in the UI
- **Counts & last-seen** stats per individual
- **Prototypical snapshots** shown per individual (list + detail, embedding-closest when available)
- **Multiple snapshot views** per observation: raw, annotated, CLIP crop, plus background reference when enabled
- **Background image configuration**: define a standard background picture without hummingbirds
- **Cross-linked** navigation:
  - Individuals list → individual page → all observations
  - Observation detail → linked individual
- **Observation detail metadata**: detector confidence, bbox area (frame/ROI ratios), IoU thresholds, sensitivity settings, identification match details (species + individual), plus a copy-ready `original_observation` JSON panel
- **Exports**
  - CSV: observations and individuals
  - `tar.gz` bundle: snapshots + clips (generated under `/data/exports/`)
  - Per-observation integration test bundle (downloadable from the observation detail page after completing
    the metadata form, including a human-verified true/false selection), containing:
    - `metadata.json` with `expected`, `source`, `sensitivity_tests`, and `original_observation`
    - `snapshot.jpg` (raw snapshot)
    - `clip.mp4` (video clip)
    - Optional `background.jpg` (reference background frame used for background subtraction)
    - Optional `snapshot_annotated.jpg` and `snapshot_clip.jpg` when available
  - Missing sensitivity/identification fields are backfilled with current defaults when exporting older observations.

### Web UI pages
- **Dashboard**: recent observations + top individuals (shows a live snapshot with ROI overlay, play/pause controls for the live feed, a detection health indicator alongside stream diagnostics when streaming, and pauses when the tab is hidden)
- **Observations**: filterable, sortable table (including dynamic extra metadata fields such as
  detector confidence) with compact thumbnails, column visibility checklist (sensitivity fields hidden
  by default), links to raw/annotated/clip snapshots + video, multi-select + bulk delete, and horizontal
  scrolling for wide metadata + detail page
- **Individuals**: sortable list + detail page with prototypical snapshot
- **ROI calibration**: draw a box on the latest snapshot
- **Config**: tune detection thresholds, re-ID thresholds, and background subtraction settings
- **Background image**: configure a reference background (select from observations, upload, or capture a live snapshot)
- **API Docs**: interactive Swagger UI for API exploration (`/docs`)
- Page headers mirror the footer status line (current time, time zone, version, and commit).

---

## Architecture

### Containers (recommended)
- **wyze-bridge**: logs into Wyze and exposes RTSP streams (using [IDisposable fork](https://github.com/IDisposable/docker-wyze-bridge) for improved performance and camera support)
- **hbmon-db**: PostgreSQL database for concurrent reads/writes with pooling
- **hbmon-redis**: Redis cache for hot query results (latest observations, health checks)
- **hbmon-worker**: reads RTSP, runs detection + CLIP + re-ID, writes to PostgreSQL
- **hbmon-web**: FastAPI + Jinja UI served by Gunicorn + Uvicorn workers, serves `/media` and exports
- **hbmon-stream**: dedicated FastAPI worker for `/api/stream.mjpeg` MJPEG streaming
- **nginx** (optional): reverse proxy on port 80 (nice “just open IP” UX)

### Container Startup Order & Healthchecks

The `docker-compose.yml` uses healthchecks to ensure containers start in the correct order:

```
wyze-bridge (healthy) → hbmon-db (healthy) → hbmon-web (healthy) → hbmon-worker
                       → hbmon-redis (healthy) → hbmon-stream (healthy) → hbmon-proxy
```

| Container     | Healthcheck                           | Wait for                       |
|---------------|---------------------------------------|--------------------------------|
| wyze-bridge   | HTTP check on port 5000               | -                              |
| hbmon-db      | `pg_isready`                          | -                              |
| hbmon-redis   | `redis-cli ping`                      | -                              |
| hbmon-web     | HTTP check on `/health` endpoint      | wyze-bridge + db + redis       |
| hbmon-stream  | HTTP check on `/health` endpoint      | wyze-bridge + db + redis       |
| hbmon-worker  | Process check for `hbmon.worker`      | wyze-bridge + hbmon-web + db   |
| hbmon-proxy   | HTTP check on port 80                 | hbmon-web + hbmon-stream       |

This ensures the database is initialized by hbmon-web before the worker starts.

### Persistent storage
- `/data` (volume): `config.json`, exports, background image
- `/data/postgres` (volume): PostgreSQL data files
- `/data/redis` (volume): Redis append-only files
- `/media` (volume): snapshots (raw, annotated, CLIP crop) + clips

---

## Quick start (Docker Compose)

### Prereqs
- Docker + Docker Compose on the Linux mini PC
- Wyze account credentials (for `wyze-bridge`)
- Your Wyze camera name as seen by `wyze-bridge` (e.g. `hummingbirdcam`)

### 1) Configure environment
Copy the example and edit it:

```bash
cp .env.example .env
nano .env
```

Key variables:

- `WYZE_EMAIL`, `WYZE_PASSWORD`: Wyze credentials for `wyze-bridge`
- `HBMON_RTSP_URL`: RTSP URL provided by `wyze-bridge`, typically:

```text
rtsp://wyze-bridge:8554/<YOUR_CAMERA_NAME>
```

> Tip: once `wyze-bridge` is running, you can also open its UI at `http://<mini-pc-ip>:5000` to confirm stream names.

Optional:
- `GIT_COMMIT`: footer label shown in the UI; usually injected by CI/build. Defaults to `unknown`.
- `HBMON_WEB_WORKERS`: number of Gunicorn workers for the web app (default: 4)
- `HBMON_DB_ASYNC_URL`: PostgreSQL connection string for the web app + worker (async driver)
- `HBMON_DB_URL`: sync connection string for optional tooling/tests (unused in Docker)
- `HBMON_REDIS_URL`: Redis cache connection string (optional but recommended)
- `HBMON_REDIS_TTL_SECONDS`: Redis cache TTL in seconds (default: 5)
- `HBMON_SQLITE_BUSY_TIMEOUT_MS`: SQLite busy timeout for fallback/testing (default: 5000)

### 2) Run
```bash
docker compose up -d --build
```

By default, Docker builds use the **CPU-only** PyTorch wheels to keep image sizes smaller. See
[GPU acceleration](#gpu-acceleration-if-available) for CUDA-enabled builds.

### 3) Open the UI on your phone
- Main UI (nginx on port 80): `http://<mini-pc-ip>/`
- Direct to app (port 8000): `http://<mini-pc-ip>:8000`
- API Documentation (Swagger): `http://<mini-pc-ip>:8000/docs`
- API Documentation (ReDoc): `http://<mini-pc-ip>:8000/redoc`
- Wyze-bridge UI: `http://<mini-pc-ip>:5000`

### 4) Find your mini PC’s LAN IP
On the mini PC:
```bash
hostname -I
# or
ip a
```

---

## Database & cache configuration

The Docker setup runs PostgreSQL for the database and Redis for short-lived cache entries. Both the
web service and worker use the async driver (`HBMON_DB_ASYNC_URL`).

Pool tuning (optional):
- `HBMON_DB_POOL_SIZE`: base pool size (default: 5)
- `HBMON_DB_MAX_OVERFLOW`: extra connections allowed during bursts (default: 10)
- `HBMON_DB_POOL_TIMEOUT`: seconds to wait for a connection (default: 30)
- `HBMON_DB_POOL_RECYCLE`: seconds before recycling connections (default: 1800)

Cache tuning (optional):
- `HBMON_REDIS_URL`: Redis connection string
- `HBMON_REDIS_TTL_SECONDS`: cache TTL in seconds (default: 5)

SQLite fallback/testing:
- `HBMON_DB_URL`: sync SQLite/Postgres URL for local tooling/tests (default: SQLite path)
- `HBMON_SQLITE_BUSY_TIMEOUT_MS`: SQLite busy timeout in milliseconds (default: 5000)

---

## Developer shortcuts (Makefile)

For local development, the repo includes a `Makefile` with common tasks. The targets use
`uv` to manage the virtual environment and run commands (matching CI expectations).
Note: `docker-up`/`docker-up-gpu` split build and run steps because build args must be
passed to `docker compose build`, not `docker compose up`.

```bash
make venv            # create .venv via uv
make sync            # install dev dependencies from pyproject.toml
make sync-gpu        # install dev dependencies with CUDA-enabled PyTorch wheels
make lint            # ruff check
make test            # full pytest + coverage
make test-unit       # unit tests + coverage (marker: not integration)
make test-integration # integration/UI tests + coverage (marker: integration)
make docker-build    # docker compose build
make docker-up       # docker compose build + up -d
make docker-build-gpu # docker compose build (CUDA-enabled PyTorch)
make docker-up-gpu   # docker compose build + up -d (CUDA-enabled PyTorch)
make docker-down     # docker compose down
make clean-db        # remove local database file only (defaults to ./data)
make clean-media     # remove local media files (defaults to ./data/media)
make clean-data      # remove all local data (defaults to ./data)
```

Run `make help` to list all available targets.

---

## Recommended setup steps

### Calibrate ROI (biggest accuracy + performance win)
1. Open **Calibrate ROI** (uses a live snapshot from the RTSP feed when available).
2. Drag a tight rectangle around the feeder/perch region.
3. Save.

If the live feed is unavailable, the calibration page falls back to the most recent observation snapshot. If there
are no observations yet, you will see a placeholder image prompting you to start the worker and wait for a visit.

Why ROI matters:
- reduces CPU (YOLO runs on fewer pixels)
- reduces false positives (trees/sky won’t trigger)
- improves crop quality for CLIP embeddings → better re-ID

---

## Wyze Bridge configuration

This project uses the [IDisposable fork](https://github.com/IDisposable/docker-wyze-bridge) of docker-wyze-bridge, which provides improved performance and camera support.

### Key features
- **Healthcheck**: The wyze-bridge container includes a healthcheck that monitors the web UI on port 5000. Other containers wait for wyze-bridge to be healthy before starting.
- **NET_MODE=LAN**: Forces local streaming from cameras for lowest latency
- **ON_DEMAND=False**: Keeps streams active for faster response
- **QUALITY setting**: Configurable stream quality (default: HD60)

### Environment variables for Wyze Bridge
- `WYZE_EMAIL`, `WYZE_PASSWORD`: Required Wyze account credentials
- `WYZE_API_KEY`, `WYZE_API_ID`: Optional API credentials (consult wyze-bridge docs)
- `WYZE_MFA_TYPE`, `WYZE_MFA_CODE`: Optional 2FA configuration
- `WYZE_REGION`: Optional region override
- `WYZE_QUALITY`: Stream quality setting (default: HD60)
  - Options: SD30, SD60, HD30, HD60, 2K30, 2K60 (2K for supported cameras)

### Network mode
The wyze-bridge container runs with `network_mode: host` for optimal performance. This means:
- Ports are exposed directly on the host (no Docker port mapping needed)
- RTSP is available at `rtsp://<host-ip>:8554/<camera-name>`
- Web UI is available at `http://<host-ip>:5000`

---


## Tuning guide (practical)

Most tuning is via environment variables (Docker) or `/data/config.json` (persisted settings).

### Detection (YOLO)
- `HBMON_DETECT_CONF` (default ~0.25–0.35)
  - Increase to reduce false triggers
  - Decrease if it misses birds
- `HBMON_DETECT_IOU` (default ~0.45)
  - Adjust if you see duplicate boxes or weird suppression behavior
- `HBMON_MIN_BOX_AREA` (default ~600)
  - Increase if tiny moving leaves/bugs trigger detections
- `HBMON_FPS_LIMIT` (default ~8)
  - Lower for CPU-constrained machines
  - Typical CPU sweet spot: **6–10**

### Event frequency control
- `HBMON_COOLDOWN_SECONDS` (default ~2–6)
  - Increase if one visit creates many events
  - Decrease if you want finer-grained logging per visit

### Clips
- `HBMON_CLIP_SECONDS` (default ~2.0)
  - Increase if you want more “arrival + feeding + departure” context

> Note: the current worker records a **post-trigger** clip (starting right after the detection).
> A true **pre-trigger buffer** (ring buffer of frames) is a great next upgrade; see “Ideas for improvement”.

### Species classification (CLIP)
- `HBMON_MIN_SPECIES_PROB` (default ~0.35)
  - Raise if species labels are noisy
  - Lower if too many are forced into “unknown”
- `HBMON_CROP_PADDING` (default ~0.05)
  - Controls how much padding is added around the detected bird bbox before species classification
  - **Lower** (e.g. 0.02): tighter crop, focuses more on the bird itself (may improve species ID)
  - **Higher** (e.g. 0.18): more background context included (may help in some cases)
  - Reduced from 0.18 to 0.05 to better focus on the bird for improved species identification

### Individual re-identification
- `HBMON_MATCH_THRESHOLD` (cosine distance; default ~0.25)
  - **Lower** (e.g. 0.20): stricter matching → more new individuals created
  - **Higher** (e.g. 0.30): more merging → may incorrectly merge different birds
- `HBMON_EMA_ALPHA` (prototype update weight; default ~0.10)
  - Lower reduces drift, higher adapts faster

### Background subtraction (motion filtering)
When a background image is configured via the UI (`/background`), the worker can use it to filter out false positives by detecting motion.
These tuning values can be adjusted on the **Config** page (`/config`) and are persisted to `config.json`. Environment variables
continue to override them when set.

- `HBMON_BG_SUBTRACTION` (default "1")
  - Set to "0" to disable background subtraction even if a background image is configured
- `HBMON_BG_MOTION_THRESHOLD` (default 30)
  - Pixel difference threshold (0-255) for detecting change
  - Lower values are more sensitive to motion
  - Higher values require more significant change to trigger
- `HBMON_BG_MOTION_BLUR` (default 5)
  - Gaussian blur kernel size for noise reduction (must be an odd positive integer, e.g., 3, 5, 7)
  - Higher values smooth out noise but may miss small birds
- `HBMON_BG_MIN_OVERLAP` (default 0.15)
  - Minimum fraction of detection area that must have motion (0.0-1.0)
  - Lower values accept detections with less motion overlap
  - Higher values require more of the detection area to show change
- `HBMON_DEBUG_BG` (default "0")
  - Set to "1" to enable debug logging for motion mask errors

**How it works:**
1. Configure a background image showing the feeder without any birds (upload, pick an observation, or capture a live snapshot)
2. The worker computes a motion mask by comparing each frame to the background
3. YOLO detections are filtered: only those overlapping significantly with motion areas are kept
4. This reduces false positives from static objects or lighting changes

### Debugging and diagnostics
- `HBMON_DEBUG_YOLO` (default "0")
  - Set to "1" to enable extra YOLO detection logging
- `HBMON_DEBUG_SAVE_FRAMES` (default "0")
  - Set to "1" to write periodic debug frames to disk
- `HBMON_DEBUG_EVERY_SECONDS` (default 10)
  - Interval (seconds) between debug snapshots when saving frames
- `HBMON_YOLO_IMGSZ` (default 1280)
  - Override the YOLO inference image size (larger values increase detail and CPU/GPU cost)

### MJPEG live stream tuning
The MJPEG endpoint (`/api/stream.mjpeg`) supports bandwidth/CPU tuning via environment variables:

- `HBMON_MJPEG_FPS` (default 10)
  - Target MJPEG frame rate for the live stream
- `HBMON_MJPEG_MAX_WIDTH` (default 1280)
  - Maximum output width; frames are downscaled if larger
  - Set to 0 to disable width limiting
- `HBMON_MJPEG_MAX_HEIGHT` (default 720)
  - Maximum output height; frames are downscaled if larger
  - Set to 0 to disable height limiting
- `HBMON_MJPEG_JPEG_QUALITY` (default 70)
  - JPEG quality for the MJPEG stream (10–100)
- `HBMON_MJPEG_ADAPTIVE` (default 0)
  - Set to 1 to enable adaptive degradation when encoding is slow
- `HBMON_MJPEG_MIN_FPS` (default 4)
  - Lowest FPS when adaptive degradation is active
- `HBMON_MJPEG_MIN_QUALITY` (default 40)
  - Lowest JPEG quality when adaptive degradation is active
- `HBMON_MJPEG_FPS_STEP` (default 1)
  - Step size for adaptive FPS adjustments
- `HBMON_MJPEG_QUALITY_STEP` (default 5)
  - Step size for adaptive quality adjustments

---

## GPU acceleration (if available)

### 1) Check if you have an NVIDIA GPU
On the host:
```bash
lspci | grep -i nvidia || true
nvidia-smi || true
```

If `nvidia-smi` works, you likely can use GPU acceleration.

### 2) Install NVIDIA Container Toolkit (host)
You need the NVIDIA runtime so Docker containers can see the GPU.

On Ubuntu, this is usually:
- install NVIDIA drivers
- install `nvidia-container-toolkit`
- restart Docker

(Exact commands vary by distro; consult NVIDIA’s official docs for your OS.)

### 3) Run containers with GPU access
In compose, you typically add a GPU reservation to the services that run PyTorch/YOLO/CLIP (the **worker**).

Common patterns:
- Compose v2 device requests (recommended)
- Or `runtime: nvidia` (older)

Example (conceptual):

```yaml
services:
  hbmon-worker:
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

Then inside the worker container:
- set `HBMON_DEVICE=cuda`
- ensure you’re using a CUDA-enabled PyTorch build

To build CUDA-enabled images, use the Makefile targets (or pass the build arg yourself):

```bash
make docker-build-gpu
make docker-up-gpu
```

These targets set `PYTORCH_INDEX_URL` to the CUDA wheel index (default: `cu121`). You can override
the index for a different CUDA version if needed:

```bash
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 docker compose build
```

### 4) Verify inside the container
```bash
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

If CUDA is available, the worker can run CLIP and YOLO faster.

### 5) Notes on non-NVIDIA GPUs
- Intel iGPU: you can potentially use OpenVINO for some workloads, but it’s a separate effort.
- AMD GPU: ROCm can work for PyTorch, but it’s platform-specific and often harder than NVIDIA on a mini PC.

---

## Exports & backups

### From the UI (top nav)
- **Obs CSV**: observations table
- **Ind CSV**: individuals table
- **Bundle tar.gz**: snapshots + clips (generated under `/data/exports/`)

### What to back up
If you want a complete backup, copy:
- `/data` (config + exports + background + PostgreSQL + Redis data)
- `/media` (all images/videos)

---

## Local development (uv)

From repo root:

```bash
uv venv
uv pip install -e ".[dev]" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
uv run pytest -q
uv run uvicorn hbmon.web:app --reload --host 0.0.0.0 --port 8000
```

To use CUDA-enabled PyTorch wheels locally, swap the index URL (or run `make sync-gpu`):

```bash
uv pip install -e ".[dev]" --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
```

Run the worker locally (requires RTSP access from your host):
```bash
HBMON_RTSP_URL="rtsp://..." uv run python -m hbmon.worker
```

---

## GitHub Actions CI

A typical `.github/workflows/ci.yml` for this repo:
- uses `uv`
- runs `ruff` + `pytest`
- (optional) builds Docker images as a smoke test

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Troubleshooting

### “VLC is unable to open the MRL …”
- Confirm `wyze-bridge` is running and the stream name matches your camera
- Try opening the stream from the host:
  ```bash
  ffplay rtsp://<host-or-container>:8554/<camera>
  ```
- If you see timeouts like `IOTC_ER_TIMEOUT`, it usually means:
  - wrong camera name
  - Wyze login issue
  - camera offline / weak Wi‑Fi
  - upstream Wyze connectivity trouble

### Worker isn’t producing snapshots
- Verify `HBMON_RTSP_URL` is correct in the worker container
- Check logs:
  ```bash
  docker compose logs -f hbmon-worker
  ```
- Confirm ROI isn’t accidentally set to a tiny/empty area (recalibrate)

### Video clips won't play in browser
If video clips don't stream properly in Chrome/Firefox:

1. **Check the observation detail page** - It now includes an inline HTML5 video player
   that shows error messages if the video fails to load, along with video resolution and duration.

2. **Use the video diagnostics API** to check file existence, size, and codec:
   ```bash
   curl http://<your-server>/api/video_info/<observation_id>
   ```
   This returns information including:
   - File existence and size
   - Detected codec (codec_hint)
   - Browser compatibility status (browser_compatible)
   - Playback warning with FFmpeg conversion command if needed

3. **FFmpeg auto-conversion**: The worker now automatically converts videos to H.264
   using FFmpeg when OpenCV falls back to a non-browser-compatible codec (mp4v/XVID).
   Check logs for conversion status:
   ```bash
   docker compose logs hbmon-worker | grep -i "ffmpeg\|h264\|converting"
   ```

4. **Common causes**:
   - **Codec incompatibility**: The worker tries H.264 codecs (avc1, H264) first,
     then falls back to mp4v/XVID. When this happens, FFmpeg automatically converts
     the video to H.264 for browser compatibility.
   - **File not found**: Video file may not exist on disk (check video_info API)
   - **Corrupted file**: Worker may have been interrupted during recording
   - **Proxy issues**: Ensure nginx is configured to forward Range headers (already
     configured in the default `nginx.conf`)

5. **Manual conversion**: If you have old clips that don't play, convert them with:
   ```bash
   ffmpeg -i input.mp4 -c:v libx264 -preset fast -crf 23 -movflags +faststart output.mp4
   ```

6. **Try direct access**: Open the video URL directly (bypassing nginx):
   ```
   http://<your-server>:8000/media/clips/2025-12-23/xxxxx.mp4
   ```

### UI looks stale after an update
The web app now sends no-cache headers on every response to prevent stale browser assets. If you
still see old content, clear the browser cache or do a hard reload.

---

## Current limitations (by design or early version)

- **No authentication**: this is LAN-only. Don’t expose ports to the internet.
- **Species classification**: CLIP + prompts is “pretty good” but not a field guide.
- **Re-ID**: embeddings are a visual fingerprint, not perfect; use:
  - Observation detail → reassign (future enhancement if you want)
  - Individuals list → merge/split (split review is implemented in UI; merge can be added next)

---

## Where this can be improved next (ideas)

### Better “eventing”
- **True pre-trigger buffer**: maintain a ring buffer of frames so clips include the *moment before* the bird arrives.
- Better visit segmentation:
  - track a bird across frames (e.g., ByteTrack/DeepSORT)
  - end a visit only after N seconds of no detections

### Better hummingbird-specific detection
- YOLO COCO “bird” class is generic.
- Improvements:
  - fine-tune YOLO on hummingbird feeder images
  - add a second-stage classifier “hummingbird vs other bird”
  - add motion gating + background subtraction to reduce triggers ✅ (now implemented using the configured background image)

### Better re-ID
- Current: cosine matching to a prototype embedding.
- Improvements:
  - store multiple prototypes per individual (mixture model)
  - add a quality score for crops (blur/size/occlusion)
  - active learning loop: the UI confirms/renames → model improves

### Better species classification
- Train a small classifier on local species (Anna’s, Allen’s, Rufous, Costa’s…)
- Use iNaturalist / eBird labeled data or your own confirmed snapshots

### Security and sharing
- Optional basic auth for peace of mind
- Read-only “guest” mode
- TLS on LAN (optional)

---

## Testing & Coverage

The `hbmon` package includes a comprehensive test suite with both **unit tests** and **integration tests**. The coverage badge and PR reports always reflect coverage of **all tests**.

### Running Tests

```bash
# Run all tests with coverage (default)
uv run pytest --cov=hbmon --cov-report=term --cov-report=html

# Run only unit tests (skip integration tests)
uv run pytest -m "not integration"

# Run only integration tests
uv run pytest -m integration
```

### Unit Tests

Unit tests exercise the core logic without requiring heavy ML dependencies. They are lightweight and fast.

### Integration Tests

Integration tests require ML dependencies (PyTorch, YOLO, CLIP) and real test data. They are marked with `@pytest.mark.integration`.

### Test Data Structure

Integration test data is located in `tests/integration/test_data/`. Each test case folder contains:

```text
tests/integration/test_data/
├── README.md                    # Full documentation
├── flying_0/                    # Test case: flying hummingbird
│   ├── snapshot.jpg             # Captured image
│   ├── clip.mp4                 # Video clip
│   ├── background.jpg           # Optional background reference frame
│   └── metadata.json            # Expected labels & sensitivity tests
├── perched_0/                   # Test case: perched bird
│   └── ...
├── feeding_0/                   # Test case: bird at feeder
│   └── ...
└── edge_cases/                  # Edge case scenarios
    ├── low_light_0/
    └── motion_blur_0/
```

Each `metadata.json` includes:
- **expected**: Detection/classification ground truth (when human_verified=true)
- **sensitivity_tests**: Parameter variations to test
- **original_observation**: Raw observation data from the worker

See `tests/integration/test_data/README.md` for the complete schema.

### Coverage Reports

Test coverage is automatically reported on every PR. The coverage badge at the top of this README reflects coverage of **all tests (unit + integration)**.

```bash
# Generate HTML coverage report
uv run pytest --cov=hbmon --cov-report=html
# Open htmlcov/index.html in browser
```

## Pre‑commit hooks

To make it easy to run the same checks locally that the CI pipeline performs, this repository includes a [pre‑commit](https://pre-commit.com) configuration.  Pre‑commit installs a Git hook that automatically runs a set of commands before each commit.  In this project the hook runs `ruff` (our linter) and the full test suite with coverage, mirroring the steps defined in the CI workflow.  If any of these checks fail, the commit will be aborted so you can fix the issues before pushing.

To set up and use the pre‑commit hooks:

1. Install the development dependencies, including the `pre‑commit` package.  If you’re using [uv](https://github.com/astral-sh/uv) as in the CI pipeline, you can run:

   ```bash
   uv pip install -e ".[dev]"
   ```

   Alternatively, with plain Python and pip:

   ```bash
   python -m pip install -e ".[dev]"
   ```

2. Install the Git hook scripts.  When using `uv`, run pre‑commit via
   `uv run` so that it uses the same virtual environment you installed the
   dependencies into:

   ```bash
   # With uv
   uv run pre-commit install

   # Or, with plain pip
   pre-commit install
   ```

   This needs to be done once per clone; it configures Git to run the hooks
   on every commit.

3. (Optional) Run all hooks against the entire repository to check everything at once.  Again,
   use `uv run` if you installed dependencies via uv:

   ```bash
   # With uv
   uv run pre-commit run --all-files

   # Or, with plain pip
   pre-commit run --all-files
   ```

The hooks will run automatically before each commit.  They execute
`ruff` against the changed files and run `pytest --cov=hbmon --cov-report=term` (with
`PYTHONPATH=src` set in the hook configuration) to ensure the tests still pass.
Running the hooks locally helps catch issues early and keeps the CI pipeline green.

## Coverage badge (Shields.io)

The coverage badge uses a shields.io JSON endpoint so the image stays fresh without manual README
edits. To configure this in another repo:

1. Ensure your CI produces a `coverage.xml` (or similar) report.
2. Add a CI step to write a `coverage-badge.json` file at the repository root with the shields.io
   schema (see `.github/workflows/ci.yml` in this repo).
3. Commit the generated `coverage-badge.json` so it is available at
   `https://raw.githubusercontent.com/<owner>/<repo>/<default-branch>/coverage-badge.json`.
4. Reference the badge in your README using:
   `https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/<owner>/<repo>/<default-branch>/coverage-badge.json&cacheSeconds=300`.

## Directory layout

```text
hummingbird-monitor/
  docker-compose.yml
  nginx.conf
  .env.example
  pyproject.toml
  Dockerfile
  src/
    hbmon/
      __init__.py
      web.py
      worker.py
      config.py
      db.py
      models.py
      schema.py
      clip_model.py
      clustering.py
      templates/
      static/
  tests/
  .github/
    workflows/
      ci.yml
```
