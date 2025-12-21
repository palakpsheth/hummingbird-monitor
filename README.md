# Hummingbird Monitor (`hbmon`)

<!-- The coverage badge is generated automatically by the GitHub Actions workflow.
     The `coverage.svg` file at the repository root is updated after every run of
     the CI pipeline via the `tj-actions/coverage-badge-py` action.  Referencing
     the SVG file directly ensures the badge always reflects the latest test
     coverage without relying on an external badge service. -->
![Coverage](coverage.svg)


LAN-only hummingbird monitoring system designed for a **Linux x64 mini PC**.

It ingests a **Wyze Cam v3** stream via **`wyze-bridge` (RTSP)**, detects birds using **YOLO**, records short clips, classifies hummingbird species using **CLIP/OpenCLIP**, and assigns sightings to **individual birds** via **embedding-based re-identification**.

The web UI is optimized for **Android Chrome** and is intentionally **no-login / no-password** (only safe on a trusted home LAN).

---

## What you get

### Core features
- **Automatic capture** of a snapshot + short video clip when a bird enters the feeder ROI
- **Species label + probability** logged for each event
- **Individual re-identification**: “bird A vs bird B” using image embeddings
- **Name individuals** in the UI
- **Counts & last-seen** stats per individual
- **Cross-linked** navigation:
  - Individuals list → individual page → all observations
  - Observation detail → linked individual
- **Exports**
  - CSV: observations and individuals
  - `tar.gz` bundle: snapshots + clips (generated under `/data/exports/`)

### Web UI pages
- **Dashboard**: recent observations + top individuals
- **Observations**: filterable gallery + detail page
- **Individuals**: sortable list + detail page
- **ROI calibration**: draw a box on the latest snapshot

---

## Architecture

### Containers (recommended)
- **wyze-bridge**: logs into Wyze and exposes RTSP streams
- **hbmon-worker**: reads RTSP, runs detection + CLIP + re-ID, writes to SQLite
- **hbmon-web**: FastAPI + Jinja UI, serves `/media` and exports
- **nginx** (optional): reverse proxy on port 80 (nice “just open IP” UX)

### Persistent storage
- `/data` (volume): SQLite DB + `config.json` + exports
- `/media` (volume): snapshots + clips

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

### 2) Run
```bash
docker compose up -d --build
```

### 3) Open the UI on your phone
- Main UI (nginx on port 80): `http://<mini-pc-ip>/`
- Direct to app (port 8000): `http://<mini-pc-ip>:8000`
- Wyze-bridge UI: `http://<mini-pc-ip>:5000`

### 4) Find your mini PC’s LAN IP
On the mini PC:
```bash
hostname -I
# or
ip a
```

---

## Recommended setup steps

### Calibrate ROI (biggest accuracy + performance win)
1. Wait until you have at least one snapshot (a bird visit helps).
2. Open **Calibrate ROI**.
3. Drag a tight rectangle around the feeder/perch region.
4. Save.

Why ROI matters:
- reduces CPU (YOLO runs on fewer pixels)
- reduces false positives (trees/sky won’t trigger)
- improves crop quality for CLIP embeddings → better re-ID

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

### Individual re-identification
- `HBMON_MATCH_THRESHOLD` (cosine distance; default ~0.25)
  - **Lower** (e.g. 0.20): stricter matching → more new individuals created
  - **Higher** (e.g. 0.30): more merging → may incorrectly merge different birds
- `HBMON_EMA_ALPHA` (prototype update weight; default ~0.10)
  - Lower reduces drift, higher adapts faster

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
- `/data` (SQLite DB + config + exports)
- `/media` (all images/videos)

---

## Local development (uv)

From repo root:

```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest -q
uv run uvicorn hbmon.web:app --reload --host 0.0.0.0 --port 8000
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
  - add motion gating + background subtraction to reduce triggers

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

The `hbmon` package includes a suite of unit tests located in the `tests/` directory.  These
tests are designed to exercise the core logic of the application without requiring
heavy optional dependencies such as SQLAlchemy, FastAPI, PyTorch or OpenCV.  To run
the tests you need to install [pytest](https://pytest.org) and the
[`pytest‑cov`](https://github.com/pytest-dev/pytest-cov) plugin.  Once installed,
you can execute the test suite with coverage.  If you’re using [uv](https://github.com/astral-sh/uv)
to manage your virtual environment (as in the CI pipeline), prefix the command with
`uv run` so that the correct interpreter and dependencies are used:

```bash
# Running with uv
uv run pytest --cov=hbmon --cov-report=term --cov-report=html

# Or, if you installed dependencies with plain pip
pytest --cov=hbmon --cov-report=term --cov-report=html
```

These commands print a coverage summary to the terminal and produce an HTML report
in the `htmlcov/` directory.  The coverage badge displayed at the top of this
README reflects the percentage of code covered by the tests.  If you modify the
code or add new tests, be sure to regenerate the coverage report so the badge
stays accurate.

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
