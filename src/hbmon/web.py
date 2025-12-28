# src/hbmon/web.py
"""
FastAPI + Jinja2 web UI for hbmon (LAN-only, no auth).

Routes:
- /                      Dashboard
- /observations           Gallery + filters
- /observations/{id}      Observation detail (with inline video player)
- /individuals            Individuals list
- /individuals/{id}       Individual detail + heatmap + rename
- /individuals/{id}/split_review  Suggest A/B split & review UI
- /individuals/{id}/split_apply   Apply split assignments
- /calibrate              ROI calibration page
- /background             Background image configuration page

API:
- /api/health
- /api/frame.jpg          Latest snapshot (or placeholder)
- /api/live_frame.jpg     Single snapshot from live RTSP feed
- /api/roi  (GET/POST)    Get/set ROI (POST accepts form)
- /api/video_info/{id}    Video file diagnostics for troubleshooting

API Documentation:
- /docs                   Swagger UI (interactive API explorer)
- /redoc                  ReDoc viewer (alternative API docs)
- /openapi.json           OpenAPI 3.1 specification

Exports:
- /export/observations.csv
- /export/individuals.csv
- /export/media_bundle.tar.gz

Notes:
- Media is served from HBMON_MEDIA_DIR (default /media)
- DB is served from HBMON_DATA_DIR (default /data)

MJPEG streaming environment variables:
- HBMON_MJPEG_FPS: target MJPEG frame rate (default: 10)
- HBMON_MJPEG_MAX_WIDTH: maximum MJPEG frame width (default: 1280, 0 disables)
- HBMON_MJPEG_MAX_HEIGHT: maximum MJPEG frame height (default: 720, 0 disables)
- HBMON_MJPEG_JPEG_QUALITY: JPEG quality for MJPEG stream (default: 70)
- HBMON_MJPEG_ADAPTIVE: enable adaptive degradation (default: 0)
- HBMON_MJPEG_MIN_FPS: lowest adaptive FPS floor (default: 4)
- HBMON_MJPEG_MIN_QUALITY: lowest adaptive JPEG quality (default: 40)
- HBMON_MJPEG_FPS_STEP: FPS step for adaptive changes (default: 1)
- HBMON_MJPEG_QUALITY_STEP: quality step for adaptive changes (default: 5)
"""

from __future__ import annotations

import asyncio
import atexit
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
import importlib.util
import io
import json
from json import JSONDecodeError
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
from urllib.parse import urlsplit
import time
import weakref
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterable, Iterator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import anyio
import numpy as np

"""
FastAPI web application for hbmon.

This module defines the web routes and API for the hummingbird monitor.  It
attempts to import FastAPI and SQLAlchemy at runtime.  If either of those
dependencies is missing, the :func:`make_app` function will raise a
``RuntimeError`` when called.  This allows the rest of the package to be
imported in minimal environments without installing heavy dependencies.
"""

try:
    from fastapi import Depends, FastAPI, Form, HTTPException, Request  # type: ignore
    from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse  # type: ignore
    from fastapi.staticfiles import StaticFiles  # type: ignore
    from fastapi.templating import Jinja2Templates  # type: ignore
    _FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover
    # FastAPI is not available; define stubs to allow import but not use.
    class _StubExc(Exception):
        pass

    Depends = FastAPI = Form = Request = object  # type: ignore
    HTTPException = _StubExc  # type: ignore
    FileResponse = HTMLResponse = RedirectResponse = StreamingResponse = object  # type: ignore
    StaticFiles = object  # type: ignore
    Jinja2Templates = object  # type: ignore
    _FASTAPI_AVAILABLE = False

try:
    from sqlalchemy import delete, desc, func, select  # type: ignore
    from sqlalchemy.exc import OperationalError  # type: ignore
    from sqlalchemy.orm import Session  # type: ignore
    from sqlalchemy.ext.asyncio import AsyncSession  # type: ignore
    _SQLA_AVAILABLE = True
except Exception:  # pragma: no cover
    delete = desc = func = select = None  # type: ignore
    Session = object  # type: ignore
    AsyncSession = object  # type: ignore
    OperationalError = Exception  # type: ignore
    _SQLA_AVAILABLE = False

ALLOWED_REVIEW_LABELS = ["true_positive", "false_positive", "false_negative"]


from hbmon import __version__
from hbmon.config import (
    Roi,
    background_dir,
    background_image_path,
    data_dir,
    env_bool,
    env_float,
    env_int,
    ensure_dirs,
    load_settings,
    media_dir,
    roi_to_str,
    save_settings,
)
from hbmon.cache import cache_get_json, cache_set_json
from hbmon.db import (
    get_async_db,
    get_session_factory,
    init_async_db,
    init_db,
    is_async_db_available,
)
from hbmon.models import Embedding, Individual, Observation, _to_utc
from hbmon.schema import HealthOut, RoiOut
from hbmon.clustering import l2_normalize, suggest_split_two_groups

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Derived from this module path; not user-controlled, safe for git cwd.
_GIT_PATH = shutil.which("git")


def _load_cv2() -> Any:
    existing = sys.modules.get("cv2")
    if existing is not None:
        return existing

    if importlib.util.find_spec("cv2") is None:
        raise HTTPException(status_code=503, detail="OpenCV not available for streaming")

    import cv2  # type: ignore

    return cv2


@dataclass(frozen=True)
class MJPEGSettings:
    target_fps: float
    max_width: int
    max_height: int
    base_quality: int
    adaptive_enabled: bool
    min_fps: float
    min_quality: int
    fps_step: float
    quality_step: int


def _load_mjpeg_settings() -> MJPEGSettings:
    target_fps = env_float("HBMON_MJPEG_FPS", 10.0)
    if target_fps <= 0:
        target_fps = 10.0
    max_width = env_int("HBMON_MJPEG_MAX_WIDTH", 1280)
    max_height = env_int("HBMON_MJPEG_MAX_HEIGHT", 720)
    base_quality = env_int("HBMON_MJPEG_JPEG_QUALITY", 70)
    adaptive_enabled = env_bool("HBMON_MJPEG_ADAPTIVE", False)
    min_fps = env_float("HBMON_MJPEG_MIN_FPS", 4.0)
    min_quality = env_int("HBMON_MJPEG_MIN_QUALITY", 40)
    fps_step = env_float("HBMON_MJPEG_FPS_STEP", 1.0)
    quality_step = env_int("HBMON_MJPEG_QUALITY_STEP", 5)

    target_fps = max(0.5, target_fps)
    min_fps = max(0.5, min(min_fps, target_fps))
    fps_step = max(0.1, fps_step)
    base_quality = max(10, min(base_quality, 100))
    min_quality = max(10, min(min_quality, base_quality))
    quality_step = max(1, quality_step)

    return MJPEGSettings(
        target_fps=target_fps,
        max_width=max_width,
        max_height=max_height,
        base_quality=base_quality,
        adaptive_enabled=adaptive_enabled,
        min_fps=min_fps,
        min_quality=min_quality,
        fps_step=fps_step,
        quality_step=quality_step,
    )


def _configure_mjpeg_capture(cap: Any, cv2: Any) -> None:
    """Configure VideoCapture for low latency MJPEG streaming."""
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        # Buffer size may not be supported on all backends; ignore silently
        pass


def _resize_mjpeg_frame(frame: Any, cv2: Any, settings: MJPEGSettings) -> Any:
    if settings.max_width <= 0 and settings.max_height <= 0:
        return frame
    height, width = frame.shape[:2]
    scale_w = (settings.max_width / width) if settings.max_width > 0 else 1.0
    scale_h = (settings.max_height / height) if settings.max_height > 0 else 1.0
    scale = min(1.0, scale_w, scale_h)
    if scale >= 1.0:
        return frame
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _update_mjpeg_adaptive(
    current_fps: float,
    current_quality: int,
    settings: MJPEGSettings,
    encode_duration: float,
) -> tuple[float, int]:
    if not settings.adaptive_enabled or current_fps <= 0:
        return current_fps, current_quality

    budget = 1.0 / current_fps
    new_fps = current_fps
    new_quality = current_quality
    if encode_duration > budget:
        if new_quality > settings.min_quality:
            new_quality = max(settings.min_quality, new_quality - settings.quality_step)
        if new_fps > settings.min_fps:
            new_fps = max(settings.min_fps, new_fps - settings.fps_step)
    elif encode_duration < budget * 0.5:
        if new_quality < settings.base_quality:
            new_quality = min(settings.base_quality, new_quality + settings.quality_step)
        if new_fps < settings.target_fps:
            new_fps = min(settings.target_fps, new_fps + settings.fps_step)
    return new_fps, new_quality


def _decode_fourcc(raw_value: float | int) -> str:
    if raw_value is None:
        return ""
    code = int(raw_value)
    if code == 0:
        return ""
    chars = [chr((code >> (8 * i)) & 0xFF) for i in range(4)]
    cleaned = "".join(chars).strip()
    return cleaned if cleaned.isprintable() else ""


class FrameBroadcaster:
    """Background video capture that keeps a cached JPEG for MJPEG streaming."""

    def __init__(self, rtsp_url: str, settings: MJPEGSettings | None = None) -> None:
        self._rtsp_url = rtsp_url
        self._settings = settings or _load_mjpeg_settings()
        self._current_fps = self._settings.target_fps
        self._current_quality = self._settings.base_quality
        self._frame_interval = 1.0 / self._current_fps if self._current_fps > 0 else 0.0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._clients = 0
        self._latest_frame: bytes | None = None
        self._latest_timestamp = 0.0
        self._last_frame_shape: tuple[int, int] | None = None
        self._last_frame_size = 0
        self._last_encode_duration = 0.0
        self._source_width = 0
        self._source_height = 0
        self._source_fps = 0.0
        self._source_codec = ""

    @property
    def rtsp_url(self) -> str:
        return self._rtsp_url

    @property
    def settings(self) -> MJPEGSettings:
        return self._settings

    def add_client(self) -> None:
        thread = None
        with self._lock:
            self._clients += 1
            thread = self._thread
            if thread is not None and thread.is_alive() and not self._stop_event.is_set():
                return
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name="hbmon-mjpeg", daemon=True)
            self._thread.start()

    def remove_client(self) -> None:
        thread = None
        with self._lock:
            self._clients = max(0, self._clients - 1)
            if self._clients == 0:
                self._stop_event.set()
                thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)

    def shutdown(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)

    def latest_frame(self) -> tuple[bytes | None, float]:
        with self._lock:
            return self._latest_frame, self._latest_timestamp

    def diagnostics(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            latest_timestamp = self._latest_timestamp
            last_frame_size = self._last_frame_size
            last_frame_shape = self._last_frame_shape
            current_fps = self._current_fps
            current_quality = self._current_quality
            settings = self._settings
            clients = self._clients
            source_width = self._source_width
            source_height = self._source_height
            source_fps = self._source_fps
            source_codec = self._source_codec
            last_encode_duration = self._last_encode_duration

        age_s = None if latest_timestamp <= 0 else max(0.0, now - latest_timestamp)
        output_resolution = None
        if last_frame_shape is not None:
            output_resolution = f"{last_frame_shape[1]}x{last_frame_shape[0]}"
        source_resolution = None
        if source_width > 0 and source_height > 0:
            source_resolution = f"{source_width}x{source_height}"

        return {
            "clients": clients,
            "source": {
                "resolution": source_resolution,
                "fps": source_fps if source_fps > 0 else None,
                "codec": source_codec or None,
            },
            "frame": {
                "last_frame_age_s": age_s,
                "last_frame_size_bytes": last_frame_size if last_frame_size > 0 else None,
                "encode_ms": round(last_encode_duration * 1000.0, 2) if last_encode_duration > 0 else None,
                "output_resolution": output_resolution,
            },
            "mjpeg": {
                "adaptive_enabled": settings.adaptive_enabled,
                "target_fps": settings.target_fps,
                "current_fps": current_fps,
                "min_fps": settings.min_fps,
                "base_quality": settings.base_quality,
                "current_quality": current_quality,
                "min_quality": settings.min_quality,
                "max_width": settings.max_width,
                "max_height": settings.max_height,
            },
        }

    def _open_capture(self, cv2: Any) -> Any | None:
        cap = cv2.VideoCapture(self._rtsp_url)
        if not cap.isOpened():
            cap.release()
            return None
        _configure_mjpeg_capture(cap, cv2)
        source_width = 0
        source_height = 0
        source_fps = 0.0
        source_codec = ""
        if hasattr(cap, "get"):
            try:
                source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                source_codec = _decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC) or 0)
            except Exception:
                source_width = 0
                source_height = 0
                source_fps = 0.0
                source_codec = ""
        with self._lock:
            self._source_width = source_width
            self._source_height = source_height
            self._source_fps = source_fps
            self._source_codec = source_codec
        return cap

    def _run(self) -> None:
        cv2 = _load_cv2()
        cap = None
        last_frame_time = 0.0
        try:
            while not self._stop_event.is_set():
                if cap is None:
                    cap = self._open_capture(cv2)
                    if cap is None:
                        time.sleep(0.5)
                        continue

                current_time = time.monotonic()
                if current_time - last_frame_time < self._frame_interval:
                    time.sleep(0.01)
                    continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.release()
                    cap = None
                    continue

                last_frame_time = current_time
                frame = _resize_mjpeg_frame(frame, cv2, self._settings)
                encode_start = time.perf_counter()
                ok, jpeg = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self._current_quality],
                )
                if not ok:
                    continue
                encode_duration = time.perf_counter() - encode_start
                self._current_fps, self._current_quality = _update_mjpeg_adaptive(
                    self._current_fps,
                    self._current_quality,
                    self._settings,
                    encode_duration,
                )
                self._frame_interval = 1.0 / self._current_fps if self._current_fps > 0 else 0.0

                frame_bytes = jpeg.tobytes()
                with self._lock:
                    self._latest_frame = frame_bytes
                    self._latest_timestamp = current_time
                    self._last_frame_shape = frame.shape[:2]
                    self._last_frame_size = len(frame_bytes)
                    self._last_encode_duration = encode_duration
        finally:
            if cap is not None:
                cap.release()
            with self._lock:
                self._thread = None


_FRAME_BROADCASTER: FrameBroadcaster | None = None
_FRAME_BROADCASTER_LOCK = threading.Lock()


def _get_frame_broadcaster(rtsp_url: str, settings: MJPEGSettings | None = None) -> FrameBroadcaster:
    global _FRAME_BROADCASTER
    with _FRAME_BROADCASTER_LOCK:
        if (
            _FRAME_BROADCASTER is None
            or _FRAME_BROADCASTER.rtsp_url != rtsp_url
            or (settings is not None and _FRAME_BROADCASTER.settings != settings)
        ):
            if _FRAME_BROADCASTER is not None:
                _FRAME_BROADCASTER.shutdown()
            _FRAME_BROADCASTER = FrameBroadcaster(rtsp_url, settings=settings)
    return _FRAME_BROADCASTER


def _shutdown_frame_broadcaster() -> None:
    global _FRAME_BROADCASTER
    with _FRAME_BROADCASTER_LOCK:
        if _FRAME_BROADCASTER is not None:
            _FRAME_BROADCASTER.shutdown()
            _FRAME_BROADCASTER = None


atexit.register(_shutdown_frame_broadcaster)


def _normalize_timezone(tz: str | None) -> str:
    txt = (tz or "").strip()
    return txt or "local"


def _read_git_head(repo_root: Path) -> str | None:
    git_path = repo_root / ".git"
    git_dir = git_path
    if git_path.is_file():
        try:
            data = git_path.read_text().strip()
        except OSError:
            return None
        if data.startswith("gitdir:"):
            rel = data.partition(":")[2].strip()
            git_dir = (git_path.parent / rel).resolve()
        else:
            return None
    head_path = git_dir / "HEAD"
    try:
        head = head_path.read_text().strip()
    except OSError:
        return None
    if not head:
        return None
    if head.startswith("ref:"):
        ref = head.partition(" ")[2].strip()
        if not ref:
            return None
        ref_path = git_dir / ref
        try:
            commit = ref_path.read_text().strip()
        except OSError:
            packed = git_dir / "packed-refs"
            try:
                with packed.open() as pf:
                    for line in pf:
                        txt = line.strip()
                        if not txt or txt.startswith("#") or txt.startswith("^"):
                            continue
                        parts = txt.split(" ", 1)
                        if len(parts) == 2 and parts[1] == ref:
                            commit = parts[0]
                            break
                    else:
                        commit = ""
            except OSError:
                commit = ""
        return commit[:7] if commit else None
    return head[:7]


def _timezone_label(tz: str | None) -> str:
    clean = _normalize_timezone(tz)
    return "Browser local" if clean.lower() == "local" else clean


def _sanitize_redirect_path(raw: str | None, default: str = "/observations") -> str:
    if not raw:
        return default
    text = str(raw)
    parsed = urlsplit(text)
    if parsed.scheme or parsed.netloc:
        return default
    if not parsed.path.startswith("/") or parsed.path.startswith("//"):
        return default
    return text


def _get_git_commit() -> str:
    env_commit = os.getenv("HBMON_GIT_COMMIT")
    # Treat "unknown" (any casing) as unset so fallback methods are tried during Docker builds
    if env_commit and env_commit.lower() != "unknown":
        return env_commit
    if _REPO_ROOT.is_dir() and _GIT_PATH is not None:
        try:
            commit = subprocess.check_output(
                [_GIT_PATH, "rev-parse", "--short", "HEAD"],
                cwd=_REPO_ROOT,
                timeout=1.0,
                shell=False,
                text=True,
            )
            cleaned = commit.strip()
            if cleaned:
                return cleaned
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            # Fallback to file-based parsing when the git CLI is unavailable or fails.
            pass
    head_commit = _read_git_head(_REPO_ROOT)
    if head_commit:
        return head_commit
    return "unknown"


_GIT_COMMIT = _get_git_commit()


# ----------------------------
# Presentation helpers
# ----------------------------

async def _run_blocking(func: Any, *args: Any, **kwargs: Any) -> Any:
    return await anyio.to_thread.run_sync(partial(func, *args, **kwargs))


def _shutdown_async_session_executors() -> None:
    for executor in list(_AsyncSessionAdapter._executors):
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=False)


class _AsyncSessionAdapter:
    _executors: "weakref.WeakSet[ThreadPoolExecutor]" = weakref.WeakSet()

    def __init__(self, session_factory: Callable[..., Session]):
        self._session_factory = session_factory
        self._session: Session | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._executors.add(self._executor)
        self._closed = False
        self._finalizer = weakref.finalize(
            self,
            self._finalize_executor,
            weakref.ref(self),
            wait=False,
        )

    async def _run(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, partial(func, *args, **kwargs))

    def _shutdown_executor(self, wait: bool) -> None:
        if self._closed:
            return
        try:
            self._executor.shutdown(wait=wait, cancel_futures=True)
        except TypeError:
            self._executor.shutdown(wait=wait)
        self._executors.discard(self._executor)
        self._closed = True

    @staticmethod
    def _finalize_executor(
        self_ref: weakref.ReferenceType["_AsyncSessionAdapter"],
        wait: bool,
    ) -> None:
        self = self_ref()
        if self is None:
            return
        self._shutdown_executor(wait=wait)

    async def _ensure_session(self) -> Session:
        if self._session is None:
            self._session = await self._run(self._session_factory)
        return self._session

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        session = await self._ensure_session()
        return await self._run(session.execute, *args, **kwargs)

    async def get(self, *args: Any, **kwargs: Any) -> Any:
        session = await self._ensure_session()
        return await self._run(session.get, *args, **kwargs)

    async def commit(self) -> None:
        session = await self._ensure_session()
        await self._run(session.commit)

    async def rollback(self) -> None:
        session = await self._ensure_session()
        await self._run(session.rollback)

    async def flush(self) -> None:
        session = await self._ensure_session()
        await self._run(session.flush)

    async def close(self) -> None:
        if self._session is not None:
            await self._run(self._session.close)
        self._shutdown_executor(wait=True)
        self._finalizer.detach()

    def __getattr__(self, name: str) -> Any:
        if self._session is None:
            self._session = self._executor.submit(self._session_factory).result()
        return getattr(self._session, name)


atexit.register(_shutdown_async_session_executors)


async def get_db_dep() -> AsyncIterator[AsyncSession | _AsyncSessionAdapter]:
    if is_async_db_available():
        async for session in get_async_db():
            yield session
    else:
        adapter = _AsyncSessionAdapter(get_session_factory())
        try:
            yield adapter
        finally:
            await adapter.close()


async def _get_latest_observation_data(
    db: AsyncSession | _AsyncSessionAdapter,
) -> dict[str, Any] | None:
    cache_key = "hbmon:latest_observation"
    cached = await cache_get_json(cache_key)
    if isinstance(cached, dict):
        required_keys = {"id", "ts_utc", "snapshot_path", "video_path"}
        if required_keys.issubset(cached.keys()):
            return cached

    latest = (
        await db.execute(select(Observation).order_by(desc(Observation.ts)).limit(1))
    ).scalars().first()
    if latest is None:
        return None

    payload = {
        "id": int(latest.id),
        "ts_utc": latest.ts_utc,
        "snapshot_path": latest.snapshot_path,
        "video_path": latest.video_path,
    }
    await cache_set_json(cache_key, payload)
    return payload


def species_to_css(label: str) -> str:
    s = (label or "").strip().lower().replace("’", "'")
    if "anna" in s:
        return "species-anna"
    if "allen" in s:
        return "species-allens"
    if "rufous" in s:
        return "species-rufous"
    if "costa" in s:
        return "species-costas"
    if "black" in s and "chinned" in s:
        return "species-black-chinned"
    if "calliope" in s:
        return "species-calliope"
    if "broad" in s and "billed" in s:
        return "species-broad-billed"
    return "species-unknown"


def get_annotated_snapshot_path(obs: Observation) -> str | None:
    """
    Get the annotated snapshot path for an observation from its extra_json.

    Returns the annotated path if available, otherwise None.
    """
    extra = obs.get_extra()
    if not extra or not isinstance(extra, dict):
        return None
    snapshots_data = extra.get("snapshots")
    if not isinstance(snapshots_data, dict):
        return None
    return snapshots_data.get("annotated_path")


async def select_prototype_observations(
    db: AsyncSession | _AsyncSessionAdapter,
    individual_ids: list[int],
) -> dict[int, Observation]:
    """
    Select a per-individual prototype observation.

    Prefer observation embeddings closest to the stored prototype embedding.
    Falls back to highest match score (then newest timestamp/id) when no
    embeddings are available.
    """
    if not individual_ids:
        return {}

    selected: dict[int, Observation] = {}

    proto_rows = (
        await db.execute(
            select(Individual).where(Individual.id.in_(individual_ids), Individual.prototype_blob.isnot(None))
        )
    ).scalars().all()
    proto_map: dict[int, np.ndarray] = {}
    for ind in proto_rows:
        vec = ind.get_prototype()
        if vec is None:
            continue
        proto_map[int(ind.id)] = l2_normalize(vec)

    if proto_map:
        emb_rows = (
            await db.execute(
                select(Embedding).where(Embedding.individual_id.in_(list(proto_map.keys())))
            )
        ).scalars().all()
        best_obs_id: dict[int, int] = {}
        best_score: dict[int, float] = {}
        for emb in emb_rows:
            if emb.individual_id is None:
                continue
            proto = proto_map.get(int(emb.individual_id))
            if proto is None:
                continue
            emb_vec = l2_normalize(emb.get_vec())
            similarity = float(np.dot(proto, emb_vec))
            if similarity > best_score.get(int(emb.individual_id), float("-inf")):
                best_score[int(emb.individual_id)] = similarity
                best_obs_id[int(emb.individual_id)] = int(emb.observation_id)
        if best_obs_id:
            obs_rows = (
                await db.execute(
                    select(Observation).where(Observation.id.in_(list(best_obs_id.values())))
                )
            ).scalars().all()
            obs_by_id = {int(o.id): o for o in obs_rows}
            for ind_id, obs_id in best_obs_id.items():
                obs = obs_by_id.get(obs_id)
                if obs is not None:
                    selected[ind_id] = obs

    remaining = [ind_id for ind_id in individual_ids if ind_id not in selected]
    if not remaining:
        return selected

    ranked = (
        select(
            Observation.id.label("obs_id"),
            func.row_number()
            .over(
                partition_by=Observation.individual_id,
                order_by=(
                    desc(Observation.match_score),
                    desc(Observation.ts),
                    desc(Observation.id),
                ),
            )
            .label("rn"),
        )
        .where(Observation.individual_id.in_(remaining))
        .subquery()
    )

    rows = (
        await db.execute(
            select(Observation)
            .join(ranked, Observation.id == ranked.c.obs_id)
            .where(ranked.c.rn == 1)
        )
    ).scalars().all()

    for obs in rows:
        if obs.individual_id is None:
            continue
        selected[int(obs.individual_id)] = obs
    return selected


def get_clip_snapshot_path(obs: Observation) -> str | None:
    """
    Get the CLIP crop snapshot path for an observation from its extra_json.

    Returns the CLIP snapshot path if available, otherwise None.
    """
    extra = obs.get_extra()
    if not extra or not isinstance(extra, dict):
        return None
    snapshots_data = extra.get("snapshots")
    if not isinstance(snapshots_data, dict):
        return None
    return snapshots_data.get("clip_path")


def get_background_snapshot_path(obs: Observation) -> str | None:
    """
    Get the background snapshot path for an observation from its extra_json.

    Returns the background snapshot path if available, otherwise None.
    """
    extra = obs.get_extra()
    if not extra or not isinstance(extra, dict):
        return None
    snapshots_data = extra.get("snapshots")
    if not isinstance(snapshots_data, dict):
        return None
    return snapshots_data.get("background_path")


def build_hour_heatmap(hours_rows: list[tuple[int, int]]) -> list[dict[str, int]]:
    """
    hours_rows: [(hour_int, count_int), ...]
    Returns 24 dicts: {"hour": h, "count": c, "level": 0..5}
    """
    counts = {int(h): int(c) for (h, c) in hours_rows}
    vals = [counts.get(h, 0) for h in range(24)]
    mx = max(vals) if vals else 0

    def lvl(c: int) -> int:
        if c <= 0 or mx <= 0:
            return 0
        frac = c / mx
        if frac <= 0.20:
            return 1
        if frac <= 0.40:
            return 2
        if frac <= 0.60:
            return 3
        if frac <= 0.80:
            return 4
        return 5

    return [{"hour": h, "count": counts.get(h, 0), "level": lvl(counts.get(h, 0))} for h in range(24)]


def pretty_json(text: str | None) -> str | None:
    """
    Best-effort pretty formatting of a JSON string.

    Returns the original text if parsing fails.
    """
    if not text:
        return None
    try:
        obj = json.loads(text)
    except JSONDecodeError:
        return text
    try:
        return json.dumps(obj, indent=4, sort_keys=True)
    except (TypeError, ValueError):
        return text


def pretty_json_obj(obj: Any | None) -> str | None:
    """
    Best-effort pretty formatting of a JSON-serializable object.

    Returns ``None`` if formatting fails.
    """
    if obj is None:
        return None
    try:
        return json.dumps(obj, indent=4, sort_keys=True)
    except (TypeError, ValueError):
        return None


def _sanitize_case_name(raw: str | None, fallback: str) -> str:
    if not raw:
        return fallback
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw.strip())
    cleaned = cleaned.strip("-_")
    return cleaned or fallback


def _as_utc_str(dt: datetime | None) -> str | None:
    """
    Convert a datetime to a UTC ISO 8601 string with a trailing "Z".

    Naive datetimes are treated as already being in UTC and are given a UTC tzinfo.
    Aware datetimes are converted to UTC before formatting. Returns None if dt is None.
    """
    if dt is None:
        return None
    return _to_utc(dt).isoformat(timespec="seconds").replace("+00:00", "Z")


def _flatten_extra_metadata(extra: dict[str, Any] | None, prefix: str = "") -> dict[str, Any]:
    if not extra or not isinstance(extra, dict):
        return {}
    flattened: dict[str, Any] = {}
    for key, value in extra.items():
        key_str = str(key)
        path = f"{prefix}.{key_str}" if prefix else key_str
        if isinstance(value, dict):
            flattened.update(_flatten_extra_metadata(value, prefix=path))
            continue
        flattened[path] = value
    return flattened


def _is_sensitivity_key(key: str) -> bool:
    """Return True when the extra metadata key belongs to sensitivity settings."""
    return key.startswith("sensitivity.")


def _is_snapshot_key(key: str) -> bool:
    """Return True when the extra metadata key refers to snapshot paths."""
    return key.startswith("snapshots.")


def _format_extra_label(key: str) -> str:
    parts = key.replace("_", " ").split(".")
    return " · ".join(part.title() for part in parts)


def _format_extra_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    return str(value)


def _extra_sort_type(values: list[Any]) -> str:
    if not values:
        return "text"
    has_value = False
    for value in values:
        if value is None:
            continue
        has_value = True
        if isinstance(value, bool):
            return "text"
        if not isinstance(value, (int, float)):
            return "text"
    return "number" if has_value else "text"


def _format_sort_value(value: Any, sort_type: str) -> str:
    if value is None:
        return ""
    if sort_type == "number":
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, (int, float)):
            return str(value)
        return ""
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value, ensure_ascii=False).lower()
        except TypeError:
            return str(value).lower()
    return str(value).lower()


def _order_extra_columns(keys: Iterator[str]) -> list[str]:
    preferred = ["detection.box_confidence"]
    key_list = list(keys)
    ordered = [key for key in preferred if key in key_list]
    ordered.extend(sorted(key for key in key_list if key not in preferred))
    return ordered


def _default_extra_column_visibility(columns: Iterable[str]) -> dict[str, bool]:
    """Define default visibility for extra columns by key."""
    return {key: not _is_sensitivity_key(key) for key in columns}


def _prepare_observation_extras(
    observations: list["Observation"],
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    values_by_key: dict[str, list[Any]] = {}
    for o in observations:
        extra_flat = _flatten_extra_metadata(o.get_extra() or {})
        o.extra_flat = extra_flat  # type: ignore[attr-defined]
        for key, value in extra_flat.items():
            if _is_snapshot_key(key):
                continue
            values_by_key.setdefault(key, []).append(value)

    columns = _order_extra_columns(values_by_key.keys())
    sort_types = {key: _extra_sort_type(values_by_key.get(key, [])) for key in columns}
    labels = {key: _format_extra_label(key) for key in columns}

    for o in observations:
        extra_display: dict[str, str] = {}
        extra_sort_values: dict[str, str] = {}
        for key in columns:
            value = o.extra_flat.get(key)  # type: ignore[attr-defined]
            sort_type = sort_types[key]
            extra_display[key] = _format_extra_value(value)
            extra_sort_values[key] = _format_sort_value(value, sort_type)
        o.extra_display = extra_display  # type: ignore[attr-defined]
        o.extra_sort_values = extra_sort_values  # type: ignore[attr-defined]

    return columns, sort_types, labels


def _validate_detection_inputs(raw: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    """
    Validate and coerce detection/ML tuning inputs from the config form.

    Parameters
    ----------
    raw: dict[str, str]
        Form values keyed by the expected field names. Values are parsed as:
        - detect_conf, detect_iou: float in [0.05, 0.95]
        - min_box_area: int in [1, 200000]
        - cooldown_seconds: float in [0.0, 120.0]
        - min_species_prob, match_threshold, ema_alpha: float in [0.0, 1.0]
        - bg_motion_threshold: int in [0, 255]
        - bg_motion_blur: odd int in [1, 99]
        - bg_min_overlap: float in [0.0, 1.0]

    Returns
    -------
    tuple[dict[str, Any], list[str]]
        Parsed numeric values (floats/ints) and a list of validation error
        messages such as "Detection confidence must be a number." or
        "Minimum box area must be between 1 and 200000.".
    """
    parsed: dict[str, Any] = {}
    errors: list[str] = []

    def parse_float(key: str, label: str, lo: float, hi: float) -> None:
        text = str(raw.get(key, "")).strip()
        try:
            val = float(text)
        except ValueError:
            errors.append(f"{label} must be a number.")
            return
        if not (lo <= val <= hi):
            errors.append(f"{label} must be between {lo} and {hi}.")
            return
        parsed[key] = val

    def parse_odd_int(key: str, label: str, lo: int, hi: int) -> None:
        text = str(raw.get(key, "")).strip()
        try:
            val_float = float(text)
        except ValueError:
            errors.append(f"{label} must be a whole number.")
            return
        if not val_float.is_integer():
            errors.append(f"{label} must be a whole number.")
            return
        val = int(val_float)
        if not (lo <= val <= hi):
            errors.append(f"{label} must be between {lo} and {hi}.")
            return
        if val % 2 == 0:
            errors.append(f"{label} must be an odd number.")
            return
        parsed[key] = val

    def parse_int(key: str, label: str, lo: int, hi: int) -> None:
        text = str(raw.get(key, "")).strip()
        try:
            val_float = float(text)
        except ValueError:
            errors.append(f"{label} must be a whole number.")
            return
        if not val_float.is_integer():
            errors.append(f"{label} must be a whole number.")
            return
        val = int(val_float)
        if not (lo <= val <= hi):
            errors.append(f"{label} must be between {lo} and {hi}.")
            return
        parsed[key] = val

    parse_float("detect_conf", "Detection confidence", 0.05, 0.95)
    parse_float("detect_iou", "IOU threshold", 0.05, 0.95)
    parse_int("min_box_area", "Minimum box area", 1, 200000)
    parse_float("cooldown_seconds", "Cooldown seconds", 0.0, 120.0)
    parse_float("min_species_prob", "Minimum species probability", 0.0, 1.0)
    parse_float("match_threshold", "Match threshold", 0.0, 1.0)
    parse_float("ema_alpha", "EMA alpha", 0.0, 1.0)
    parse_int("bg_motion_threshold", "Background motion threshold", 0, 255)
    parse_odd_int("bg_motion_blur", "Background motion blur", 1, 99)
    parse_float("bg_min_overlap", "Background minimum overlap", 0.0, 1.0)

    bg_enabled_raw = str(raw.get("bg_subtraction_enabled", "")).strip().lower()
    parsed["bg_subtraction_enabled"] = bg_enabled_raw in {"1", "true", "yes", "on"}

    tz_text = str(raw.get("timezone", "")).strip()
    if not tz_text:
        parsed["timezone"] = "local"
    else:
        tz_clean = _normalize_timezone(tz_text)
        if tz_clean.lower() == "local":
            parsed["timezone"] = "local"
        else:
            try:
                ZoneInfo(tz_clean)
                parsed["timezone"] = tz_clean
            except ZoneInfoNotFoundError:
                errors.append("Timezone must be a valid IANA name (e.g., America/Los_Angeles) or 'local'.")

    return parsed, errors


def paginate(total_count: int, page: int, page_size: int, max_page_size: int = 100) -> tuple[int, int, int, int]:
    """
    Clamp page/page_size and return (page, page_size, total_pages, offset).
    total_pages is at least 1 even when there are zero rows.
    """
    safe_total = max(0, int(total_count))
    size = max(1, min(int(page_size), max_page_size))
    total_pages = max(1, math.ceil(safe_total / size))
    current = max(1, min(int(page), total_pages))
    offset = (current - 1) * size
    return current, size, total_pages, offset


# ----------------------------
# App factory
# ----------------------------

def make_app() -> Any:
    """
    Create and return a FastAPI application configured with routes and static
    file mounts.  This function will raise ``RuntimeError`` if either
    FastAPI or SQLAlchemy is unavailable.  The return type is ``Any`` to
    avoid import-time type errors when the dependencies are missing.
    """
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is not installed; cannot create the web application."
        )
    if not _SQLA_AVAILABLE:
        raise RuntimeError(
            "SQLAlchemy is not installed; cannot create the web application."
        )

    ensure_dirs()

    @asynccontextmanager
    async def _lifespan(_: FastAPI):
        try:
            await init_async_db()
        except RuntimeError:
            init_db()
        yield

    app = FastAPI(
        title="hbmon",
        description=(
            "LAN-only hummingbird monitoring system. "
            "Uses YOLO for detection, CLIP for species classification, "
            "and embedding-based re-identification for tracking individual birds."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=_lifespan,
    )

    @app.middleware("http")
    async def _disable_cache_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    here = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(here / "templates"))

    # Static assets (CSS/JS)
    app.mount("/static", StaticFiles(directory=str(here / "static")), name="static")

    # Media (snapshots/clips)
    mdir = media_dir()
    mdir.mkdir(parents=True, exist_ok=True)
    app.mount("/media", StaticFiles(directory=str(mdir)), name="media")

    def _safe_unlink_media(rel_path: str | None) -> None:
        if not rel_path:
            return
        p = media_dir() / rel_path
        try:
            p.unlink(missing_ok=True)
        except Exception:
            # Best-effort cleanup; log for visibility but do not block user action.
            print(f"[web] failed to remove media file {p}")

    async def _recompute_individual_stats(
        db: AsyncSession | _AsyncSessionAdapter,
        individual_id: int,
    ) -> None:
        """Update visit_count/last_seen_at for an individual. Caller must commit."""
        ind = await db.get(Individual, individual_id)
        if ind is None:
            print(f"[web] individual {individual_id} not found while recomputing stats")
            return
        rows = (
            await db.execute(
                select(func.count(Observation.id), func.max(Observation.ts))
                .where(Observation.individual_id == individual_id)
            )
        ).one()
        ind.visit_count = int(rows[0] or 0)
        ind.last_seen_at = rows[1]

    async def _commit_with_retry(
        db: AsyncSession | _AsyncSessionAdapter,
        retries: int = 3,
        delay: float = 0.5,
    ) -> None:
        for i in range(max(1, retries)):
            try:
                await db.commit()
                return
            except OperationalError as e:  # pragma: no cover
                msg = str(e).lower()
                if "database is locked" in msg and i < retries - 1:
                    print(f"[web] commit retry due to lock (attempt {i + 1}/{retries})")
                    await anyio.sleep(delay)
                    continue
                raise

    # ----------------------------
    # UI routes
    # ----------------------------

    def _config_form_values(settings, raw: dict[str, str] | None = None) -> dict[str, str]:
        vals = {
            "detect_conf": f"{float(settings.detect_conf):.2f}",
            "detect_iou": f"{float(settings.detect_iou):.2f}",
            "min_box_area": str(int(settings.min_box_area)),
            "cooldown_seconds": f"{float(settings.cooldown_seconds):.2f}",
            "min_species_prob": f"{float(settings.min_species_prob):.2f}",
            "match_threshold": f"{float(settings.match_threshold):.2f}",
            "ema_alpha": f"{float(settings.ema_alpha):.2f}",
            "timezone": str(getattr(settings, "timezone", "local")),
            "bg_subtraction_enabled": "1" if getattr(settings, "bg_subtraction_enabled", True) else "0",
            "bg_motion_threshold": str(int(getattr(settings, "bg_motion_threshold", 30))),
            "bg_motion_blur": str(int(getattr(settings, "bg_motion_blur", 5))),
            "bg_min_overlap": f"{float(getattr(settings, 'bg_min_overlap', 0.15)):.2f}",
        }
        if raw:
            for k, v in raw.items():
                if k in vals:
                    vals[k] = str(v)
        return vals

    def _context(request: Request, title: str, settings=None, **extra: Any) -> dict[str, Any]:
        s_local = settings or load_settings()
        tz_value = _normalize_timezone(getattr(s_local, "timezone", "local"))
        base = {
            "request": request,
            "title": title,
            "timezone": tz_value,
            "timezone_label": _timezone_label(tz_value),
            "app_version": __version__,
            "git_commit": _GIT_COMMIT,
        }
        base.update(extra)
        return base

    @app.get("/", response_class=HTMLResponse)
    async def index(
        request: Request,
        page: int = 1,
        page_size: int = 10,
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> HTMLResponse:
        s = load_settings()
        title = "Hummingbird Monitor"

        cache_key = f"hbmon:index:{page}:{page_size}"
        cached = await cache_get_json(cache_key)
        if cached:
            top_inds_out = cached["top_inds_out"]
            recent = cached["recent"]
            current_page = cached["current_page"]
            clamped_page_size = cached["clamped_page_size"]
            total_pages = cached["total_pages"]
            total_recent = cached["total_recent"]
            last_capture_utc = cached["last_capture_utc"]
        else:
            top_inds = (
                await db.execute(
                    select(Individual.id, Individual.name, Individual.visit_count, Individual.last_seen_at)
                    .order_by(desc(Individual.visit_count))
                    .limit(20)
                )
            ).all()

            # Convert last_seen to ISO for template
            top_inds_out: list[tuple[int, str, int, str | None]] = []
            for iid, name, visits, last_seen in top_inds:
                top_inds_out.append((int(iid), str(name), int(visits), _as_utc_str(last_seen)))

            total_recent = (await db.execute(select(func.count(Observation.id)))).scalar_one()
            current_page, clamped_page_size, total_pages, offset = paginate(
                total_recent, page=page, page_size=page_size, max_page_size=200
            )

            recent_rows = (
                await db.execute(
                    select(Observation)
                    .order_by(desc(Observation.ts))
                    .offset(offset)
                    .limit(clamped_page_size)
                )
            ).scalars().all()

            recent = []
            for o in recent_rows:
                # attach computed presentation attrs (not in DB)
                o.species_css = species_to_css(o.species_label)  # type: ignore[attr-defined]
                # Use annotated snapshot if available, otherwise fall back to raw
                annotated = get_annotated_snapshot_path(o)
                o.display_snapshot_path = annotated if annotated else o.snapshot_path  # type: ignore[attr-defined]
                recent.append(
                    {
                        "id": int(o.id),
                        "species_label": o.species_label,
                        "species_prob": float(o.species_prob),
                        "individual_id": o.individual_id,
                        "match_score": float(o.match_score or 0.0),
                        "ts_utc": _as_utc_str(o.ts),
                        "display_snapshot_path": o.display_snapshot_path,  # type: ignore[attr-defined]
                        "video_path": o.video_path,
                        "species_css": o.species_css,  # type: ignore[attr-defined]
                    }
                )

            latest_data = await _get_latest_observation_data(db)
            last_capture_utc = latest_data["ts_utc"] if latest_data else None

            await cache_set_json(
                cache_key,
                {
                    "top_inds_out": top_inds_out,
                    "recent": recent,
                    "current_page": current_page,
                    "clamped_page_size": clamped_page_size,
                    "total_pages": total_pages,
                    "total_recent": int(total_recent),
                    "last_capture_utc": last_capture_utc,
                },
            )

        roi = s.roi.clamp() if s.roi else None
        roi_str = roi_to_str(roi) if roi else ""
        rtsp = s.rtsp_url or ""
        snapshot_src = "/api/live_frame.jpg" if rtsp else "/api/frame.jpg"

        return templates.TemplateResponse(
            request,
            "index.html",
            _context(
                request,
                title,
                settings=s,
                top_inds=top_inds_out,
                recent=recent,
                recent_page=current_page,
                recent_page_size=clamped_page_size,
                recent_total_pages=total_pages,
                recent_total=int(total_recent),
                recent_page_size_options=[10, 20, 50, 100],
                roi=roi,
                roi_str=roi_str,
                rtsp_url=rtsp,
                last_capture_utc=last_capture_utc,
                snapshot_src=snapshot_src,
                ts=int(time.time()),
            ),
        )

    @app.get("/observations", response_class=HTMLResponse)
    async def observations(
        request: Request,
        individual_id: int | None = None,
        limit: int = 200,
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> HTMLResponse:
        s = load_settings()

        limit = max(10, min(int(limit), 2000))

        q = select(Observation).order_by(desc(Observation.ts)).limit(limit)
        if individual_id is not None:
            q = q.where(Observation.individual_id == individual_id)

        obs = (await db.execute(q)).scalars().all()
        for o in obs:
            o.species_css = species_to_css(o.species_label)  # type: ignore[attr-defined]
            # Use annotated snapshot if available, otherwise fall back to raw
            annotated = get_annotated_snapshot_path(o)
            o.display_snapshot_path = annotated if annotated else o.snapshot_path  # type: ignore[attr-defined]
            o.annotated_snapshot_path = annotated  # type: ignore[attr-defined]
            o.clip_snapshot_path = get_clip_snapshot_path(o)  # type: ignore[attr-defined]
            o.background_snapshot_path = get_background_snapshot_path(o)  # type: ignore[attr-defined]

        extra_columns, extra_sort_types, extra_labels = _prepare_observation_extras(obs)
        extra_column_defaults = _default_extra_column_visibility(extra_columns)

        inds = (
            await db.execute(
                select(Individual).order_by(desc(Individual.visit_count)).limit(2000)
            )
        ).scalars().all()

        total = (await db.execute(select(func.count(Observation.id)))).scalar_one()

        return templates.TemplateResponse(
            request,
            "observations.html",
            _context(
                request,
                "Observations",
                settings=s,
                observations=obs,
                individuals=inds,
                extra_columns=extra_columns,
                extra_column_sort_types=extra_sort_types,
                extra_column_labels=extra_labels,
                extra_column_defaults=extra_column_defaults,
                selected_individual=individual_id,
                selected_limit=limit,
                count_shown=len(obs),
                count_total=int(total),
                rtsp_url=s.rtsp_url,
            ),
        )

    @app.get("/observations/{obs_id}", response_class=HTMLResponse)
    async def observation_detail(
        obs_id: int,
        request: Request,
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> HTMLResponse:
        o = await db.get(Observation, obs_id)
        if o is None:
            raise HTTPException(status_code=404, detail="Observation not found")

        o.species_css = species_to_css(o.species_label)  # type: ignore[attr-defined]
        extra = o.get_extra() or {}
        bbox_xyxy = list(o.bbox_xyxy) if o.bbox_xyxy else None
        original_observation = {
            "species_label": o.species_label,
            "species_prob": o.species_prob,
            "bbox_xyxy": bbox_xyxy,
            "match_score": o.match_score,
            "extra": extra,
        }
        o.original_observation_pretty = pretty_json_obj(original_observation)  # type: ignore[attr-defined]

        # Get annotated snapshot path from extra data (if available)
        annotated_snapshot_path = get_annotated_snapshot_path(o)
        clip_snapshot_path = get_clip_snapshot_path(o)
        background_snapshot_path = get_background_snapshot_path(o)

        # Video file diagnostics
        video_info: dict[str, Any] | None = None
        if o.video_path:
            video_file = media_dir() / o.video_path
            exists = await _run_blocking(video_file.exists)
            size_kb = 0.0
            suffix = ""
            if exists:
                try:
                    size_kb = round((await _run_blocking(video_file.stat)).st_size / 1024, 2)
                except OSError:
                    pass
                suffix = video_file.suffix.lower()
            video_info = {"exists": exists, "size_kb": size_kb, "suffix": suffix}

        return templates.TemplateResponse(
            request,
            "observation_detail.html",
            _context(
                request,
                f"Observation {o.id}",
                o=o,
                extra=extra,
                allowed_review_labels=ALLOWED_REVIEW_LABELS,
                video_info=video_info,
                annotated_snapshot_path=annotated_snapshot_path,
                clip_snapshot_path=clip_snapshot_path,
                background_snapshot_path=background_snapshot_path,
            ),
        )

    @app.post("/observations/{obs_id}/label")
    async def label_observation(
        obs_id: int,
        label: str = Form(...),
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> RedirectResponse:
        """Update the review label for a single observation and redirect to its detail page.

        This endpoint is intended for the observation detail UI. It accepts a form field
        ``label`` for the review label, normalizes it (strip + lower-case), and only
        persists it if it is contained in :data:`ALLOWED_REVIEW_LABELS`. Labels longer
        than 64 characters are rejected with HTTP 400.

        If a valid review label is provided, the observation's ``extra`` JSON is updated
        to include a ``"review"`` section with the label and a ``"labeled_at_utc"``
        timestamp. If the provided label is empty or not allowed, any existing review
        label and timestamp are removed from ``extra`` (and the ``"review"`` section is
        dropped entirely if it becomes empty).

        On success, the change is committed and the client is redirected (HTTP 303) back
        to ``/observations/{obs_id}``.
        """
        o = await db.get(Observation, obs_id)
        if o is None:
            raise HTTPException(status_code=404, detail="Observation not found")

        raw_label = label or ""
        if len(raw_label) > 64:
            raise HTTPException(status_code=400, detail="Label too long")
        clean = raw_label.strip().lower()
        allowed = set(ALLOWED_REVIEW_LABELS)
        review_label = clean if clean in allowed else ""

        if review_label:
            o.merge_extra(
                {
                    "review": {
                        "label": review_label,
                        "labeled_at_utc": _as_utc_str(datetime.now(timezone.utc)),
                    }
                }
            )
        else:
            extra = o.get_extra() or {}
            if isinstance(extra, dict):
                # Work on a copy to avoid mutating the dict returned by get_extra() in place.
                extra_copy = dict(extra)
                raw_review = extra_copy.get("review")
                review = dict(raw_review) if isinstance(raw_review, dict) else {}
                review.pop("label", None)
                review.pop("labeled_at_utc", None)
                if review:
                    extra_copy["review"] = review
                else:
                    # Drop the review section entirely if it's now empty.
                    extra_copy.pop("review", None)
                o.set_extra(extra_copy)
        await _commit_with_retry(db)

        return RedirectResponse(url=f"/observations/{obs_id}", status_code=303)

    @app.post("/observations/{obs_id}/delete")
    async def delete_observation(
        obs_id: int,
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> RedirectResponse:
        o = await db.get(Observation, obs_id)
        if o is None:
            raise HTTPException(status_code=404, detail="Observation not found")

        ind_id = o.individual_id

        # Clean up media
        _safe_unlink_media(o.snapshot_path)
        _safe_unlink_media(o.video_path)
        _safe_unlink_media(get_annotated_snapshot_path(o))
        _safe_unlink_media(get_clip_snapshot_path(o))
        _safe_unlink_media(get_background_snapshot_path(o))

        await db.execute(delete(Embedding).where(Embedding.observation_id == obs_id))
        await db.delete(o)
        await _commit_with_retry(db)

        if ind_id is not None:
            await _recompute_individual_stats(db, int(ind_id))
        await _commit_with_retry(db)

        return RedirectResponse(url="/observations", status_code=303)

    @app.post("/observations/{obs_id}/export_integration_test")
    async def export_observation_integration_test(
        obs_id: int,
        case_name: str = Form(""),
        description: str = Form(""),
        behavior: str = Form(""),
        location: str = Form(""),
        human_verified: str = Form("false"),
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> FileResponse:
        o = await db.get(Observation, obs_id)
        if o is None:
            raise HTTPException(status_code=404, detail="Observation not found")

        extra = o.get_extra() or {}
        bbox_xyxy = list(o.bbox_xyxy) if o.bbox_xyxy else None
        settings = load_settings()

        default_sensitivity = {
            "detect_conf": settings.detect_conf,
            "detect_iou": settings.detect_iou,
            "min_box_area": settings.min_box_area,
            "bg_motion_threshold": settings.bg_motion_threshold,
            "bg_motion_blur": settings.bg_motion_blur,
            "bg_min_overlap": settings.bg_min_overlap,
            "bg_subtraction_enabled": settings.bg_subtraction_enabled,
        }
        default_identification = {
            "individual_id": o.individual_id,
            "match_score": o.match_score,
            "species_label": o.species_label,
            "species_prob": o.species_prob,
            "species_label_final": o.species_label,
            "species_accepted": False,
        }

        extra_copy = extra.copy() if isinstance(extra, dict) else {}
        raw_sensitivity = extra_copy.get("sensitivity")
        raw_identification = extra_copy.get("identification")

        sensitivity: dict[str, Any] = {}
        if isinstance(raw_sensitivity, dict):
            sensitivity.update(raw_sensitivity)
        for key, default_value in default_sensitivity.items():
            if sensitivity.get(key) is None:
                sensitivity[key] = default_value
        extra_copy["sensitivity"] = sensitivity

        identification: dict[str, Any] = {}
        if isinstance(raw_identification, dict):
            identification.update(raw_identification)
        for key, default_value in default_identification.items():
            if identification.get(key) is None:
                identification[key] = default_value
        extra_copy["identification"] = identification

        background_rel = get_background_snapshot_path(o)
        if background_rel:
            snapshots = extra_copy.get("snapshots")
            snapshots_data = dict(snapshots) if isinstance(snapshots, dict) else {}
            snapshots_data["background_path"] = "background.jpg"
            extra_copy["snapshots"] = snapshots_data

        species_label_final = identification.get("species_label_final")
        species_accepted = identification.get("species_accepted")

        original_observation = {
            "species_label": o.species_label,
            "species_prob": o.species_prob,
            "bbox_xyxy": bbox_xyxy,
            "match_score": o.match_score,
            "extra": extra_copy,
        }

        expected = {
            "detection": o.bbox_xyxy is not None,
            "species_label": o.species_label,
            "species_label_final": species_label_final or o.species_label,
            "species_accepted": bool(species_accepted) if species_accepted is not None else False,
            "behavior": behavior or "unknown",
            "human_verified": human_verified.strip().lower() in {"1", "true", "yes", "y", "on"},
        }
        expected_detection = expected["detection"]

        sensitivity_params: dict[str, Any] = {}
        for key in (
            "detect_conf",
            "detect_iou",
            "min_box_area",
            "bg_motion_threshold",
            "bg_motion_blur",
            "bg_min_overlap",
            "bg_subtraction_enabled",
        ):
            sensitivity_params[key] = sensitivity.get(key)

        metadata = {
            "description": description or f"Observation {o.id} integration test",
            "expected": expected,
            "source": {
                "camera": o.camera_name or "",
                "timestamp_utc": o.ts_utc,
                "location": location or "",
            },
            "sensitivity_tests": [
                {
                    "name": "default",
                    "params": sensitivity_params,
                    "expected_detection": expected_detection,
                }
            ],
            "original_observation": original_observation,
        }

        ensure_dirs()
        out_dir = data_dir() / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        safe_case = _sanitize_case_name(case_name, f"observation_{o.id}")
        out_path = out_dir / f"hbmon-integration-{safe_case}-{stamp}.tar.gz"

        metadata_bytes = json.dumps(metadata, indent=4, sort_keys=True).encode("utf-8")
        metadata_info = tarfile.TarInfo(name=f"{safe_case}/metadata.json")
        metadata_info.size = len(metadata_bytes)

        snapshot_path = media_dir() / o.snapshot_path
        video_path = media_dir() / o.video_path
        annotated_rel = get_annotated_snapshot_path(o)
        clip_rel = get_clip_snapshot_path(o)
        annotated_path = (media_dir() / annotated_rel) if annotated_rel else None
        clip_path = (media_dir() / clip_rel) if clip_rel else None
        background_path = (media_dir() / background_rel) if background_rel else None

        def _build_bundle() -> None:
            with tarfile.open(out_path, "w:gz") as tf:
                tf.addfile(metadata_info, io.BytesIO(metadata_bytes))
                if snapshot_path.exists():
                    tf.add(snapshot_path, arcname=f"{safe_case}/snapshot.jpg")
                if video_path.exists():
                    tf.add(video_path, arcname=f"{safe_case}/clip.mp4")
                if annotated_path and annotated_path.exists():
                    tf.add(annotated_path, arcname=f"{safe_case}/snapshot_annotated.jpg")
                if clip_path and clip_path.exists():
                    tf.add(clip_path, arcname=f"{safe_case}/snapshot_clip.jpg")
                if background_path and background_path.exists():
                    tf.add(background_path, arcname=f"{safe_case}/background.jpg")

        await _run_blocking(_build_bundle)

        return FileResponse(str(out_path), filename=out_path.name, media_type="application/gzip")

    @app.post("/observations/bulk_delete")
    async def bulk_delete_observations(
        obs_ids: list[int] = Form([]),
        redirect_to: str | None = Form(None),
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> RedirectResponse:
        redirect_path = _sanitize_redirect_path(redirect_to)
        ids = sorted({int(obs_id) for obs_id in obs_ids if int(obs_id) > 0})
        if not ids:
            return RedirectResponse(url=redirect_path, status_code=303)

        obs_rows = (await db.execute(select(Observation).where(Observation.id.in_(ids)))).scalars().all()
        if not obs_rows:
            return RedirectResponse(url=redirect_path, status_code=303)

        individual_ids = {o.individual_id for o in obs_rows if o.individual_id is not None}

        for o in obs_rows:
            _safe_unlink_media(o.snapshot_path)
            _safe_unlink_media(o.video_path)
            _safe_unlink_media(get_background_snapshot_path(o))

        await db.execute(delete(Embedding).where(Embedding.observation_id.in_(ids)))
        await db.execute(delete(Observation).where(Observation.id.in_(ids)))
        await _commit_with_retry(db)

        for ind_id in individual_ids:
            await _recompute_individual_stats(db, int(ind_id))
        await _commit_with_retry(db)

        return RedirectResponse(url=redirect_path, status_code=303)

    @app.get("/individuals", response_class=HTMLResponse)
    async def individuals(
        request: Request,
        sort: str = "visits",
        limit: int = 200,
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> HTMLResponse:
        limit = max(10, min(int(limit), 5000))
        sort = (sort or "visits").lower()

        q = select(Individual)
        if sort == "id":
            q = q.order_by(Individual.id)
        elif sort == "recent":
            q = q.order_by(desc(Individual.last_seen_at.nulls_last()))
        else:
            q = q.order_by(desc(Individual.visit_count))

        inds = (await db.execute(q.limit(limit))).scalars().all()
        total = (await db.execute(select(func.count(Individual.id)))).scalar_one()
        prototype_map = await select_prototype_observations(db, [int(ind.id) for ind in inds])
        for ind in inds:
            proto_obs = prototype_map.get(int(ind.id))
            if proto_obs is None:
                continue
            annotated = get_annotated_snapshot_path(proto_obs)
            proto_obs.display_snapshot_path = annotated if annotated else proto_obs.snapshot_path  # type: ignore[attr-defined]
            ind.prototype_observation = proto_obs  # type: ignore[attr-defined]

        return templates.TemplateResponse(
            request,
            "individuals.html",
            _context(
                request,
                "Individuals",
                individuals=inds,
                sort=sort,
                limit=limit,
                count_shown=len(inds),
                count_total=int(total),
            ),
        )

    @app.get("/individuals/{individual_id}", response_class=HTMLResponse)
    async def individual_detail(
        individual_id: int,
        request: Request,
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> HTMLResponse:
        ind = await db.get(Individual, individual_id)
        if ind is None:
            raise HTTPException(status_code=404, detail="Individual not found")

        obs = (
            await db.execute(
                select(Observation)
                .where(Observation.individual_id == individual_id)
                .order_by(desc(Observation.ts))
                .limit(500)
            )
        ).scalars().all()

        for o in obs:
            o.species_css = species_to_css(o.species_label)  # type: ignore[attr-defined]
            # Use annotated snapshot if available, otherwise fall back to raw
            annotated = get_annotated_snapshot_path(o)
            o.display_snapshot_path = annotated if annotated else o.snapshot_path  # type: ignore[attr-defined]
            o.annotated_snapshot_path = annotated  # type: ignore[attr-defined]
            o.clip_snapshot_path = get_clip_snapshot_path(o)  # type: ignore[attr-defined]

        extra_columns, extra_sort_types, extra_labels = _prepare_observation_extras(obs)
        extra_column_defaults = _default_extra_column_visibility(extra_columns)

        total = int(ind.visit_count)

        last_seen = _as_utc_str(ind.last_seen_at)

        # SQLite hour-of-day counts in UTC:
        # Observation.ts stored as timezone-aware; SQLite stores as text.
        # Use strftime('%H', ts) which yields 00..23.
        rows = (
            await db.execute(
                select(func.strftime("%H", Observation.ts).label("hh"), func.count(Observation.id))
                .where(Observation.individual_id == individual_id)
                .group_by("hh")
                .order_by("hh")
            )
        ).all()

        hours_rows: list[tuple[int, int]] = [(int(hh), int(cnt)) for (hh, cnt) in rows if hh is not None]
        heatmap = build_hour_heatmap(hours_rows)
        prototype_map = await select_prototype_observations(db, [individual_id])
        proto_obs = prototype_map.get(individual_id)
        if proto_obs is not None:
            annotated = get_annotated_snapshot_path(proto_obs)
            proto_obs.display_snapshot_path = annotated if annotated else proto_obs.snapshot_path  # type: ignore[attr-defined]

        return templates.TemplateResponse(
            request,
            "individual_detail.html",
            _context(
                request,
                f"Individual {ind.id}",
                individual=ind,
                observations=obs,
                extra_columns=extra_columns,
                extra_column_sort_types=extra_sort_types,
                extra_column_labels=extra_labels,
                extra_column_defaults=extra_column_defaults,
                heatmap=heatmap,
                total=total,
                last_seen=last_seen,
                prototype_observation=proto_obs,
            ),
        )

    @app.post("/individuals/{individual_id}/rename")
    async def rename_individual(
        individual_id: int,
        name: str = Form(...),
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> RedirectResponse:
        ind = await db.get(Individual, individual_id)
        if ind is None:
            raise HTTPException(status_code=404, detail="Individual not found")

        new_name = (name or "").strip()
        if not new_name:
            new_name = "(unnamed)"
        ind.name = new_name[:128]
        await db.commit()

        return RedirectResponse(url=f"/individuals/{individual_id}", status_code=303)

    @app.post("/individuals/{individual_id}/delete")
    async def delete_individual(
        individual_id: int,
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> RedirectResponse:
        ind = await db.get(Individual, individual_id)
        if ind is None:
            raise HTTPException(status_code=404, detail="Individual not found")

        obs_rows = (
            await db.execute(
                select(Observation.id, Observation.snapshot_path, Observation.video_path)
                .where(Observation.individual_id == individual_id)
            )
        ).all()
        obs_ids = [int(r[0]) for r in obs_rows]
        for _, snap, vid in obs_rows:
            _safe_unlink_media(snap)
            _safe_unlink_media(vid)

        if obs_ids:
            await db.execute(
                delete(Embedding).where(Embedding.observation_id.in_(obs_ids))
            )
            await db.execute(delete(Observation).where(Observation.id.in_(obs_ids)))

        await db.delete(ind)
        await _commit_with_retry(db, retries=5, delay=0.6)

        return RedirectResponse(url="/individuals", status_code=303)

    @app.post("/individuals/{individual_id}/refresh_embedding")
    async def refresh_embedding(
        individual_id: int,
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> RedirectResponse:
        ind = await db.get(Individual, individual_id)
        if ind is None:
            raise HTTPException(status_code=404, detail="Individual not found")

        embs = (
            await db.execute(
                select(Embedding)
                .where(Embedding.individual_id == individual_id)
                .order_by(desc(Embedding.created_at))
                .limit(500)
            )
        ).scalars().all()

        if not embs:
            return RedirectResponse(url=f"/individuals/{individual_id}", status_code=303)

        vecs = [e.get_vec() for e in embs]
        proto = l2_normalize(sum(vecs) / max(1, len(vecs)))
        ind.set_prototype(proto)
        await db.commit()

        return RedirectResponse(url=f"/individuals/{individual_id}", status_code=303)

    @app.get("/individuals/{individual_id}/split_review", response_class=HTMLResponse)
    async def split_review(
        individual_id: int,
        request: Request,
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> HTMLResponse:
        ind = await db.get(Individual, individual_id)
        if ind is None:
            raise HTTPException(status_code=404, detail="Individual not found")

        obs = (
            await db.execute(
                select(Observation)
                .where(Observation.individual_id == individual_id)
                .order_by(desc(Observation.ts))
                .limit(120)
            )
        ).scalars().all()

        for o in obs:
            o.species_css = species_to_css(o.species_label)  # type: ignore[attr-defined]
            o.suggested_side = "A"  # type: ignore[attr-defined]
            # Use annotated snapshot if available, otherwise fall back to raw
            annotated = get_annotated_snapshot_path(o)
            o.display_snapshot_path = annotated if annotated else o.snapshot_path  # type: ignore[attr-defined]

        emb_rows = (
            await db.execute(
                select(Embedding).join(Observation, Embedding.observation_id == Observation.id)
                .where(Observation.individual_id == individual_id)
                .order_by(desc(Observation.ts))
                .limit(120)
            )
        ).scalars().all()

        # Map obs_id -> embedding vec
        emb_map: dict[int, Any] = {}
        for e in emb_rows:
            emb_map[int(e.observation_id)] = e.get_vec()

        # Build aligned list
        aligned_vecs = []
        aligned_obs_ids = []
        for o in obs:
            v = emb_map.get(int(o.id))
            if v is not None:
                aligned_vecs.append(v)
                aligned_obs_ids.append(int(o.id))

        suggestion = suggest_split_two_groups(aligned_vecs, min_samples=12)

        if suggestion.ok and suggestion.labels and suggestion.centroid_a is not None and suggestion.centroid_b is not None:
            side_map = {oid: lab for oid, lab in zip(aligned_obs_ids, suggestion.labels)}
            for o in obs:
                if int(o.id) in side_map:
                    o.suggested_side = side_map[int(o.id)]  # type: ignore[attr-defined]
        else:
            # Keep A default
            pass

        return templates.TemplateResponse(
            request,
            "split_review.html",
            _context(
                request,
                f"Split review {ind.id}",
                individual=ind,
                observations=obs,
                suggestion_reason=suggestion.reason,
            ),
        )

    @app.post("/individuals/{individual_id}/split_apply")
    async def split_apply(
        individual_id: int,
        request: Request,
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> RedirectResponse:
        ind_a = await db.get(Individual, individual_id)
        if ind_a is None:
            raise HTTPException(status_code=404, detail="Individual not found")

        form = await request.form()
        # Parse assignments: assign_<obsid> => "A" or "B"
        assign: dict[int, str] = {}
        for k, v in form.items():
            if not k.startswith("assign_"):
                continue
            try:
                oid = int(k.split("_", 1)[1])
            except Exception:
                continue
            side = str(v).strip().upper()
            assign[oid] = "B" if side == "B" else "A"

        # Determine which observations move to B
        obs_to_b = [oid for (oid, side) in assign.items() if side == "B"]
        if not obs_to_b:
            # no-op
            return RedirectResponse(url=f"/individuals/{individual_id}", status_code=303)

        # Create individual B
        ind_b = Individual(name=f"(split from {ind_a.id})", visit_count=0, last_seen_at=None)
        db.add(ind_b)
        await db.flush()  # get ind_b.id

        # Reassign observations to B
        moved_obs = (
            await db.execute(select(Observation).where(Observation.id.in_(obs_to_b)))
        ).scalars().all()
        for o in moved_obs:
            o.individual_id = ind_b.id

        # Reassign embeddings rows to B too
        moved_embs = (
            await db.execute(select(Embedding).where(Embedding.observation_id.in_(obs_to_b)))
        ).scalars().all()
        for e in moved_embs:
            e.individual_id = ind_b.id

        await db.commit()

        # Recompute visit counts + last seen
        async def recompute_stats(ind: Individual) -> None:
            rows = (
                await db.execute(
                    select(func.count(Observation.id), func.max(Observation.ts))
                    .where(Observation.individual_id == ind.id)
                )
            ).one()
            ind.visit_count = int(rows[0] or 0)
            ind.last_seen_at = rows[1]

        await recompute_stats(ind_a)
        await recompute_stats(ind_b)

        # Recompute prototypes from embeddings (if available)
        async def recompute_proto(ind: Individual) -> None:
            embs = (
                await db.execute(
                    select(Embedding).where(Embedding.individual_id == ind.id).limit(2000)
                )
            ).scalars().all()
            if not embs:
                return
            vecs = [e.get_vec() for e in embs]
            proto = l2_normalize(sum(vecs) / max(1, len(vecs)))
            ind.set_prototype(proto)

        await recompute_proto(ind_a)
        await recompute_proto(ind_b)

        await db.commit()

        return RedirectResponse(url=f"/individuals/{ind_b.id}", status_code=303)

    @app.get("/config", response_class=HTMLResponse)
    async def config_page(request: Request) -> HTMLResponse:
        s = load_settings()
        saved = request.query_params.get("saved") == "1"
        return templates.TemplateResponse(
            request,
            "config.html",
            _context(
                request,
                "Config",
                settings=s,
                form_values=_config_form_values(s),
                errors=[],
                saved=saved,
            ),
        )

    @app.post("/config", response_class=HTMLResponse)
    async def config_save(request: Request) -> HTMLResponse:
        s = load_settings()
        form = await request.form()
        field_names = (
            "detect_conf",
            "detect_iou",
            "min_box_area",
            "cooldown_seconds",
            "min_species_prob",
            "match_threshold",
            "ema_alpha",
            "bg_motion_threshold",
            "bg_motion_blur",
            "bg_min_overlap",
        )
        raw = {name: str(form.get(name, "") or "").strip() for name in field_names}
        raw["timezone"] = str(form.get("timezone", "") or "").strip()
        raw["bg_subtraction_enabled"] = "1" if form.get("bg_subtraction_enabled") else "0"
        parsed, errors = _validate_detection_inputs(raw)

        if errors:
            return templates.TemplateResponse(
                request,
                "config.html",
                _context(
                    request,
                    "Config",
                    settings=s,
                    form_values=_config_form_values(s, raw),
                    errors=errors,
                    saved=False,
                ),
                status_code=400,
            )

        s.detect_conf = parsed["detect_conf"]
        s.detect_iou = parsed["detect_iou"]
        s.min_box_area = parsed["min_box_area"]
        s.cooldown_seconds = parsed["cooldown_seconds"]
        s.min_species_prob = parsed["min_species_prob"]
        s.match_threshold = parsed["match_threshold"]
        s.ema_alpha = parsed["ema_alpha"]
        s.timezone = parsed["timezone"]
        s.bg_subtraction_enabled = parsed["bg_subtraction_enabled"]
        s.bg_motion_threshold = parsed["bg_motion_threshold"]
        s.bg_motion_blur = parsed["bg_motion_blur"]
        s.bg_min_overlap = parsed["bg_min_overlap"]
        save_settings(s)

        return RedirectResponse(url="/config?saved=1", status_code=303)

    @app.get("/calibrate", response_class=HTMLResponse)
    async def calibrate(request: Request) -> HTMLResponse:
        s = load_settings()
        roi_str = roi_to_str(s.roi) if s.roi else ""
        return templates.TemplateResponse(
            request,
            "calibrate.html",
            _context(
                request,
                "Calibrate ROI",
                settings=s,
                roi=s.roi,
                roi_str=roi_str,
                live_frame_url="/api/live_frame.jpg",
                fallback_frame_url="/api/frame.jpg",
                ts=int(time.time()),
            ),
        )

    @app.get("/background", response_class=HTMLResponse)
    async def background_page(
        request: Request,
        page: int = 1,
        page_size: int = 20,
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> HTMLResponse:
        """
        Background image configuration page.

        Allows users to:
        - View the current background image
        - Select an observation snapshot as background
        - Upload a custom background image
        - Clear the background image
        """
        s = load_settings()
        bg_path = background_image_path()
        bg_exists = await _run_blocking(bg_path.exists)

        # Get recent observations for selection
        total_obs = (await db.execute(select(func.count(Observation.id)))).scalar_one()
        current_page, clamped_page_size, total_pages, offset = paginate(
            total_obs, page=page, page_size=page_size, max_page_size=100
        )

        recent_obs = (
            await db.execute(
                select(Observation)
                .order_by(desc(Observation.ts))
                .offset(offset)
                .limit(clamped_page_size)
            )
        ).scalars().all()

        for o in recent_obs:
            o.species_css = species_to_css(o.species_label)  # type: ignore[attr-defined]

        return templates.TemplateResponse(
            request,
            "background.html",
            _context(
                request,
                "Background Image",
                settings=s,
                background_configured=bool(s.background_image),
                background_exists=bg_exists,
                live_frame_url="/api/live_frame.jpg",
                fallback_frame_url="/api/frame.jpg",
                rtsp_configured=bool(s.rtsp_url),
                observations=recent_obs,
                obs_page=current_page,
                obs_page_size=clamped_page_size,
                obs_total_pages=total_pages,
                obs_total=int(total_obs),
                ts=int(time.time()),
            ),
        )

    # ----------------------------
    # API routes
    # ----------------------------

    @app.get("/api/health", response_model=HealthOut)
    async def health(db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep)) -> HealthOut:
        s = load_settings()
        db_ok = True
        try:
            latest_data = await _get_latest_observation_data(db)
        except Exception:
            db_ok = False
            latest_data = None

        return HealthOut(
            ok=True,
            version="0.1.0",
            db_ok=db_ok,
            last_observation_utc=(latest_data["ts_utc"] if latest_data else None),
            rtsp_url=(s.rtsp_url or None),
        )

    @app.get("/api/roi", response_model=RoiOut)
    async def get_roi() -> RoiOut:
        s = load_settings()
        if s.roi is None:
            return RoiOut(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
        r = s.roi.clamp()
        return RoiOut(x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2)

    @app.post("/api/roi")
    async def set_roi(
        x1: float = Form(...),
        y1: float = Form(...),
        x2: float = Form(...),
        y2: float = Form(...),
    ) -> RedirectResponse:
        s = load_settings()
        r = Roi(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)).clamp()
        s.roi = r
        save_settings(s)
        return RedirectResponse(url="/calibrate", status_code=303)

    @app.get("/api/background")
    async def get_background() -> dict[str, Any]:
        """
        Return information about the current background image.

        The response includes:
        - configured: whether a background image is configured
        - path: relative path to the background image (if configured)
        - exists: whether the background image file exists on disk
        """
        s = load_settings()
        bg_path = background_image_path()
        exists = await _run_blocking(bg_path.exists)
        return {
            "configured": bool(s.background_image),
            "path": s.background_image or None,
            "exists": exists,
        }

    def _capture_live_frame(rtsp_url: str) -> tuple[Any, Any]:
        cv2 = _load_cv2()

        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            raise HTTPException(status_code=503, detail="Unable to open RTSP stream")

        try:
            ok, frame = cap.read()
        finally:
            cap.release()

        if not ok or frame is None:
            raise HTTPException(status_code=503, detail="Failed to read RTSP frame")

        return frame, cv2

    def _encode_jpeg(frame: Any, cv2: Any, quality: int = 80) -> bytes:
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode frame")
        return jpeg.tobytes()

    @app.get("/api/background.jpg")
    async def get_background_jpg() -> Any:
        """
        Return the configured background image, or a placeholder if none set.
        """
        bg_path = background_image_path()
        if await _run_blocking(bg_path.exists):
            return FileResponse(str(bg_path), media_type="image/jpeg")

        # Return placeholder
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            raise HTTPException(status_code=404, detail="PIL library not available for image generation")

        def _build_placeholder() -> io.BytesIO:
            img = Image.new("RGB", (960, 540), (28, 28, 38))
            d = ImageDraw.Draw(img)
            d.text(
                (24, 24),
                "No background image configured.\nSelect an observation snapshot or upload an image.",
                fill=(180, 180, 200),
            )
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            return buf

        buf = await _run_blocking(_build_placeholder)
        return StreamingResponse(buf, media_type="image/jpeg")

    @app.post("/api/background/from_observation/{obs_id}")
    async def set_background_from_observation(
        obs_id: int,
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> RedirectResponse:
        """
        Set the background image by copying a snapshot from an existing observation.

        The snapshot is copied to a static location (/data/background/background.jpg)
        so that it persists even if the original observation is deleted.
        """
        o = await db.get(Observation, obs_id)
        if o is None:
            raise HTTPException(status_code=404, detail="Observation not found")

        if not o.snapshot_path:
            raise HTTPException(status_code=400, detail="Observation has no snapshot")

        src_path = media_dir() / o.snapshot_path
        if not await _run_blocking(src_path.exists):
            raise HTTPException(status_code=404, detail="Snapshot file not found")

        # Ensure background directory exists
        bg_dir = background_dir()
        await _run_blocking(bg_dir.mkdir, parents=True, exist_ok=True)

        # Copy to static location
        dst_path = background_image_path()
        await _run_blocking(shutil.copy2, src_path, dst_path)

        # Update settings
        s = load_settings()
        s.background_image = "background/background.jpg"
        save_settings(s)

        return RedirectResponse(url="/background", status_code=303)

    @app.post("/api/background/from_live")
    async def set_background_from_live() -> RedirectResponse:
        """
        Capture a live RTSP frame and save it as the background image.
        """
        s = load_settings()
        if not s.rtsp_url:
            raise HTTPException(status_code=503, detail="RTSP URL not configured")

        frame, cv2 = await _run_blocking(_capture_live_frame, s.rtsp_url)
        jpeg_bytes = await _run_blocking(_encode_jpeg, frame, cv2, 85)

        # Ensure background directory exists
        bg_dir = background_dir()
        await _run_blocking(bg_dir.mkdir, parents=True, exist_ok=True)
        dst_path = background_image_path()
        await _run_blocking(dst_path.write_bytes, jpeg_bytes)

        s.background_image = "background/background.jpg"
        save_settings(s)

        return RedirectResponse(url="/background", status_code=303)

    @app.post("/api/background/upload")
    async def upload_background(request: Request) -> RedirectResponse:
        """
        Upload a custom background image.

        The uploaded file is saved to /data/background/background.jpg.
        Accepts JPEG and PNG images; PNG is converted to JPEG.
        """
        form = await request.form()
        upload = form.get("file")
        if upload is None:
            raise HTTPException(status_code=400, detail="No file uploaded")

        # Validate that upload has a read method (is file-like)
        if not hasattr(upload, "read"):
            raise HTTPException(status_code=400, detail="Invalid file upload")

        # Read file content
        try:
            content = await upload.read()
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to read uploaded file")

        if not content or len(content) < 100:
            raise HTTPException(status_code=400, detail="File is empty or too small")

        # Limit file size to 10MB
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")

        # Validate and convert image
        try:
            from PIL import Image
        except ImportError:
            raise HTTPException(status_code=500, detail="PIL not available for image processing")

        try:
            img = Image.open(io.BytesIO(content))
            # Convert to RGB (handles PNG with alpha, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Ensure background directory exists
        bg_dir = background_dir()
        await _run_blocking(bg_dir.mkdir, parents=True, exist_ok=True)

        # Save as JPEG
        dst_path = background_image_path()
        await _run_blocking(img.save, dst_path, format="JPEG", quality=90)

        # Update settings
        s = load_settings()
        s.background_image = "background/background.jpg"
        save_settings(s)

        return RedirectResponse(url="/background", status_code=303)

    @app.post("/api/background/clear")
    async def clear_background() -> RedirectResponse:
        """
        Clear the configured background image.

        Removes the background image file and clears the setting.
        """
        # Remove file if it exists
        bg_path = background_image_path()
        try:
            await _run_blocking(bg_path.unlink, missing_ok=True)
        except Exception:
            # Ignore errors during file removal (e.g., permission issues, race conditions).
            # The setting will still be cleared below, and the file can be cleaned up later.
            pass

        # Clear setting
        s = load_settings()
        s.background_image = ""
        save_settings(s)

        return RedirectResponse(url="/background", status_code=303)

    @app.get("/api/frame.jpg")
    async def frame_jpg(
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> Any:
        """
        Return latest snapshot image, or a placeholder if none exist yet.
        """
        latest_data = await _get_latest_observation_data(db)
        if latest_data is not None:
            p = media_dir() / latest_data["snapshot_path"]
            if await _run_blocking(p.exists):
                return FileResponse(str(p), media_type="image/jpeg")

        # Placeholder
        try:
            from PIL import Image, ImageDraw
        except Exception:
            raise HTTPException(status_code=404, detail="No frames yet")

        def _build_placeholder() -> io.BytesIO:
            img = Image.new("RGB", (960, 540), (18, 18, 28))
            d = ImageDraw.Draw(img)
            d.text((24, 24), "No snapshots yet.\nStart the worker and wait for a visit.", fill=(220, 220, 235))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            return buf

        buf = await _run_blocking(_build_placeholder)
        return StreamingResponse(buf, media_type="image/jpeg")

    @app.get("/api/live_frame.jpg")
    async def live_frame_jpg() -> Any:
        """
        Return a single snapshot from the live RTSP stream.
        """
        s = load_settings()
        if not s.rtsp_url:
            raise HTTPException(status_code=503, detail="RTSP URL not configured")

        frame, cv2 = await _run_blocking(_capture_live_frame, s.rtsp_url)
        jpeg_bytes = await _run_blocking(_encode_jpeg, frame, cv2, 80)

        return StreamingResponse(io.BytesIO(jpeg_bytes), media_type="image/jpeg")

    @app.get("/api/stream.mjpeg")
    async def stream_mjpeg() -> Any:
        """
        Stream live MJPEG video from the RTSP camera.

        Returns a multipart/x-mixed-replace stream of JPEG frames.
        This provides a true live video feed from the camera.

        If the RTSP URL is not configured or the camera is unavailable,
        returns an error response.
        """
        s = load_settings()
        if not s.rtsp_url:
            raise HTTPException(status_code=503, detail="RTSP URL not configured")

        _load_cv2()
        mjpeg_settings = _load_mjpeg_settings()
        broadcaster = _get_frame_broadcaster(s.rtsp_url, mjpeg_settings)
        broadcaster.add_client()
        frame_interval = 1.0 / mjpeg_settings.target_fps

        async def _wait_for_frame(timeout: float = 2.0) -> None:
            start = time.monotonic()
            while time.monotonic() - start < timeout:
                frame_bytes, _ = broadcaster.latest_frame()
                if frame_bytes:
                    return
                await anyio.sleep(0.01)

        await _wait_for_frame()

        from starlette.concurrency import iterate_in_threadpool

        initial_frame, initial_timestamp = broadcaster.latest_frame()
        initial_chunk = None
        last_seen_timestamp = 0.0
        last_update = time.monotonic()
        if initial_frame:
            initial_chunk = (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(initial_frame)).encode() + b"\r\n"
                b"\r\n" + initial_frame + b"\r\n"
            )
            last_seen_timestamp = initial_timestamp
            last_update = time.monotonic()

        def _generate_frames() -> Iterator[bytes]:
            nonlocal last_seen_timestamp, last_update
            if initial_chunk is not None:
                yield initial_chunk
            while True:
                frame_bytes, frame_timestamp = broadcaster.latest_frame()
                if frame_bytes:
                    if frame_timestamp and frame_timestamp != last_seen_timestamp:
                        last_seen_timestamp = frame_timestamp
                        last_update = time.monotonic()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n"
                        b"\r\n" + frame_bytes + b"\r\n"
                    )
                if time.monotonic() - last_update > 2.0:
                    break
                time.sleep(frame_interval)

        async def generate_frames() -> AsyncIterator[bytes]:
            iterator = _generate_frames()
            try:
                async for chunk in iterate_in_threadpool(iterator):
                    yield chunk
            finally:
                await anyio.to_thread.run_sync(iterator.close)
                broadcaster.remove_client()

        return StreamingResponse(
            generate_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/stream_diagnostics")
    async def stream_diagnostics() -> dict[str, Any]:
        """
        Return diagnostic information about the live MJPEG stream.

        Includes source stream properties, MJPEG encoding settings, and recent frame stats.
        """
        s = load_settings()
        if not s.rtsp_url:
            raise HTTPException(status_code=503, detail="RTSP URL not configured")

        mjpeg_settings = _load_mjpeg_settings()
        broadcaster = _get_frame_broadcaster(s.rtsp_url, mjpeg_settings)
        data = broadcaster.diagnostics()

        frame_data = data.get("frame", {})
        mjpeg_data = data.get("mjpeg", {})
        last_frame_size = frame_data.get("last_frame_size_bytes")
        current_fps = mjpeg_data.get("current_fps")
        if last_frame_size and current_fps and current_fps > 0:
            frame_data["estimated_bitrate_kbps"] = round((last_frame_size * 8 * current_fps) / 1000.0, 2)
        else:
            frame_data["estimated_bitrate_kbps"] = None
        data["frame"] = frame_data
        return data

    @app.get("/api/video_info/{obs_id}")
    async def video_info(
        obs_id: int,
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> dict[str, Any]:
        """
        Return diagnostic information about a video file for troubleshooting.

        This endpoint helps diagnose video streaming issues by providing:
        - file_exists: whether the video file exists on disk
        - file_size_bytes: size of the video file
        - file_suffix: file extension (e.g., .mp4, .avi)
        - video_path: relative path stored in the database
        - absolute_path: full path on the server filesystem
        - codec_hint: detected codec from file header (if available)
        - browser_compatible: whether the codec is likely supported by browsers
        """
        o = await db.get(Observation, obs_id)
        if o is None:
            raise HTTPException(status_code=404, detail="Observation not found")

        video_path = o.video_path or ""
        if not video_path:
            return {
                "observation_id": obs_id,
                "video_path": "",
                "file_exists": False,
                "error": "No video path stored for this observation",
            }

        full_path = media_dir() / video_path
        exists = await _run_blocking(full_path.exists)

        result: dict[str, Any] = {
            "observation_id": obs_id,
            "video_path": video_path,
            "absolute_path": str(full_path),
            "file_exists": exists,
        }

        if exists:
            try:
                stat = await _run_blocking(full_path.stat)
                result["file_size_bytes"] = stat.st_size
                result["file_size_kb"] = round(stat.st_size / 1024, 2)
            except OSError as e:
                result["stat_error"] = str(e)
            result["file_suffix"] = full_path.suffix.lower()

            # Try to detect codec from file header
            codec_hint = "unknown"
            browser_compatible = False
            try:
                with open(full_path, "rb") as f:
                    header = f.read(32)
                    # Check for common MP4/MOV signatures
                    if b"ftyp" in header:
                        # Extract the brand after ftyp
                        ftyp_pos = header.find(b"ftyp")
                        if ftyp_pos >= 0 and ftyp_pos + 8 <= len(header):
                            brand = header[ftyp_pos + 4 : ftyp_pos + 8].decode("ascii", errors="ignore")
                            codec_hint = f"MP4 (brand: {brand})"
                            # isom, mp41, mp42, avc1 are generally compatible
                            if brand in ("isom", "mp41", "mp42", "avc1", "M4V "):
                                browser_compatible = True
                            else:
                                # mp4v (MPEG-4 Part 2) is NOT well-supported
                                browser_compatible = False
                    elif header.startswith(b"RIFF") and b"AVI " in header:
                        codec_hint = "AVI container"
                        browser_compatible = False
                    elif header.startswith(b"\x1a\x45\xdf\xa3"):
                        codec_hint = "WebM/Matroska"
                        browser_compatible = True
            except Exception as e:
                result["codec_detection_error"] = str(e)

            result["codec_hint"] = codec_hint
            result["browser_compatible"] = browser_compatible
            if not browser_compatible:
                result["playback_warning"] = (
                    "This video may not play in browsers. OpenCV often uses mp4v (MPEG-4 Part 2) "
                    "which is not supported by most browsers. Consider using VLC or converting "
                    "with FFmpeg: ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4"
                )
        else:
            result["error"] = f"Video file not found at {full_path}"

        return result

    # ----------------------------
    # Export routes
    # ----------------------------

    def _stream_csv(rows: Iterator[list[Any]], header: list[str]) -> StreamingResponse:
        def gen() -> Iterator[bytes]:
            sio = io.StringIO()
            w = csv.writer(sio)
            w.writerow(header)
            yield sio.getvalue().encode("utf-8")
            sio.seek(0)
            sio.truncate(0)

            for r in rows:
                w.writerow(r)
                yield sio.getvalue().encode("utf-8")
                sio.seek(0)
                sio.truncate(0)

        return StreamingResponse(gen(), media_type="text/csv")

    @app.get("/export/observations.csv")
    async def export_observations_csv(
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> StreamingResponse:
        q = (await db.execute(select(Observation).order_by(desc(Observation.ts)))).scalars().all()

        def rows() -> Iterator[list[Any]]:
            for o in q:
                yield [
                    o.id,
                    o.ts_utc,
                    o.camera_name or "",
                    o.species_label,
                    f"{o.species_prob:.6f}",
                    o.individual_id if o.individual_id is not None else "",
                    f"{o.match_score:.6f}",
                    o.snapshot_path,
                    o.video_path,
                    o.bbox_str or "",
                ]

        return _stream_csv(
            rows(),
            header=[
                "observation_id",
                "ts_utc",
                "camera_name",
                "species_label",
                "species_prob",
                "individual_id",
                "match_score",
                "snapshot_path",
                "video_path",
                "bbox_xyxy",
            ],
        )

    @app.get("/export/individuals.csv")
    async def export_individuals_csv(
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> StreamingResponse:
        q = (await db.execute(select(Individual).order_by(desc(Individual.visit_count)))).scalars().all()

        def rows() -> Iterator[list[Any]]:
            for i in q:
                yield [
                    i.id,
                    i.name,
                    i.visit_count,
                    _as_utc_str(i.created_at) or "",
                    _as_utc_str(i.last_seen_at) or "",
                    i.last_species_label or "",
                ]

        return _stream_csv(
            rows(),
            header=["individual_id", "name", "visit_count", "created_utc", "last_seen_utc", "last_species_label"],
        )

    @app.get("/export/media_bundle.tar.gz")
    async def export_media_bundle() -> FileResponse:
        """
        Create a tar.gz containing /media/snapshots and /media/clips.
        (Created on-demand under /data/exports.)
        """
        from hbmon.config import data_dir, snapshots_dir, clips_dir

        ensure_dirs()
        out_dir = data_dir() / "exports"
        await _run_blocking(out_dir.mkdir, parents=True, exist_ok=True)

        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        out_path = out_dir / f"hbmon-media-{stamp}.tar.gz"

        snap = snapshots_dir()
        clips = clips_dir()

        def _build_bundle() -> None:
            with tarfile.open(out_path, "w:gz") as tf:
                if snap.exists():
                    tf.add(snap, arcname="snapshots")
                if clips.exists():
                    tf.add(clips, arcname="clips")

        await _run_blocking(_build_bundle)

        return FileResponse(str(out_path), filename=out_path.name, media_type="application/gzip")

    return app


# Default ASGI app for uvicorn - lazy initialization to avoid
# running make_app() at import time (which causes issues in tests)
_app_instance: Any = None


def get_app() -> Any:
    """Get or create the FastAPI app instance (lazy singleton)."""
    global _app_instance
    if _app_instance is None:
        _app_instance = make_app()
    return _app_instance


# For uvicorn: create app lazily on first access
def __getattr__(name: str) -> Any:
    """Module-level __getattr__ for lazy app initialization."""
    if name == "app":
        return get_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
