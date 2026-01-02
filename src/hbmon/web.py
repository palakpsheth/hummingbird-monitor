# src/hbmon/web.py
"""
FastAPI + Jinja2 web UI for hbmon (LAN-only, no auth).

Routes:
- /                      Dashboard
- /observations           Gallery + filters
- /observations/{id}      Observation detail (with inline video player)
- /candidates             Motion-rejected detections for review
- /candidates/{id}        Candidate detail + labeling
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
from datetime import date, datetime, timedelta, timezone
from functools import partial
import importlib.util
import io
import json
from json import JSONDecodeError
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen, Request as UrllibRequest
from urllib.error import URLError, HTTPError
import time
import weakref
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterable, Iterator, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import anyio
import numpy as np

logger = logging.getLogger(__name__)

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
    from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse, Response  # type: ignore
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
    from sqlalchemy import JSON, Integer, cast, delete, desc, func, select  # type: ignore
    from sqlalchemy.dialects.postgresql import JSONB  # type: ignore
    from sqlalchemy.exc import OperationalError  # type: ignore
    from sqlalchemy.orm import Session  # type: ignore
    from sqlalchemy.ext.asyncio import AsyncSession  # type: ignore
    _SQLA_AVAILABLE = True
except Exception:  # pragma: no cover
    delete = desc = func = select = JSON = JSONB = None  # type: ignore
    Session = object  # type: ignore
    AsyncSession = object  # type: ignore
    OperationalError = Exception  # type: ignore
    _SQLA_AVAILABLE = False

ALLOWED_REVIEW_LABELS = ["true_positive", "false_positive", "unknown"]
CANDIDATE_REVIEW_LABELS = ["false_negative", "true_negative", "unknown"]


from hbmon import __version__
from hbmon.config import (
    Roi,
    background_dir,
    background_image_path,
    data_dir,
    ensure_dirs,
    load_settings,
    media_dir,
    roi_to_str,
    save_settings,
)
from hbmon.cache import cache_get_json, cache_set_json
from hbmon.db import (
    dispose_async_engine,
    get_async_db,
    get_session_factory,
    init_async_db,
    init_db,
    is_async_db_available,
)
from hbmon.models import Candidate, Embedding, Individual, Observation, _to_utc
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


def _get_db_dialect_name(db: AsyncSession | _AsyncSessionAdapter) -> str:
    bind = None
    if hasattr(db, "get_bind"):
        try:
            bind = db.get_bind()
        except Exception:  # pragma: no cover - defensive fallback
            bind = None
    if bind is None:
        bind = getattr(db, "bind", None)
    if bind is None:
        return ""
    if hasattr(bind, "dialect"):
        return str(getattr(bind.dialect, "name", ""))
    if hasattr(bind, "sync_engine") and hasattr(bind.sync_engine, "dialect"):
        return str(getattr(bind.sync_engine.dialect, "name", ""))
    return ""


def _candidate_json_value(expr: Any, path: Sequence[str], dialect: str) -> Any | None:
    if not path:
        return None
    if dialect == "sqlite":
        return func.json_extract(expr, f"$.{'.'.join(path)}")
    if dialect in {"postgres", "postgresql"}:
        if JSONB is not None:
            return func.jsonb_extract_path_text(cast(expr, JSONB), *path)
        return func.json_extract_path_text(cast(expr, JSON), *path)
    return None











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


def _safe_internal_url(path: str, resource_id: int | None = None) -> str:
    """
    Construct a safe internal URL from a base path and optional resource ID.
    
    This function ensures URLs are constructed safely to prevent open redirect
    vulnerabilities flagged by CodeQL. Path must start with "/" and resource_id
    is validated as an integer.
    
    Args:
        path: Base path (e.g., "/observations", "/individuals")
        resource_id: Optional integer ID to append to path
        
    Returns:
        Safe internal URL path
        
    Examples:
        _safe_internal_url("/observations", 123) -> "/observations/123"
        _safe_internal_url("/individuals") -> "/individuals"
    """
    if not path.startswith("/"):
        raise ValueError(f"Path must start with '/': {path}")
    
    if resource_id is not None:
        # Validate resource_id is an integer to prevent injection
        validated_id = int(resource_id)
        return f"{path}/{validated_id}"
    
    return path


def _sanitize_redirect_path(raw: str | None, default: str = "/observations") -> str:
    """
    Sanitize a user-provided redirect path so that only internal, absolute
    paths (starting with a single "/") are allowed.

    This prevents open redirects to external sites by rejecting any value
    that has a scheme or netloc, or that could be interpreted as a
    protocol-relative URL.
    """
    if not raw:
        return default
    # Normalize to string and replace backslashes, which some browsers treat
    # as equivalent to forward slashes in URLs.
    text = str(raw)
    clean = text.replace("\\", "/")
    # Reject any paths with embedded control characters (CR/LF) to prevent response splitting.
    if '\r' in clean or '\n' in clean:
        return default
    # Quickly reject obvious scheme-based URLs like "https:/example.com" or
    # "custom-scheme:foo" that some browsers may still treat as external.
    # If there is a ":" before any "/", treat it as a (possibly malformed) scheme.
    first_colon = clean.find(":")
    first_slash = clean.find("/")
    if first_colon != -1 and (first_slash == -1 or first_colon < first_slash):
        return default
    # Also reject protocol-relative URLs or anything starting with multiple slashes.
    if clean.startswith("//"):
        return default
    parsed = urlsplit(clean)
    # Disallow any explicit scheme or network location (external URLs).
    if parsed.scheme or parsed.netloc:
        return default
    # Require a leading "/" for internal absolute paths.
    if not parsed.path.startswith("/"):
        return default
    return urlunsplit(('', '', parsed.path, parsed.query, parsed.fragment))


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

    async def delete(self, *args: Any, **kwargs: Any) -> Any:
        session = await self._ensure_session()
        return await self._run(session.delete, *args, **kwargs)

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


def _get_snapshot_path(obs: Observation, key: str) -> str | None:
    extra = obs.get_extra()
    if not extra or not isinstance(extra, dict):
        return None
    snapshots_data = extra.get("snapshots")
    if not isinstance(snapshots_data, dict):
        return None
    return snapshots_data.get(key)


def get_annotated_snapshot_path(obs: Observation) -> str | None:
    """
    Get the annotated snapshot path for an observation from its extra_json.

    Returns the annotated path if available, otherwise None.
    """
    return _get_snapshot_path(obs, "annotated_path")


def get_observation_media_paths(obs: Observation) -> dict[str, str]:
    """
    Extract media-related paths from observation extra metadata.
    """
    extra = obs.get_extra()
    if not extra or not isinstance(extra, dict):
        return {}
    media = extra.get("media")
    if not isinstance(media, dict):
        return {}
    return {k: v for k, v in media.items() if isinstance(v, str) and v}


def _normalize_candidate_label(raw: str | None) -> str:
    if not raw:
        return ""
    clean = raw.strip().lower().replace(" ", "_")
    return clean


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
    return _get_snapshot_path(obs, "clip_path")
    
    
def get_roi_snapshot_path(obs: Observation) -> str | None:
    """
    Get the ROI-cropped snapshot path for an observation from its extra_json.
    
    Returns the ROI snapshot path if available, otherwise None.
    """
    return _get_snapshot_path(obs, "roi_path")


def get_background_snapshot_path(obs: Observation) -> str | None:
    """
    Get the background snapshot path for an observation from its extra_json.

    Returns the background snapshot path if available, otherwise None.
    """
    return _get_snapshot_path(obs, "background_path")


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


def _parse_date_filter(raw: str | None, *, end: bool = False) -> datetime | None:
    if not raw:
        return None
    try:
        if len(raw) == 10:
            day = date.fromisoformat(raw)
            dt = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    if end and len(raw) == 10:
        dt = dt + timedelta(days=1)
    return dt


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
    # Full capture
    parse_float("arrival_buffer_seconds", "Arrival buffer seconds", 0.0, 30.0)
    parse_float("departure_timeout_seconds", "Departure timeout seconds", 0.5, 60.0)
    parse_float("post_departure_buffer_seconds", "Post-departure buffer seconds", 0.0, 30.0)

    # New fields
    parse_float("fps_limit", "FPS limit", 1.0, 60.0)

    parse_float("crop_padding", "Crop padding", 0.0, 0.5)
    parse_float("bg_rejected_cooldown_seconds", "Rejected cooldown", 0.0, 60.0)

    bg_enabled_raw = str(raw.get("bg_subtraction_enabled", "")).strip().lower()
    parsed["bg_subtraction_enabled"] = bg_enabled_raw in {"1", "true", "yes", "on"}

    # New boolean fields
    bg_log_rejected_raw = str(raw.get("bg_log_rejected", "")).strip().lower()
    parsed["bg_log_rejected"] = bg_log_rejected_raw in {"1", "true", "yes", "on"}

    bg_rejected_save_clip_raw = str(raw.get("bg_rejected_save_clip", "")).strip().lower()
    parsed["bg_rejected_save_clip"] = bg_rejected_save_clip_raw in {"1", "true", "yes", "on"}

    bg_save_masks_raw = str(raw.get("bg_save_masks", "")).strip().lower()
    parsed["bg_save_masks"] = bg_save_masks_raw in {"1", "true", "yes", "on"}

    bg_save_mask_overlay_raw = str(raw.get("bg_save_mask_overlay", "")).strip().lower()
    parsed["bg_save_mask_overlay"] = bg_save_mask_overlay_raw in {"1", "true", "yes", "on"}

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


PAGE_SIZE_OPTIONS = [10, 25, 50, 100, 200, 500]


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
        try:
            yield
        finally:
            await dispose_async_engine()

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
            # New fields
            "fps_limit": f"{float(getattr(settings, 'fps_limit', 8.0)):.1f}",

            "arrival_buffer_seconds": f"{float(getattr(settings, 'arrival_buffer_seconds', 5.0)):.1f}",
            "departure_timeout_seconds": f"{float(getattr(settings, 'departure_timeout_seconds', 2.0)):.1f}",
            "post_departure_buffer_seconds": f"{float(getattr(settings, 'post_departure_buffer_seconds', 3.0)):.1f}",
            "crop_padding": f"{float(getattr(settings, 'crop_padding', 0.05)):.2f}",
            "bg_log_rejected": "1" if getattr(settings, "bg_log_rejected", False) else "0",
            "bg_rejected_cooldown_seconds": f"{float(getattr(settings, 'bg_rejected_cooldown_seconds', 3.0)):.1f}",
            "bg_rejected_save_clip": "1" if getattr(settings, "bg_rejected_save_clip", False) else "0",
            "bg_save_masks": "1" if getattr(settings, "bg_save_masks", True) else "0",
            "bg_save_mask_overlay": "1" if getattr(settings, "bg_save_mask_overlay", True) else "0",
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
                total_recent, page=page, page_size=page_size, max_page_size=500
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
                recent_page_size_options=PAGE_SIZE_OPTIONS,
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
        page: int = 1,
        page_size: int = 10,
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> HTMLResponse:
        s = load_settings()

        count_query = select(func.count(Observation.id))
        if individual_id is not None:
            count_query = count_query.where(Observation.individual_id == individual_id)
        total = (await db.execute(count_query)).scalar_one()

        current_page, clamped_page_size, total_pages, offset = paginate(
            total, page=page, page_size=page_size, max_page_size=max(PAGE_SIZE_OPTIONS)
        )

        q = select(Observation).order_by(desc(Observation.ts)).offset(offset).limit(clamped_page_size)
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

        # Cache individuals for dropdown - rarely changes and expensive to query
        individuals_cache_key = "hbmon:individuals:dropdown"
        cached_inds = await cache_get_json(individuals_cache_key)
        if cached_inds:
            inds_data = cached_inds
        else:
            inds = (
                await db.execute(
                    select(Individual.id, Individual.name, Individual.visit_count)
                    .order_by(desc(Individual.visit_count))
                    .limit(100)  # Reduced from 2000 - most users won't need more
                )
            ).all()
            inds_data = [{"id": r[0], "name": r[1], "visit_count": r[2]} for r in inds]
            await cache_set_json(individuals_cache_key, inds_data, ttl_seconds=60)  # Cache for 60 seconds

        return templates.TemplateResponse(
            request,
            "observations.html",
            _context(
                request,
                "Observations",
                settings=s,
                observations=obs,
                individuals=inds_data,
                extra_columns=extra_columns,
                extra_column_sort_types=extra_sort_types,
                extra_column_labels=extra_labels,
                extra_column_defaults=extra_column_defaults,
                selected_individual=individual_id,
                obs_page=current_page,
                obs_page_size=clamped_page_size,
                obs_total_pages=total_pages,
                limit_options=PAGE_SIZE_OPTIONS,
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
        roi_snapshot_path = get_roi_snapshot_path(o)
        clip_snapshot_path = get_clip_snapshot_path(o)
        background_snapshot_path = get_background_snapshot_path(o)
        media_paths = get_observation_media_paths(o)
        mask_path = media_paths.get("mask_path")
        mask_overlay_path = media_paths.get("mask_overlay_path")

        # Video file diagnostics
        video_info: dict[str, Any] | None = None
        if o.video_path:
            video_file = media_dir() / o.video_path
            exists = await _run_blocking(video_file.exists)
            size_kb = 0.0
            suffix = ""
            fps = None
            width = None
            height = None
            duration = None
            fourcc = None
            
            if exists:
                try:
                    size_kb = round((await _run_blocking(video_file.stat)).st_size / 1024, 2)
                except OSError:
                    pass
                suffix = video_file.suffix.lower()
                
                # Extract video metadata using OpenCV
                def _extract_video_metadata(path: Path) -> dict[str, Any]:
                    """Extract video metadata using OpenCV."""
                    metadata: dict[str, Any] = {}
                    try:
                        cv2 = _load_cv2()
                        cap = cv2.VideoCapture(str(path))
                        if cap.isOpened():
                            # Get FPS
                            fps_val = cap.get(cv2.CAP_PROP_FPS)
                            if fps_val > 0:
                                metadata["fps"] = round(fps_val, 2)
                            
                            # Get resolution
                            width_val = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height_val = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            if width_val > 0 and height_val > 0:
                                metadata["width"] = width_val
                                metadata["height"] = height_val
                            
                            # Get frame count and duration
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            if frame_count > 0 and fps_val > 0:
                                metadata["duration"] = round(frame_count / fps_val, 2)
                            
                            # Get codec (fourcc)
                            fourcc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
                            if fourcc_val != 0:
                                codec_str = "".join([chr((fourcc_val >> 8 * i) & 0xFF) for i in range(4)])
                                metadata["fourcc"] = codec_str
                            
                            cap.release()
                    except Exception:
                        pass
                    return metadata
                
                try:
                    video_metadata = await _run_blocking(_extract_video_metadata, video_file)
                    fps = video_metadata.get("fps")
                    width = video_metadata.get("width")
                    height = video_metadata.get("height")
                    duration = video_metadata.get("duration")
                    fourcc = video_metadata.get("fourcc")
                except Exception:
                    pass
            
            video_info = {
                "exists": exists,
                "size_kb": size_kb,
                "suffix": suffix,
                "fps": fps,
                "width": width,
                "height": height,
                "duration": duration,
                "fourcc": fourcc,
            }

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
                roi_snapshot_path=roi_snapshot_path,
                clip_snapshot_path=clip_snapshot_path,
                background_snapshot_path=background_snapshot_path,
                mask_path=mask_path,
                mask_overlay_path=mask_overlay_path,
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

        return RedirectResponse(url=_safe_internal_url("/observations", obs_id), status_code=303)

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
        media_paths = get_observation_media_paths(o)
        _safe_unlink_media(media_paths.get("mask_path"))
        _safe_unlink_media(media_paths.get("mask_overlay_path"))

        await db.execute(delete(Embedding).where(Embedding.observation_id == obs_id))
        await db.delete(o)
        await _commit_with_retry(db)

        if ind_id is not None:
            await _recompute_individual_stats(db, int(ind_id))
        await _commit_with_retry(db)

        return RedirectResponse(url="/observations", status_code=303)

    @app.get("/candidates", response_class=HTMLResponse)
    async def candidates(
        request: Request,
        page: int = 1,
        page_size: int = 10,
        label: str | None = None,
        reason: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> HTMLResponse:
        s = load_settings()

        start_dt = _parse_date_filter(start_date)
        end_dt = _parse_date_filter(end_date, end=True)
        label_filter = _normalize_candidate_label(label)
        reason_filter = (reason or "").strip()

        q = select(Candidate).order_by(desc(Candidate.ts))
        count_query = select(func.count(Candidate.id))
        if start_dt is not None:
            q = q.where(Candidate.ts >= start_dt)
            count_query = count_query.where(Candidate.ts >= start_dt)
        if end_dt is not None:
            q = q.where(Candidate.ts < end_dt)
            count_query = count_query.where(Candidate.ts < end_dt)

        dialect = _get_db_dialect_name(db)
        use_db_filters = dialect in {"sqlite", "postgres", "postgresql"}
        if use_db_filters:
            if reason_filter:
                reason_expr = _candidate_json_value(Candidate.extra_json, ["reason"], dialect)
                if reason_expr is None:
                    use_db_filters = False
                else:
                    q = q.where(reason_expr == reason_filter)
                    count_query = count_query.where(reason_expr == reason_filter)
            if use_db_filters and label_filter:
                label_expr = _candidate_json_value(Candidate.extra_json, ["review", "label"], dialect)
                if label_expr is None:
                    use_db_filters = False
                else:
                    q = q.where(label_expr == label_filter)
                    count_query = count_query.where(label_expr == label_filter)

        def _decorate_candidate(c: Candidate) -> Candidate:
            extra = c.get_extra() or {}
            review = extra.get("review") if isinstance(extra, dict) else {}
            if not isinstance(review, dict):
                review = {}
            review_label = _normalize_candidate_label(review.get("label"))
            motion = extra.get("motion") if isinstance(extra, dict) else {}
            detection = extra.get("detection") if isinstance(extra, dict) else {}
            media = extra.get("media") if isinstance(extra, dict) else {}
            if not isinstance(media, dict):
                media = {}
            annotated_fallback = media.get("snapshot_annotated_path")
            if not isinstance(annotated_fallback, str):
                annotated_fallback = None
            c.display_snapshot_path = (  # type: ignore[attr-defined]
                c.annotated_snapshot_path or annotated_fallback or c.snapshot_path
            )
            c.review_label = review_label or None  # type: ignore[attr-defined]
            c.motion_overlap_ratio = None  # type: ignore[attr-defined]
            if isinstance(motion, dict):
                c.motion_overlap_ratio = motion.get("bbox_overlap_ratio")  # type: ignore[attr-defined]
            c.detection_confidence = None  # type: ignore[attr-defined]
            if isinstance(detection, dict):
                c.detection_confidence = detection.get("confidence")  # type: ignore[attr-defined]
            return c

        if use_db_filters:
            total = int((await db.execute(count_query)).scalar_one())
            current_page, clamped_page_size, total_pages, offset = paginate(
                total, page=page, page_size=page_size, max_page_size=max(PAGE_SIZE_OPTIONS)
            )
            page_rows = (
                await db.execute(q.offset(offset).limit(clamped_page_size))
            ).scalars().all()
            page_rows = [_decorate_candidate(c) for c in page_rows]
        else:
            rows = (await db.execute(q)).scalars().all()
            filtered: list[Candidate] = []
            for c in rows:
                extra = c.get_extra() or {}
                review = extra.get("review") if isinstance(extra, dict) else {}
                if not isinstance(review, dict):
                    review = {}
                review_label = _normalize_candidate_label(review.get("label"))
                reason_value = extra.get("reason") if isinstance(extra, dict) else None

                if reason_filter and reason_value != reason_filter:
                    continue
                if label_filter and review_label != label_filter:
                    continue
                filtered.append(_decorate_candidate(c))

            total = len(filtered)
            current_page, clamped_page_size, total_pages, offset = paginate(
                total, page=page, page_size=page_size, max_page_size=max(PAGE_SIZE_OPTIONS)
            )
            page_rows = filtered[offset:offset + clamped_page_size]

        return templates.TemplateResponse(
            request,
            "candidates.html",
            _context(
                request,
                "Candidates",
                settings=s,
                candidates=page_rows,
                candidate_labels=CANDIDATE_REVIEW_LABELS,
                candidates_page=current_page,
                candidates_page_size=clamped_page_size,
                candidates_total_pages=total_pages,
                limit_options=PAGE_SIZE_OPTIONS,
                selected_label=label_filter,
                selected_reason=reason_filter,
                selected_start_date=start_date or "",
                selected_end_date=end_date or "",
                count_shown=len(page_rows),
                count_total=int(total),
            ),
        )

    @app.get("/candidates/{candidate_id}", response_class=HTMLResponse)
    async def candidate_detail(
        candidate_id: int,
        request: Request,
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> HTMLResponse:
        c = await db.get(Candidate, candidate_id)
        if c is None:
            raise HTTPException(status_code=404, detail="Candidate not found")

        extra = c.get_extra() or {}
        media = extra.get("media") if isinstance(extra, dict) else {}
        if not isinstance(media, dict):
            media = {}
        review = extra.get("review") if isinstance(extra, dict) else {}
        if not isinstance(review, dict):
            review = {}
        review_label = _normalize_candidate_label(review.get("label"))

        mask_path = c.mask_path or media.get("mask_path")
        mask_overlay_path = c.mask_overlay_path or media.get("mask_overlay_path")
        annotated_path = c.annotated_snapshot_path or (
            media.get("snapshot_annotated_path") if isinstance(media.get("snapshot_annotated_path"), str) else None
        )

        c.extra_pretty = pretty_json_obj(extra)  # type: ignore[attr-defined]
        c.review_label = review_label or None  # type: ignore[attr-defined]

        return templates.TemplateResponse(
            request,
            "candidate_detail.html",
            _context(
                request,
                f"Candidate {c.id}",
                candidate=c,
                extra=extra,
                allowed_review_labels=CANDIDATE_REVIEW_LABELS,
                mask_path=mask_path,
                mask_overlay_path=mask_overlay_path,
                annotated_snapshot_path=annotated_path,
            ),
        )

    @app.post("/candidates/{candidate_id}/label")
    async def label_candidate(
        candidate_id: int,
        label: str = Form(""),
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> RedirectResponse:
        c = await db.get(Candidate, candidate_id)
        if c is None:
            raise HTTPException(status_code=404, detail="Candidate not found")

        raw_label = label or ""
        if len(raw_label) > 64:
            raise HTTPException(status_code=400, detail="Label too long")
        clean = _normalize_candidate_label(raw_label)
        allowed = set(CANDIDATE_REVIEW_LABELS)
        review_label = clean if clean in allowed else ""

        if review_label:
            c.merge_extra(
                {
                    "review": {
                        "label": review_label,
                        "labeled_at_utc": _as_utc_str(datetime.now(timezone.utc)),
                    }
                }
            )
        else:
            extra = c.get_extra() or {}
            if isinstance(extra, dict):
                extra_copy = dict(extra)
                raw_review = extra_copy.get("review")
                review = dict(raw_review) if isinstance(raw_review, dict) else {}
                review.pop("label", None)
                review.pop("labeled_at_utc", None)
                if review:
                    extra_copy["review"] = review
                else:
                    extra_copy.pop("review", None)
                c.set_extra(extra_copy)

        await _commit_with_retry(db)
        return RedirectResponse(url=_safe_internal_url("/candidates", candidate_id), status_code=303)

    @app.post("/candidates/{candidate_id}/export_integration_test")
    async def export_candidate_integration_test(
        candidate_id: int,
        case_name: str = Form(""),
        description: str = Form(""),
        behavior: str = Form(""),
        location: str = Form(""),
        human_verified: str = Form("false"),
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> FileResponse:
        c = await db.get(Candidate, candidate_id)
        if c is None:
            raise HTTPException(status_code=404, detail="Candidate not found")

        extra = c.get_extra() or {}
        extra_copy = extra.copy() if isinstance(extra, dict) else {}
        media = extra_copy.get("media")
        media_data = dict(media) if isinstance(media, dict) else {}

        mask_rel = c.mask_path or media_data.get("mask_path")
        mask_overlay_rel = c.mask_overlay_path or media_data.get("mask_overlay_path")
        annotated_rel = c.annotated_snapshot_path or media_data.get("snapshot_annotated_path")
        clip_rel = c.clip_path or media_data.get("clip_path")

        if mask_rel:
            media_data["mask_path"] = "mask.png"
        if mask_overlay_rel:
            media_data["mask_overlay_path"] = "mask_overlay.png"
        if annotated_rel:
            media_data["snapshot_annotated_path"] = "snapshot_annotated.jpg"
        if clip_rel:
            media_data["clip_path"] = "clip.mp4"
        if media_data:
            extra_copy["media"] = media_data

        bg = extra_copy.get("bg") if isinstance(extra_copy.get("bg"), dict) else {}
        motion = extra_copy.get("motion") if isinstance(extra_copy.get("motion"), dict) else {}

        original_candidate = {
            "bbox_xyxy": list(c.bbox_xyxy) if c.bbox_xyxy else None,
            "extra": extra_copy,
        }

        expected = {
            "detection": False,
            "behavior": behavior or "unknown",
            "human_verified": human_verified.strip().lower() in {"1", "true", "yes", "y", "on"},
        }

        sensitivity_params = {
            "bg_threshold": bg.get("threshold") if isinstance(bg, dict) else None,
            "bg_blur": bg.get("blur") if isinstance(bg, dict) else None,
            "bg_min_overlap": bg.get("min_overlap") if isinstance(bg, dict) else None,
            "bg_active": bg.get("active") if isinstance(bg, dict) else None,
        }

        metadata = {
            "description": description or f"Candidate {c.id} integration test",
            "expected": expected,
            "source": {
                "camera": c.camera_name or "",
                "timestamp_utc": c.ts_utc,
                "location": location or "",
            },
            "sensitivity_tests": [
                {
                    "name": "default",
                    "params": sensitivity_params,
                    "expected_detection": expected["detection"],
                }
            ],
            "motion": motion,
            "original_candidate": original_candidate,
        }

        ensure_dirs()
        out_dir = data_dir() / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        safe_case = _sanitize_case_name(case_name, f"candidate_{c.id}")
        out_path = out_dir / f"hbmon-candidate-{safe_case}-{stamp}.tar.gz"

        metadata_bytes = json.dumps(metadata, indent=4, sort_keys=True).encode("utf-8")
        metadata_info = tarfile.TarInfo(name=f"{safe_case}/metadata.json")
        metadata_info.size = len(metadata_bytes)

        snapshot_path = media_dir() / c.snapshot_path
        annotated_path = (media_dir() / annotated_rel) if isinstance(annotated_rel, str) else None
        clip_path = (media_dir() / clip_rel) if isinstance(clip_rel, str) else None
        mask_path = (media_dir() / mask_rel) if isinstance(mask_rel, str) else None
        mask_overlay_path = (media_dir() / mask_overlay_rel) if isinstance(mask_overlay_rel, str) else None

        def _build_bundle() -> None:
            with tarfile.open(out_path, "w:gz") as tf:
                tf.addfile(metadata_info, io.BytesIO(metadata_bytes))
                if snapshot_path.exists():
                    tf.add(snapshot_path, arcname=f"{safe_case}/snapshot.jpg")
                if annotated_path and annotated_path.exists():
                    tf.add(annotated_path, arcname=f"{safe_case}/snapshot_annotated.jpg")
                if clip_path and clip_path.exists():
                    tf.add(clip_path, arcname=f"{safe_case}/clip.mp4")
                if mask_path and mask_path.exists():
                    tf.add(mask_path, arcname=f"{safe_case}/mask.png")
                if mask_overlay_path and mask_overlay_path.exists():
                    tf.add(mask_overlay_path, arcname=f"{safe_case}/mask_overlay.png")

        await _run_blocking(_build_bundle)

        return FileResponse(str(out_path), filename=out_path.name, media_type="application/gzip")

    @app.post("/candidates/bulk_delete")
    async def bulk_delete_candidates(
        candidate_ids: list[int] = Form([]),
        redirect_to: str | None = Form(None),
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> RedirectResponse:
        ids = [int(i) for i in candidate_ids if int(i) > 0]
        if not ids:
            redirect_path = _sanitize_redirect_path(redirect_to, default="/candidates")
            return RedirectResponse(url=redirect_path, status_code=303)

        rows = (await db.execute(select(Candidate).where(Candidate.id.in_(ids)))).scalars().all()
        for c in rows:
            _safe_unlink_media(c.snapshot_path)
            _safe_unlink_media(c.annotated_snapshot_path)
            _safe_unlink_media(c.mask_path)
            _safe_unlink_media(c.mask_overlay_path)
            _safe_unlink_media(c.clip_path)
            extra = c.get_extra() or {}
            media = extra.get("media") if isinstance(extra, dict) else {}
            if isinstance(media, dict):
                _safe_unlink_media(media.get("snapshot_raw_path"))
                _safe_unlink_media(media.get("snapshot_annotated_path"))
                _safe_unlink_media(media.get("mask_path"))
                _safe_unlink_media(media.get("mask_overlay_path"))
                _safe_unlink_media(media.get("clip_path"))

        await db.execute(delete(Candidate).where(Candidate.id.in_(ids)))
        await _commit_with_retry(db)

        redirect_path = _sanitize_redirect_path(redirect_to, default="/candidates")
        return RedirectResponse(url=redirect_path, status_code=303)

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
        roi_rel = get_roi_snapshot_path(o)
        media_paths = get_observation_media_paths(o)
        if background_rel or roi_rel:
            snapshots = extra_copy.get("snapshots")
            snapshots_data = dict(snapshots) if isinstance(snapshots, dict) else {}
            if background_rel:
                snapshots_data["background_path"] = "background.jpg"
            if roi_rel:
                snapshots_data["roi_path"] = "roi.jpg"
            extra_copy["snapshots"] = snapshots_data
        if media_paths:
            media_copy = dict(media_paths)
            if media_copy.get("mask_path"):
                media_copy["mask_path"] = "mask.png"
            if media_copy.get("mask_overlay_path"):
                media_copy["mask_overlay_path"] = "mask_overlay.png"
            extra_copy["media"] = media_copy

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
        roi_path = (media_dir() / roi_rel) if roi_rel else None
        mask_rel = media_paths.get("mask_path")
        mask_overlay_rel = media_paths.get("mask_overlay_path")
        mask_path = (media_dir() / mask_rel) if mask_rel else None
        mask_overlay_path = (media_dir() / mask_overlay_rel) if mask_overlay_rel else None

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
                if roi_path and roi_path.exists():
                    tf.add(roi_path, arcname=f"{safe_case}/roi.jpg")
                if mask_path and mask_path.exists():
                    tf.add(mask_path, arcname=f"{safe_case}/mask.png")
                if mask_overlay_path and mask_overlay_path.exists():
                    tf.add(mask_overlay_path, arcname=f"{safe_case}/mask_overlay.png")

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

    @app.post("/individuals/bulk_delete")
    async def bulk_delete_individuals(
        ind_ids: list[int] = Form([]),
        redirect_to: str | None = Form(None),
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> RedirectResponse:
        redirect_path = _sanitize_redirect_path(redirect_to, default="/individuals")
        ids = [int(i) for i in ind_ids if int(i) > 0]
        if not ids:
            return RedirectResponse(url=redirect_path, status_code=303)

        # Delete individuals (observations effectively become orphaned via ON DELETE SET NULL)
        await db.execute(delete(Individual).where(Individual.id.in_(ids)))
        await _commit_with_retry(db)

        return RedirectResponse(url=redirect_path, status_code=303)

    @app.get("/individuals", response_class=HTMLResponse)
    async def individuals(
        request: Request,
        page: int = 1,
        page_size: int = 10,
        sort: str = "visits",
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> HTMLResponse:
        sort = (sort or "visits").lower()

        total = (await db.execute(select(func.count(Individual.id)))).scalar_one()
        current_page, clamped_page_size, total_pages, offset = paginate(
            total, page=page, page_size=page_size, max_page_size=max(PAGE_SIZE_OPTIONS)
        )

        q = select(Individual)
        if sort == "id":
            q = q.order_by(Individual.id)
        elif sort == "recent":
            q = q.order_by(Individual.last_seen_at.desc().nullslast())
        else:
            q = q.order_by(desc(Individual.visit_count))

        inds = (await db.execute(q.offset(offset).limit(clamped_page_size))).scalars().all()
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
                individuals_page=current_page,
                individuals_page_size=clamped_page_size,
                individuals_total_pages=total_pages,
                limit_options=PAGE_SIZE_OPTIONS,
                count_shown=len(inds),
                count_total=int(total),
            ),
        )

    @app.get("/individuals/{individual_id}", response_class=HTMLResponse)
    async def individual_detail(
        individual_id: int,
        request: Request,
        page: int = 1,
        page_size: int = 10,
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> HTMLResponse:
        ind = await db.get(Individual, individual_id)
        if ind is None:
            raise HTTPException(status_code=404, detail="Individual not found")

        total_obs = (
            await db.execute(
                select(func.count(Observation.id)).where(Observation.individual_id == individual_id)
            )
        ).scalar_one()
        current_page, clamped_page_size, total_pages, offset = paginate(
            total_obs, page=page, page_size=page_size, max_page_size=500
        )

        obs = (
            await db.execute(
                select(Observation)
                .where(Observation.individual_id == individual_id)
                .order_by(desc(Observation.ts))
                .offset(offset)
                .limit(clamped_page_size)
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

        total = int(total_obs)

        last_seen = _as_utc_str(ind.last_seen_at)

        # Hour-of-day counts in UTC for heatmap bucketing.
        # Observation.ts stored as timezone-aware.
        db_dialect = _get_db_dialect_name(db)
        if db_dialect == "sqlite":
            hour_expr = cast(func.strftime("%H", Observation.ts), Integer).label("hh")
        elif db_dialect == "postgresql":
            hour_expr = cast(
                func.extract("hour", func.timezone("UTC", Observation.ts)),
                Integer,
            ).label("hh")
        else:
            hour_expr = cast(func.extract("hour", Observation.ts), Integer).label("hh")
        rows = (
            await db.execute(
                select(hour_expr, func.count(Observation.id))
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
                obs_page=current_page,
                obs_page_size=clamped_page_size,
                obs_total_pages=total_pages,
                obs_total=int(total_obs),
                obs_page_size_options=PAGE_SIZE_OPTIONS,
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

        return RedirectResponse(url=_safe_internal_url("/individuals", individual_id), status_code=303)

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
            return RedirectResponse(url=_safe_internal_url("/individuals", individual_id), status_code=303)

        vecs = [e.get_vec() for e in embs]
        proto = l2_normalize(sum(vecs) / max(1, len(vecs)))
        ind.set_prototype(proto)
        await db.commit()

        return RedirectResponse(url=_safe_internal_url("/individuals", individual_id), status_code=303)

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
            return RedirectResponse(url=_safe_internal_url("/individuals", individual_id), status_code=303)

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

        return RedirectResponse(url=_safe_internal_url("/individuals", ind_b.id), status_code=303)

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
            # New fields
            "fps_limit",

            "arrival_buffer_seconds",
            "departure_timeout_seconds",
            "post_departure_buffer_seconds",
            "crop_padding",
            "bg_rejected_cooldown_seconds",
        )
        raw = {name: str(form.get(name, "") or "").strip() for name in field_names}
        raw["timezone"] = str(form.get("timezone", "") or "").strip()
        bool_field_names = (
            "bg_subtraction_enabled",
            "bg_log_rejected",
            "bg_rejected_save_clip",
            "bg_save_masks",
            "bg_save_mask_overlay",
        )
        for name in bool_field_names:
            raw[name] = "1" if form.get(name) else "0"
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
        # New fields
        s.fps_limit = parsed["fps_limit"]

        s.arrival_buffer_seconds = parsed["arrival_buffer_seconds"]
        s.departure_timeout_seconds = parsed["departure_timeout_seconds"]
        s.post_departure_buffer_seconds = parsed["post_departure_buffer_seconds"]
        s.crop_padding = parsed["crop_padding"]
        s.bg_log_rejected = parsed["bg_log_rejected"]
        s.bg_rejected_cooldown_seconds = parsed["bg_rejected_cooldown_seconds"]
        s.bg_rejected_save_clip = parsed["bg_rejected_save_clip"]
        s.bg_save_masks = parsed["bg_save_masks"]
        s.bg_save_mask_overlay = parsed["bg_save_mask_overlay"]
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
        page_size: int = 10,
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
            total_obs, page=page, page_size=page_size, max_page_size=500
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
                obs_page_size_options=PAGE_SIZE_OPTIONS,
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

    def _get_snapshot_config():
        """
        Return (snapshot_url: str|None, timeout: float, retries: int).
        Env vars:
          - HBMON_RTSP_SNAPSHOT_URL: explicit full snapshot URL (preferred)
          - HBMON_SNAPSHOT_TIMEOUT: float seconds (default 10.0)
          - HBMON_SNAPSHOT_RETRIES: int retries (default 10)
        """
        url = os.getenv("HBMON_RTSP_SNAPSHOT_URL") or None
        try:
            timeout = float(os.getenv("HBMON_SNAPSHOT_TIMEOUT", "10"))
        except Exception:
            timeout = 10.0
        try:
            retries = int(os.getenv("HBMON_SNAPSHOT_RETRIES", "10"))
        except Exception:
            retries = 10
        if retries < 1:
            retries = 1
        if timeout <= 0:
            timeout = 10.0
        return url, timeout, retries

    def _fetch_snapshot_http(url: str, timeout: float = 10.0, retries: int = 10) -> bytes:
        """
        Fetch a snapshot via HTTP with simple retries and return raw bytes.

        Raises HTTPException(status_code=503, detail=...) on failure.
        """
        last_exc = None
        attempts = max(1, int(retries))
        for attempt in range(1, attempts + 1):
            try:
                req = UrllibRequest(url, headers={"User-Agent": "hbmon/1"})
                with urlopen(req, timeout=timeout) as res:
                    ct = (res.getheader("Content-Type") or "").lower()
                    if not ct.startswith("image/"):
                        raise HTTPException(status_code=503, detail="Snapshot endpoint did not return an image")
                    data = res.read()
                    if not data:
                        raise HTTPException(status_code=503, detail="Snapshot endpoint returned empty body")
                    return data
            except (HTTPError, URLError, ValueError) as e:
                last_exc = e
                # backoff: small sleep (blocking; caller should run this in threadpool)
                try:
                    time.sleep(0.5 * attempt)
                except Exception:
                    pass
                continue
            except Exception as e:
                last_exc = e
                try:
                    time.sleep(0.5 * attempt)
                except Exception:
                    pass
                continue
        detail = f"Snapshot fetch failed: {last_exc}" if last_exc is not None else "Snapshot fetch failed"
        raise HTTPException(status_code=503, detail=detail)

    def _encode_jpeg(frame: Any, cv2: Any, quality: int = 100) -> bytes:
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode frame")
        return jpeg.tobytes()

    def _get_live_snapshot_bytes_sync(settings, jpeg_quality: int = 100) -> bytes:
        """
        Synchronous helper: return JPEG bytes for a live snapshot.

        Preference:
          1) HBMON_RTSP_SNAPSHOT_URL via _fetch_snapshot_http
          2) RTSP capture via _capture_live_frame + _encode_jpeg

        This is meant to be run inside _run_blocking from async endpoints.
        """
        snapshot_url, timeout, retries = _get_snapshot_config()
        if snapshot_url:
            # _fetch_snapshot_http is synchronous/blocking (uses urlopen)
            return _fetch_snapshot_http(snapshot_url, timeout, retries)

        # fallback to RTSP
        rtsp_url = getattr(settings, "rtsp_url", "") if settings is not None else None
        if not rtsp_url:
            raise HTTPException(status_code=503, detail="RTSP URL not configured")

        frame, cv2 = _capture_live_frame(rtsp_url)
        return _encode_jpeg(frame, cv2, jpeg_quality)

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
        s = load_settings()
        # get bytes (http snapshot preferred, otherwise RTSP)
        jpeg_bytes = await _run_blocking(_get_live_snapshot_bytes_sync, s, 100)

        # Ensure background directory exists and save (same as current logic)
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

    @app.get("/api/snapshot.jpg")
    async def snapshot_jpg() -> Any:
        """
        Fetch and serve a snapshot from the wyze-bridge HTTP snapshot endpoint.

        This endpoint is optimized for polling-based refresh (every 1-2 seconds)
        and avoids opening RTSP connections. Falls back to live_frame if HTTP
        snapshot URL is not configured.

        Returns the image with short caching headers to allow browser caching.
        """
        snapshot_url, timeout, retries = _get_snapshot_config()
        if not snapshot_url:
            # Fall back to live_frame behavior
            s = load_settings()
            jpeg_bytes = await _run_blocking(_get_live_snapshot_bytes_sync, s, 80)
            return Response(
                content=jpeg_bytes,
                media_type="image/jpeg",
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )

        # Use fast HTTP snapshot with reduced retries for polling
        try:
            jpeg_bytes = await _run_blocking(_fetch_snapshot_http, snapshot_url, timeout, 2)
        except HTTPException:
            # If HTTP snapshot fails, try RTSP fallback
            s = load_settings()
            jpeg_bytes = await _run_blocking(_get_live_snapshot_bytes_sync, s, 80)

        return Response(
            content=jpeg_bytes,
            media_type="image/jpeg",
            headers={
                # Allow browser to cache for 1 second - suitable for 2s refresh
                "Cache-Control": "max-age=1, must-revalidate",
            },
        )

    @app.get("/api/live_frame.jpg")
    async def live_frame_jpg() -> Any:
        s = load_settings()
        # run blocking helper once in threadpool
        jpeg_bytes = await _run_blocking(_get_live_snapshot_bytes_sync, s, 100)
        return Response(
            content=jpeg_bytes,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    @app.get("/api/video/{obs_id}")
    async def stream_video(
        obs_id: int,
        request: Request,
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> Any:
        """
        Stream a video file with on-the-fly H.264 compression for browser playback.
        
        Videos are stored uncompressed on disk to preserve quality for ML training.
        When streaming to browsers, they are compressed on-the-fly using FFmpeg with:
        - HTTP 206 Partial Content support for seeking
        - H.264 codec with configurable CRF (quality)
        - Browser-compatible baseline profile
        - Proper Content-Type headers for MP4 files
        
        This approach provides:
        - Pristine uncompressed videos for ML training/export
        - Efficient compressed streaming for browser playback
        - No duplicate storage (only uncompressed on disk)
        """
        o = await db.get(Observation, obs_id)
        if o is None:
            raise HTTPException(status_code=404, detail="Observation not found")
        
        video_path = o.video_path or ""
        if not video_path:
            raise HTTPException(status_code=404, detail="No video path for this observation")
        
        full_path = media_dir() / video_path
        if not await _run_blocking(full_path.exists):
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Check if on-the-fly compression is enabled
        enable_compression = os.getenv("HBMON_VIDEO_STREAM_COMPRESSION", "1") in ("1", "true", "yes", "on")
        
        if not enable_compression:
            # Serve uncompressed video directly
            return FileResponse(
                str(full_path),
                media_type="video/mp4",
                filename=full_path.name,
            )
        
        # Compress on-the-fly using FFmpeg
        # For range requests, we need to pre-compress to a temp file to support seeking
        # (FFmpeg streaming output doesn't support random access)
        
        import hashlib
        
        # Create cache key from observation ID and compression settings
        crf = int(os.getenv("HBMON_VIDEO_CRF", "23"))
        preset = os.getenv("HBMON_VIDEO_PRESET", "fast")  # Use "fast" for on-the-fly to reduce latency
        cache_key = f"{obs_id}_{crf}_{preset}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        
        # Use temp directory for compressed cache
        temp_dir = media_dir() / ".cache" / "compressed"
        await _run_blocking(temp_dir.mkdir, parents=True, exist_ok=True)
        
        cached_path = temp_dir / f"{full_path.stem}_{cache_hash}.mp4"
        
        # Check if cached compressed version exists and is newer than source
        needs_compression = True
        if await _run_blocking(cached_path.exists):
            try:
                source_mtime = (await _run_blocking(full_path.stat)).st_mtime
                cache_mtime = (await _run_blocking(cached_path.stat)).st_mtime
                if cache_mtime >= source_mtime:
                    needs_compression = False
            except OSError:
                pass
        
        if needs_compression:
            # Compress to cache using FFmpeg
            def _compress_for_streaming(input_path: Path, output_path: Path) -> bool:
                """Compress video for browser streaming."""
                import subprocess
                
                ffmpeg_path = os.getenv("HBMON_FFMPEG_PATH", "ffmpeg")
                
                try:
                    cmd = [
                        ffmpeg_path,
                        "-i", str(input_path),
                        "-c:v", "libx264",
                        "-crf", str(crf),
                        "-preset", preset,
                        "-profile:v", "baseline",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        "-y",
                        str(output_path)
                    ]
                    
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    
                    return result.returncode == 0
                except Exception as e:
                    logger.error(f"FFmpeg compression error: {e}")
                    return False
            
            # Compress in background thread
            success = await _run_blocking(_compress_for_streaming, full_path, cached_path)
            
            if not success:
                # Fall back to uncompressed if compression fails
                logger.warning(f"FFmpeg compression failed for observation {obs_id}, serving uncompressed")
                return FileResponse(
                    str(full_path),
                    media_type="video/mp4",
                    filename=full_path.name,
                )
        
        # Serve compressed cached version
        return FileResponse(
            str(cached_path),
            media_type="video/mp4",
            filename=full_path.name,
        )


    @app.get("/api/streaming_bitrate/{obs_id}")
    async def streaming_bitrate(
        obs_id: int,
        db: AsyncSession | _AsyncSessionAdapter = Depends(get_db_dep),
    ) -> dict[str, Any]:
        """
        Get streaming bitrate information for a cached compressed video.
        
        Returns bitrate and compression ratio if cached version exists.
        This is called by the UI after video loads to show streaming efficiency.
        """
        o = await db.get(Observation, obs_id)
        if o is None:
            raise HTTPException(status_code=404, detail="Observation not found")
        
        video_path = o.video_path or ""
        if not video_path:
            return {"bitrate_mbps": None}
        
        full_path = media_dir() / video_path
        if not await _run_blocking(full_path.exists):
            return {"bitrate_mbps": None}
        
        # Get cached compressed video path
        import hashlib
        crf = int(os.getenv("HBMON_VIDEO_CRF", "23"))
        preset = os.getenv("HBMON_VIDEO_PRESET", "fast")
        cache_key = f"{obs_id}_{crf}_{preset}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        
        temp_dir = media_dir() / ".cache" / "compressed"
        cached_path = temp_dir / f"{full_path.stem}_{cache_hash}.mp4"
        
        # Check if cached version exists
        if not await _run_blocking(cached_path.exists):
            return {"bitrate_mbps": None}
        
        try:
            # Get file sizes
            source_size = (await _run_blocking(full_path.stat)).st_size
            cached_size = (await _run_blocking(cached_path.stat)).st_size
            
            # Extract duration from metadata in extra_json if available
            duration = None
            if o.extra_json:
                extra = o.get_extra()
                if isinstance(extra, dict) and "media" in extra:
                    media = extra.get("media", {})
                    if isinstance(media, dict) and "video" in media:
                        video_meta = media.get("video", {})
                        if isinstance(video_meta, dict):
                            duration = video_meta.get("duration")
            
            # If duration not in metadata, extract from video using OpenCV
            if not duration:
                def _get_duration(path: Path) -> float | None:
                    try:
                        cv2 = _load_cv2()
                        cap = cv2.VideoCapture(str(path))
                        if cap.isOpened():
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            cap.release()
                            if fps > 0 and frame_count > 0:
                                return frame_count / fps
                    except Exception:
                        pass
                    return None
                
                duration = await _run_blocking(_get_duration, full_path)
            
            if duration and duration > 0:
                # Calculate bitrate in Mbps: (file_size_bytes * 8) / (duration_seconds * 1_000_000)
                bitrate_mbps = (cached_size * 8) / (duration * 1_000_000)
                compression_ratio = source_size / cached_size if cached_size > 0 else 1.0
                
                return {
                    "bitrate_mbps": bitrate_mbps,
                    "compression_ratio": compression_ratio,
                    "cached_size_kb": round(cached_size / 1024, 2),
                    "source_size_kb": round(source_size / 1024, 2),
                }
            
            return {"bitrate_mbps": None}
            
        except Exception as e:
            logger.error(f"Error calculating streaming bitrate: {e}")
            return {"bitrate_mbps": None}


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

            # Extract video metadata using OpenCV (FPS, resolution, codec)
            def _extract_video_metadata(path: Path) -> dict[str, Any]:
                """Extract video metadata using OpenCV."""
                metadata: dict[str, Any] = {}
                try:
                    cv2 = _load_cv2()
                    cap = cv2.VideoCapture(str(path))
                    if cap.isOpened():
                        # Get FPS
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if fps > 0:
                            metadata["fps"] = round(fps, 2)
                        
                        # Get resolution
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        if width > 0 and height > 0:
                            metadata["width"] = width
                            metadata["height"] = height
                            metadata["resolution"] = f"{width}×{height}"
                        
                        # Get frame count and duration
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if frame_count > 0 and fps > 0:
                            metadata["frame_count"] = frame_count
                            metadata["duration_seconds"] = round(frame_count / fps, 2)
                        
                        # Get codec (fourcc)
                        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                        if fourcc != 0:
                            # Convert fourcc int to string
                            codec_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
                            metadata["fourcc"] = codec_str
                        
                        cap.release()
                except Exception as e:
                    metadata["extraction_error"] = str(e)
                return metadata
            
            video_metadata = await _run_blocking(_extract_video_metadata, full_path)
            result.update(video_metadata)

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
