# src/hbmon/config.py
"""
Configuration + paths for hbmon.

Design goals:
- Works the same on bare Linux + in Docker.
- Uses environment variables for "ops" overrides.
- Persists user-tuned settings (ROI, thresholds, etc.) to /data/config.json.
- Keeps everything dependency-free (standard library only).

Key directories (defaults):
- HBMON_DATA_DIR=/data        (sqlite + config.json)
- HBMON_MEDIA_DIR=/media      (snapshots + clips)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# ----------------------------
# Low-level env parsing
# ----------------------------

def env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None and v.strip() else default


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or not v.strip():
        return default
    try:
        return int(v)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or not v.strip():
        return default
    try:
        return float(v)
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or not v.strip():
        return default
    s = v.strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


# ----------------------------
# Data types
# ----------------------------

@dataclass
class Roi:
    """
    Normalized ROI (0..1), inclusive-ish.
    x1 < x2 and y1 < y2.
    """
    x1: float
    y1: float
    x2: float
    y2: float

    def clamp(self) -> "Roi":
        def c(v: float) -> float:
            return max(0.0, min(1.0, float(v)))

        x1, y1, x2, y2 = c(self.x1), c(self.y1), c(self.x2), c(self.y2)
        # enforce ordering
        x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
        y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
        # avoid zero-area ROI
        if x2 - x1 < 1e-6:
            x2 = min(1.0, x1 + 1e-3)
        if y2 - y1 < 1e-6:
            y2 = min(1.0, y1 + 1e-3)
        return Roi(x1=x1, y1=y1, x2=x2, y2=y2)

    def as_tuple(self) -> tuple[float, float, float, float]:
        r = self.clamp()
        return (r.x1, r.y1, r.x2, r.y2)


@dataclass
class Settings:
    """
    Persistent user settings.

    Notes:
    - rtsp_url is usually provided by env HBMON_RTSP_URL, but we also persist it
      so the UI can display what it was last set to.
    - roi is optional until user calibrates.
    """
    # camera / capture
    rtsp_url: str = ""
    camera_name: str = "camera"
    fps_limit: float = 8.0
    clip_seconds: float = 2.0

    # detection tuning
    detect_conf: float = 0.25
    detect_iou: float = 0.45
    min_box_area: int = 600         # ignore tiny boxes
    cooldown_seconds: float = 2.0   # avoid duplicate triggers per same visit

    # classification / re-id tuning
    min_species_prob: float = 0.35
    match_threshold: float = 0.25   # cosine distance threshold (lower = stricter)
    ema_alpha: float = 0.10

    # ROI
    roi: Roi | None = None

    # misc
    last_updated_utc: float = 0.0

    def with_env_overrides(self) -> "Settings":
        """
        Environment overrides always win (runtime ops),
        but we keep the underlying persisted values.
        """
        s = Settings(**asdict(self))
        s.rtsp_url = env_str("HBMON_RTSP_URL", s.rtsp_url)
        s.camera_name = env_str("HBMON_CAMERA_NAME", s.camera_name)

        s.fps_limit = env_float("HBMON_FPS_LIMIT", s.fps_limit)
        s.clip_seconds = env_float("HBMON_CLIP_SECONDS", s.clip_seconds)

        s.detect_conf = env_float("HBMON_DETECT_CONF", s.detect_conf)
        s.detect_iou = env_float("HBMON_DETECT_IOU", s.detect_iou)
        s.min_box_area = env_int("HBMON_MIN_BOX_AREA", s.min_box_area)
        s.cooldown_seconds = env_float("HBMON_COOLDOWN_SECONDS", s.cooldown_seconds)

        s.min_species_prob = env_float("HBMON_MIN_SPECIES_PROB", s.min_species_prob)
        s.match_threshold = env_float("HBMON_MATCH_THRESHOLD", s.match_threshold)
        s.ema_alpha = env_float("HBMON_EMA_ALPHA", s.ema_alpha)

        # ROI can also be overridden via env for debugging
        roi_env = env_str("HBMON_ROI", "")
        if roi_env:
            # format: "x1,y1,x2,y2"
            parts = [p.strip() for p in roi_env.split(",")]
            if len(parts) == 4:
                try:
                    r = Roi(float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])).clamp()
                    s.roi = r
                except ValueError:
                    pass

        return s


# ----------------------------
# Paths
# ----------------------------

def data_dir() -> Path:
    return Path(env_str("HBMON_DATA_DIR", "/data")).expanduser().resolve()


def media_dir() -> Path:
    return Path(env_str("HBMON_MEDIA_DIR", "/media")).expanduser().resolve()


def config_path() -> Path:
    return data_dir() / "config.json"


def db_path() -> Path:
    # used by db.py default if HBMON_DB_URL not set
    return data_dir() / "hbmon.sqlite"


def snapshots_dir() -> Path:
    return media_dir() / "snapshots"


def clips_dir() -> Path:
    return media_dir() / "clips"


def _ensure_dir(path: Path) -> Path:
    """
    Attempt to create a directory and return the path.  If creation fails due
    to permissions (e.g. `/data` on a system without root access), fall back
    to a directory with the same basename in the current working directory.

    Parameters
    ----------
    path : Path
        The desired directory path.

    Returns
    -------
    Path
        The path that was successfully created.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except PermissionError:
        # When running under an unprivileged user (e.g. during testing),
        # attempting to create system-level directories like `/data` or
        # `/media` may raise a permission error.  In that case, create a
        # fallback directory named after the basename of the desired path in
        # the current working directory.  For example, if `path` is `/data`,
        # the fallback will be `./data`.  This avoids cluttering `/` and
        # keeps test artifacts within the repository checkout.
        fallback = Path.cwd() / path.name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def ensure_dirs() -> None:
    """
    Ensure that the data/media directories and their subdirectories exist.

    This function attempts to create the configured directories returned by
    `data_dir()` and `media_dir()`.  If creation fails due to permission
    issues (e.g. lack of access to `/data` or `/media`), it falls back to
    creating similarly named directories in the current working directory
    (`./data` and `./media`) and updates the corresponding environment
    variables (`HBMON_DATA_DIR` and `HBMON_MEDIA_DIR`) so subsequent calls to
    `data_dir()` or `media_dir()` return the fallback paths.  This behavior
    allows the application and tests to run without requiring elevated
    privileges while still respecting user overrides via environment
    variables.
    """
    # handle data dir
    dd_original = data_dir()
    dd = _ensure_dir(dd_original)
    if dd != dd_original:
        # update env so that data_dir() returns the fallback on subsequent
        # calls within this process.  This does not persist across
        # processes, which is fine for tests.
        os.environ['HBMON_DATA_DIR'] = str(dd)

    # handle media dir
    md_original = media_dir()
    md = _ensure_dir(md_original)
    if md != md_original:
        os.environ['HBMON_MEDIA_DIR'] = str(md)

    # create subdirs relative to whatever media_dir() now resolves to
    _ensure_dir(snapshots_dir())
    _ensure_dir(clips_dir())


# ----------------------------
# Serialization
# ----------------------------

def _settings_from_dict(d: dict[str, Any]) -> Settings:
    roi = None
    if isinstance(d.get("roi"), dict):
        rr = d["roi"]
        try:
            roi = Roi(
                x1=float(rr["x1"]),
                y1=float(rr["y1"]),
                x2=float(rr["x2"]),
                y2=float(rr["y2"]),
            ).clamp()
        except Exception:
            roi = None

    # Only accept known fields (avoid config drift)
    s = Settings(
        rtsp_url=str(d.get("rtsp_url", "")),
        camera_name=str(d.get("camera_name", "camera")),
        fps_limit=float(d.get("fps_limit", 8.0)),
        clip_seconds=float(d.get("clip_seconds", 2.0)),
        detect_conf=float(d.get("detect_conf", 0.25)),
        detect_iou=float(d.get("detect_iou", 0.45)),
        min_box_area=int(d.get("min_box_area", 600)),
        cooldown_seconds=float(d.get("cooldown_seconds", 2.0)),
        min_species_prob=float(d.get("min_species_prob", 0.35)),
        match_threshold=float(d.get("match_threshold", 0.25)),
        ema_alpha=float(d.get("ema_alpha", 0.10)),
        roi=roi,
        last_updated_utc=float(d.get("last_updated_utc", 0.0)),
    )
    return s


def load_settings() -> Settings:
    """
    Load persisted settings from /data/config.json, create defaults if missing,
    then apply env overrides.
    """
    ensure_dirs()
    p = config_path()
    if not p.exists():
        s = Settings(last_updated_utc=time.time())
        save_settings(s)
        return s.with_env_overrides()

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("config.json root is not an object")
        s = _settings_from_dict(data)
    except Exception:
        # If config is corrupted, fall back safely.
        s = Settings(last_updated_utc=time.time())

    return s.with_env_overrides()


def save_settings(s: Settings) -> None:
    """
    Persist settings to /data/config.json (atomic write).
    """
    ensure_dirs()
    p = config_path()
    s = Settings(**asdict(s))
    s.last_updated_utc = time.time()

    out: dict[str, Any] = asdict(s)
    # dataclasses nests roi as dict or None, which is fine

    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def roi_to_str(roi: Roi | None) -> str:
    if roi is None:
        return ""
    x1, y1, x2, y2 = roi.clamp().as_tuple()
    return f"{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}"
