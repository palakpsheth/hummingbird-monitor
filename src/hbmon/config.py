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

import errno
import json
import os
import time
from dataclasses import asdict, dataclass, replace
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
    temporal_window_frames: int = 5
    temporal_min_detections: int = 1
    arrival_buffer_seconds: float = 5.0
    departure_timeout_seconds: float = 2.0
    post_departure_buffer_seconds: float = 3.0

    # detection tuning
    detect_conf: float = 0.1
    detect_iou: float = 0.45
    min_box_area: int = 600         # ignore tiny boxes
    cooldown_seconds: float = 2.0   # avoid duplicate triggers per same visit

    # classification / re-id tuning
    min_species_prob: float = 0.35
    match_threshold: float = 0.25   # cosine distance threshold (lower = stricter)
    ema_alpha: float = 0.10
    crop_padding: float = 0.05      # padding fraction around bird bbox for CLIP (lower = tighter crop)

    # ROI
    roi: Roi | None = None

    # Background image: relative path from data_dir (e.g., "background/background.jpg")
    # This is the path to a static copy of the background image.
    background_image: str = ""
    bg_subtraction_enabled: bool = True
    bg_motion_threshold: int = 30
    bg_motion_blur: int = 5
    bg_min_overlap: float = 0.15
    # Background subtraction extras (for rejected candidate logging)
    bg_log_rejected: bool = False
    bg_rejected_cooldown_seconds: float = 3.0
    bg_rejected_save_clip: bool = False
    bg_save_masks: bool = True
    bg_save_mask_overlay: bool = True

    # Tracking mode (alternative to temporal voting)
    use_tracking: bool = False
    track_high_thresh: float = 0.1
    track_low_thresh: float = 0.01
    track_new_thresh: float = 0.15
    track_match_thresh: float = 0.7
    track_buffer_frames: int = 40

    # misc
    timezone: str = "local"
    last_updated_utc: float = 0.0

    def with_env_overrides(self) -> "Settings":
        """
        Environment overrides always win (runtime ops),
        but we keep the underlying persisted values.

        IMPORTANT:
        Do NOT use dataclasses.asdict(self) here, because that converts nested
        dataclasses (like Roi) into plain dicts, which breaks code expecting
        Roi methods like .clamp().
        """
        s = replace(self)
        s.rtsp_url = env_str("HBMON_RTSP_URL", s.rtsp_url)
        s.camera_name = env_str("HBMON_CAMERA_NAME", s.camera_name)

        s.fps_limit = env_float("HBMON_FPS_LIMIT", s.fps_limit)
        s.temporal_window_frames = env_int("HBMON_TEMPORAL_WINDOW_FRAMES", s.temporal_window_frames)
        s.temporal_min_detections = env_int("HBMON_TEMPORAL_MIN_DETECTIONS", s.temporal_min_detections)
        s.arrival_buffer_seconds = env_float("HBMON_ARRIVAL_BUFFER_SECONDS", s.arrival_buffer_seconds)
        s.departure_timeout_seconds = env_float("HBMON_DEPARTURE_TIMEOUT_SECONDS", s.departure_timeout_seconds)
        s.post_departure_buffer_seconds = env_float("HBMON_POST_DEPARTURE_BUFFER_SECONDS", s.post_departure_buffer_seconds)

        s.detect_conf = env_float("HBMON_DETECT_CONF", s.detect_conf)
        s.detect_iou = env_float("HBMON_DETECT_IOU", s.detect_iou)
        s.min_box_area = env_int("HBMON_MIN_BOX_AREA", s.min_box_area)
        s.cooldown_seconds = env_float("HBMON_COOLDOWN_SECONDS", s.cooldown_seconds)

        s.min_species_prob = env_float("HBMON_MIN_SPECIES_PROB", s.min_species_prob)
        s.match_threshold = env_float("HBMON_MATCH_THRESHOLD", s.match_threshold)
        s.ema_alpha = env_float("HBMON_EMA_ALPHA", s.ema_alpha)
        s.crop_padding = env_float("HBMON_CROP_PADDING", s.crop_padding)
        s.timezone = env_str("HBMON_TIMEZONE", s.timezone)
        s.bg_subtraction_enabled = env_bool("HBMON_BG_SUBTRACTION", s.bg_subtraction_enabled)
        s.bg_motion_threshold = env_int("HBMON_BG_MOTION_THRESHOLD", s.bg_motion_threshold)
        s.bg_motion_blur = env_int("HBMON_BG_MOTION_BLUR", s.bg_motion_blur)
        s.bg_min_overlap = env_float("HBMON_BG_MIN_OVERLAP", s.bg_min_overlap)
        s.bg_log_rejected = env_bool("HBMON_BG_LOG_REJECTED", s.bg_log_rejected)
        s.bg_rejected_cooldown_seconds = env_float(
            "HBMON_BG_REJECTED_COOLDOWN_SECONDS", s.bg_rejected_cooldown_seconds
        )
        s.bg_rejected_save_clip = env_bool("HBMON_BG_REJECTED_SAVE_CLIP", s.bg_rejected_save_clip)
        s.bg_save_masks = env_bool("HBMON_BG_SAVE_MASKS", s.bg_save_masks)
        s.bg_save_mask_overlay = env_bool("HBMON_BG_SAVE_MASK_OVERLAY", s.bg_save_mask_overlay)

        # Tracking mode overrides
        s.use_tracking = env_bool("HBMON_USE_TRACKING", s.use_tracking)
        s.track_high_thresh = env_float("HBMON_TRACK_HIGH_THRESH", s.track_high_thresh)
        s.track_low_thresh = env_float("HBMON_TRACK_LOW_THRESH", s.track_low_thresh)
        s.track_new_thresh = env_float("HBMON_TRACK_NEW_THRESH", s.track_new_thresh)
        s.track_match_thresh = env_float("HBMON_TRACK_MATCH_THRESH", s.track_match_thresh)
        s.track_buffer_frames = env_int("HBMON_TRACK_BUFFER_FRAMES", s.track_buffer_frames)

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


def background_dir() -> Path:
    """Directory for storing the static background image."""
    return data_dir() / "background"


def yolo_config_dir() -> Path:
    """Directory for YOLO configuration and settings."""
    return Path(env_str("YOLO_CONFIG_DIR", str(data_dir() / "yolo"))).expanduser().resolve()


def trackers_dir() -> Path:
    """Directory for tracker configuration (ByteTrack, etc.)."""
    return data_dir() / "trackers"


def background_image_path() -> Path:
    """Full path to the persisted background image file."""
    return background_dir() / "background.jpg"


def _ensure_dir(path: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except (PermissionError, OSError) as exc:
        if isinstance(exc, OSError) and exc.errno not in {
            errno.EACCES,
            errno.ENOENT,
            errno.EROFS,
            errno.EPERM,
        }:
            raise
        fallback = Path.cwd() / path.name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def ensure_dirs() -> None:
    dd_original = data_dir()
    dd = _ensure_dir(dd_original)
    if dd != dd_original:
        os.environ["HBMON_DATA_DIR"] = str(dd)

    md_original = media_dir()
    md = _ensure_dir(md_original)
    if md != md_original:
        os.environ["HBMON_MEDIA_DIR"] = str(md)

    _ensure_dir(snapshots_dir())
    _ensure_dir(clips_dir())
    _ensure_dir(background_dir())
    _ensure_dir(yolo_config_dir())
    _ensure_dir(trackers_dir())


# ----------------------------
# Serialization
# ----------------------------

def _settings_from_dict(d: dict[str, Any]) -> Settings:
    def _parse_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if cleaned in {"1", "true", "yes", "y", "on"}:
                return True
            if cleaned in {"0", "false", "no", "n", "off"}:
                return False
        return default

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

    s = Settings(
        rtsp_url=str(d.get("rtsp_url", "")),
        camera_name=str(d.get("camera_name", "camera")),
        fps_limit=float(d.get("fps_limit", 8.0)),
        temporal_window_frames=int(d.get("temporal_window_frames", 5)),
        temporal_min_detections=int(d.get("temporal_min_detections", 1)),
        arrival_buffer_seconds=float(d.get("arrival_buffer_seconds", 5.0)),
        departure_timeout_seconds=float(d.get("departure_timeout_seconds", 2.0)),
        post_departure_buffer_seconds=float(d.get("post_departure_buffer_seconds", 3.0)),
        detect_conf=float(d.get("detect_conf", 0.25)),
        detect_iou=float(d.get("detect_iou", 0.45)),
        min_box_area=int(d.get("min_box_area", 600)),
        cooldown_seconds=float(d.get("cooldown_seconds", 2.0)),
        min_species_prob=float(d.get("min_species_prob", 0.35)),
        match_threshold=float(d.get("match_threshold", 0.25)),
        ema_alpha=float(d.get("ema_alpha", 0.10)),
        crop_padding=float(d.get("crop_padding", 0.05)),
        timezone=str(d.get("timezone", "local")),
        roi=roi,
        background_image=str(d.get("background_image", "")),
        bg_subtraction_enabled=_parse_bool(d.get("bg_subtraction_enabled"), True),
        bg_motion_threshold=int(d.get("bg_motion_threshold", 30)),
        bg_motion_blur=int(d.get("bg_motion_blur", 5)),
        bg_min_overlap=float(d.get("bg_min_overlap", 0.15)),
        bg_log_rejected=_parse_bool(d.get("bg_log_rejected"), False),
        bg_rejected_cooldown_seconds=float(d.get("bg_rejected_cooldown_seconds", 3.0)),
        bg_rejected_save_clip=_parse_bool(d.get("bg_rejected_save_clip"), False),
        bg_save_masks=_parse_bool(d.get("bg_save_masks"), True),
        bg_save_mask_overlay=_parse_bool(d.get("bg_save_mask_overlay"), True),
        # Tracking mode
        use_tracking=_parse_bool(d.get("use_tracking"), True),
        track_high_thresh=float(d.get("track_high_thresh", 0.1)),
        track_low_thresh=float(d.get("track_low_thresh", 0.01)),
        track_new_thresh=float(d.get("track_new_thresh", 0.15)),
        track_match_thresh=float(d.get("track_match_thresh", 0.7)),
        track_buffer_frames=int(d.get("track_buffer_frames", 40)),
        last_updated_utc=float(d.get("last_updated_utc", 0.0)),
    )
    return s


def _settings_from_env(
    *,
    last_updated_utc: float | None = None,
    use_env: bool = True,
) -> Settings:
    """
    Build Settings using environment variables as defaults.
    This is used when no config file exists or when a fallback is needed.
    """
    s = Settings()
    if use_env:
        s = s.with_env_overrides()
    if last_updated_utc is not None:
        s.last_updated_utc = float(last_updated_utc)
    return s


def _apply_env_overrides_if_needed(
    s: Settings,
    *,
    apply_env_overrides: bool,
    env_applied: bool,
) -> Settings:
    if apply_env_overrides and not env_applied:
        return s.with_env_overrides()
    return s


def _seed_settings(bootstrap_from_env: bool) -> tuple[Settings, bool]:
    env_applied = bootstrap_from_env
    s = _settings_from_env(last_updated_utc=time.time(), use_env=env_applied)
    return s, env_applied


def load_settings(*, apply_env_overrides: bool = True, bootstrap_from_env: bool = True) -> Settings:
    """
    Load the persisted Settings from config.json, optionally applying environment overrides.

    Parameters
    ----------
    apply_env_overrides:
        Controls whether environment variables override values loaded from the persisted config file.
        When True (default), any relevant HBMON_* environment variables are applied on top of the
        loaded Settings. This is appropriate for the web UI and other components that should respect
        operator-level environment overrides even if the user changes settings via the UI.

        When False, the returned Settings reflect only the persisted configuration (plus any
        seeding behavior controlled by ``bootstrap_from_env``). Use this for runtime components
        (e.g., workers) that need to pick up user-configured changes written by the web UI without
        having those values masked by current environment variables.

    bootstrap_from_env:
        Controls how a fresh Settings instance is bootstrapped when no valid config file is
        available (missing or corrupted). When True (default), environment variables are used as
        the initial source of values for a new Settings object, which is then persisted to
        config.json. When False, a new Settings object is created using only code defaults, ignoring
        environment variables during this initial seeding step.
    """
    ensure_dirs()
    p = config_path()
    if not p.exists():
        s, env_applied = _seed_settings(bootstrap_from_env)
        save_settings(s)
        return _apply_env_overrides_if_needed(
            s,
            apply_env_overrides=apply_env_overrides,
            env_applied=env_applied,
        )

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("config.json root is not an object")
        s = _settings_from_dict(data)
    except Exception:
        s, env_applied = _seed_settings(bootstrap_from_env)
        return _apply_env_overrides_if_needed(
            s,
            apply_env_overrides=apply_env_overrides,
            env_applied=env_applied,
        )

    if apply_env_overrides:
        s = s.with_env_overrides()
    return s


def save_settings(s: Settings) -> None:
    ensure_dirs()
    p = config_path()

    out: dict[str, Any] = asdict(s)
    out["last_updated_utc"] = time.time()

    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def roi_to_str(roi: Roi | None) -> str:
    if roi is None:
        return ""
    x1, y1, x2, y2 = roi.clamp().as_tuple()
    return f"{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}"
