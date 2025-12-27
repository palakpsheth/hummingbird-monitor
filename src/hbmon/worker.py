# src/hbmon/worker.py
"""
RTSP worker: detect hummingbird visits, record snapshot + short clip,
classify species (CLIP) and assign individual (embedding re-ID), then write to DB.

High-level pipeline per loop:
- Read frame from RTSP
- Apply ROI crop (if configured)
- Run YOLO (ultralytics) to detect "bird" class (COCO 'bird' class)
- Pick best detection (largest area)
- If not in cooldown: save snapshots (raw, annotated, CLIP crop) + record clip
- Crop around bbox (with configurable padding) for CLIP classification + embedding
- Match embedding to an Individual prototype (cosine distance threshold)
- Insert Observation + (optional) Embedding, update Individual stats/prototype

This is CPU-friendly by default. You can tune via config.json or env vars.

Expected env vars (common):
- HBMON_RTSP_URL=rtsp://...
- HBMON_CAMERA_NAME=hummingbirdcam
- HBMON_FPS_LIMIT=8
- HBMON_CLIP_SECONDS=2.0
- HBMON_CROP_PADDING=0.05 (padding fraction around bird bbox for CLIP; lower = tighter crop)
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select  # type: ignore

from hbmon.clip_model import ClipModel
from hbmon.clustering import l2_normalize, update_prototype_ema, cosine_distance
from hbmon.config import Settings, background_image_path, ensure_dirs, load_settings, snapshots_dir
from hbmon.db import async_session_scope, init_async_db
from hbmon.models import Embedding, Individual, Observation

# ---------------------------------------------------------------------------
# Optional heavy dependencies
# ---------------------------------------------------------------------------
# opencv-python and ultralytics are optional; wrap imports so the module can
# still be imported without them.  When missing, functions that rely on
# these packages will raise at runtime.
try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    _CV2_AVAILABLE = False

try:
    from ultralytics import YOLO  # type: ignore
    _YOLO_AVAILABLE = True
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore
    _YOLO_AVAILABLE = False


# Default COCO class id for 'bird'. We attempt to resolve this from the loaded model
# class names at runtime, but fall back to this value (or HBMON_BIRD_CLASS_ID) if needed.
DEFAULT_BIRD_CLASS_ID = 14


@dataclass
class Det:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


@dataclass(frozen=True)
class ObservationMediaPaths:
    """Relative media paths for a single observation."""

    observation_uuid: str
    snapshot_rel: str
    snapshot_annotated_rel: str
    snapshot_clip_rel: str
    clip_rel: str


def _build_observation_media_paths(stamp: str, observation_uuid: str | None = None) -> ObservationMediaPaths:
    """Build the observation media paths with a shared UUID."""

    obs_uuid = observation_uuid or uuid.uuid4().hex
    return ObservationMediaPaths(
        observation_uuid=obs_uuid,
        snapshot_rel=f"snapshots/{stamp}/{obs_uuid}.jpg",
        snapshot_annotated_rel=f"snapshots/{stamp}/{obs_uuid}_annotated.jpg",
        snapshot_clip_rel=f"snapshots/{stamp}/{obs_uuid}_clip.jpg",
        clip_rel=f"clips/{stamp}/{obs_uuid}.mp4",
    )


def _build_observation_extra_data(
    *,
    observation_uuid: str,
    sensitivity: dict[str, Any],
    detection: dict[str, Any],
    identification: dict[str, Any],
    snapshots: dict[str, Any],
    review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the extra metadata payload for an observation."""

    return {
        "observation_uuid": observation_uuid,
        "sensitivity": sensitivity,
        "detection": detection,
        "identification": identification,
        "snapshots": snapshots,
        "review": review if review is not None else {"label": None},
    }


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _apply_roi(frame_bgr: np.ndarray, s: Settings) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Return (roi_frame, (x_off, y_off)) where offsets map ROI coords to original.
    """
    if s.roi is None:
        return frame_bgr, (0, 0)

    h, w = frame_bgr.shape[:2]
    r = s.roi.clamp()
    x1 = int(r.x1 * w)
    y1 = int(r.y1 * h)
    x2 = int(r.x2 * w)
    y2 = int(r.y2 * h)

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(1, min(w, x2))
    y2 = max(1, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return frame_bgr, (0, 0)

    return frame_bgr[y1:y2, x1:x2], (x1, y1)


def _load_background_image() -> np.ndarray | None:
    """
    Load the configured background image from disk.

    Returns the image as a BGR numpy array, or None if not configured or not available.
    """
    if not _CV2_AVAILABLE:
        return None

    bg_path = background_image_path()
    if not bg_path.exists():
        return None

    try:
        img = cv2.imread(str(bg_path))
        if img is None:
            print(f"[worker] Failed to load background image: {bg_path}")
            return None
        print(f"[worker] Loaded background image: {bg_path} shape={img.shape}")
        return img
    except Exception as e:
        print(f"[worker] Error loading background image: {e}")
        return None


def _sanitize_bg_params(
    *,
    enabled: bool,
    threshold: int,
    blur: int,
    min_overlap: float,
) -> tuple[bool, int, int, float]:
    threshold_safe = max(0, min(255, int(threshold)))
    blur_safe = max(1, int(blur))
    if blur_safe % 2 == 0:
        blur_safe += 1
    overlap_safe = float(min_overlap)
    if overlap_safe < 0.0:
        overlap_safe = 0.0
    elif overlap_safe > 1.0:
        overlap_safe = 1.0
    return enabled, threshold_safe, blur_safe, overlap_safe


def _compute_motion_mask(
    frame: np.ndarray,
    background: np.ndarray,
    *,
    threshold: int = 30,
    blur_size: int = 5,
) -> np.ndarray:
    """
    Compute a binary motion mask using background subtraction.

    Compares the current frame with the reference background image to detect
    areas with significant change (motion/new objects).

    Parameters:
    - frame: Current BGR frame from the camera
    - background: Reference background BGR image
    - threshold: Pixel difference threshold (0-255) for detecting change
    - blur_size: Gaussian blur kernel size for noise reduction

    Returns:
    - Binary mask (255 where motion detected, 0 elsewhere)
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError("OpenCV required for motion detection")

    assert cv2 is not None

    # Resize background to match frame if needed
    if frame.shape[:2] != background.shape[:2]:
        background = cv2.resize(background, (frame.shape[1], frame.shape[0]))

    # Convert to grayscale for comparison
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    if blur_size > 0:
        # GaussianBlur requires odd kernel sizes; normalize even values to the next odd.
        kernel_size = blur_size | 1
        gray_frame = cv2.GaussianBlur(gray_frame, (kernel_size, kernel_size), 0)
        gray_bg = cv2.GaussianBlur(gray_bg, (kernel_size, kernel_size), 0)

    # Compute absolute difference
    diff = cv2.absdiff(gray_frame, gray_bg)

    # Apply threshold to create binary mask
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes

    return mask


def _detection_overlaps_motion(
    det: "Det",
    motion_mask: np.ndarray,
    *,
    min_overlap_ratio: float = 0.15,
) -> bool:
    """
    Check if a detection bounding box overlaps sufficiently with motion areas.

    Returns True if the detection should be kept (has motion), False if it's
    likely a false positive (no significant motion in that area).

    Parameters:
    - det: Detection bounding box
    - motion_mask: Binary mask from background subtraction
    - min_overlap_ratio: Minimum fraction of detection area that must have motion
    """
    h, w = motion_mask.shape[:2]

    # Clamp bbox to mask dimensions, ensuring x2 >= x1 and y2 >= y1
    x1 = max(0, min(det.x1, w - 1))
    y1 = max(0, min(det.y1, h - 1))
    x2 = max(x1 + 1, min(det.x2, w))
    y2 = max(y1 + 1, min(det.y2, h))

    if x2 <= x1 or y2 <= y1:
        return True  # Edge case: keep detection if bbox is invalid

    # Extract the region of the mask corresponding to the detection
    roi_mask = motion_mask[y1:y2, x1:x2]

    # Calculate the fraction of the detection area with motion
    motion_pixels = np.count_nonzero(roi_mask)
    total_pixels = roi_mask.size

    if total_pixels == 0:
        return True  # Edge case

    overlap_ratio = motion_pixels / total_pixels

    return overlap_ratio >= min_overlap_ratio


def _pick_best_bird_det(
    results: Any,
    min_box_area: int,
    bird_class_id: int,
    *,
    motion_mask: np.ndarray | None = None,
    min_motion_overlap: float = 0.15,
) -> Det | None:
    """
    Extract best detection from ultralytics results (largest area bird).

    If motion_mask is provided, detections without sufficient motion overlap
    are filtered out as likely false positives.
    """
    if not results:
        return None
    r0 = results[0]
    if r0.boxes is None:
        return None

    boxes = r0.boxes
    if len(boxes) == 0:
        return None

    best: Det | None = None
    for b in boxes:
        try:
            cls = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
            conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
            if cls != bird_class_id:
                continue

            xyxy = b.xyxy[0].detach().cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in xyxy.tolist()]
            d = Det(x1=x1, y1=y1, x2=x2, y2=y2, conf=conf)
            if d.area < min_box_area:
                continue

            # Filter by motion if mask is available
            if motion_mask is not None:
                if not _detection_overlaps_motion(d, motion_mask, min_overlap_ratio=min_motion_overlap):
                    continue

            if best is None or d.area > best.area:
                best = d
        except Exception:
            continue

    return best


def _write_jpeg(path: Path, frame_bgr: np.ndarray) -> None:
    """
    Write a JPEG image to disk.  Requires OpenCV; raises RuntimeError if
    OpenCV is unavailable.
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError("OpenCV (cv2) is not installed; cannot write JPEGs")
    assert cv2 is not None  # for type checkers
    _safe_mkdir(path.parent)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("cv2.imencode failed for jpeg")
    path.write_bytes(buf.tobytes())


def _draw_bbox(
    frame_bgr: np.ndarray,
    det: Det,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_confidence: bool = True,
) -> np.ndarray:
    """
    Draw a bounding box on the frame around the detected object with optional confidence label.

    Args:
        frame_bgr: BGR image as numpy array.
        det: Detection with bounding box coordinates.
        color: BGR color for the box (default: green).
        thickness: Line thickness in pixels (default: 2).
        show_confidence: If True, draw the detection confidence and bbox area above the box.

    Returns:
        A copy of the frame with the bounding box and optional label drawn.
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError("OpenCV (cv2) is not installed; cannot draw bounding boxes")
    assert cv2 is not None  # for type checkers

    annotated = frame_bgr.copy()
    cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), color, thickness)

    if show_confidence:
        label = _format_bbox_label(det)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        # Get text size to draw background rectangle
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Position text above the bounding box
        text_x = det.x1
        text_y = det.y1 - 6
        if text_y - text_h < 0:
            # If not enough space above, put it inside the box
            text_y = det.y1 + text_h + 6

        # Draw background rectangle for better readability
        cv2.rectangle(
            annotated,
            (text_x, text_y - text_h - 4),
            (text_x + text_w + 4, text_y + 4),
            color,
            -1,  # filled
        )

        # Draw text in black for contrast
        cv2.putText(annotated, label, (text_x + 2, text_y), font, font_scale, (0, 0, 0), font_thickness)

    return annotated


def _format_bbox_label(det: Det) -> str:
    """Format a label that includes detection confidence and bounding-box area."""
    return f"{det.conf:.2f} | {det.area}px^2"


def _bbox_area_ratio(det: Det, frame_shape: tuple[int, int]) -> float:
    """Return the fraction of the reference area occupied by the detection's bbox area."""
    h, w = frame_shape
    frame_area = max(h, 0) * max(w, 0)
    if frame_area <= 0:
        return 0.0
    return det.area / frame_area


def _ffmpeg_available() -> bool:
    """Check if FFmpeg is available on the system."""
    return shutil.which("ffmpeg") is not None


def _convert_to_h264(input_path: Path, output_path: Path, *, timeout: float = 60.0) -> bool:
    """
    Convert a video file to H.264/AAC MP4 using FFmpeg for browser compatibility.

    Uses libx264 with faststart for progressive download/streaming.
    Returns True if conversion succeeded, False otherwise.
    """
    if not _ffmpeg_available():
        print("[worker] FFmpeg not available, skipping H.264 conversion")
        return False

    try:
        cmd = [
            "ffmpeg",
            "-y",  # overwrite output
            "-i", str(input_path),
            "-c:v", "libx264",  # H.264 video codec
            "-preset", "fast",  # balance speed/quality
            "-crf", "23",  # quality (lower = better, 18-28 is typical)
            "-c:a", "aac",  # AAC audio codec (if any audio)
            "-movflags", "+faststart",  # Move moov atom to start for streaming
            "-an",  # No audio (RTSP streams often have no audio anyway)
            str(output_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            check=False,
        )

        if result.returncode == 0 and output_path.exists():
            return True
        else:
            stderr = result.stderr.decode("utf-8", errors="replace")[:500]
            print(f"[worker] FFmpeg conversion failed: {stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"[worker] FFmpeg conversion timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"[worker] FFmpeg conversion error: {e}")
        return False


def _record_clip_opencv(
    cap: cv2.VideoCapture,
    out_path: Path,
    seconds: float,
    *,
    max_fps: float = 20.0,
) -> Path:
    """
    Record a short clip from the existing VideoCapture, preferring AVC1 MP4 and
    falling back to MP4V or AVI. Returns the actual path used for the clip.

    After recording, if the codec used was not browser-compatible (e.g., mp4v),
    the clip is post-processed with FFmpeg to convert to H.264 for browser playback.

    This blocks while recording (simple + robust for early versions).
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError("OpenCV (cv2) is not installed; cannot record clips")
    assert cv2 is not None  # for type checkers
    _safe_mkdir(out_path.parent)

    # Read one frame to get size
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Unable to read frame for clip start")

    h, w = frame.shape[:2]

    # Prefer H.264/AVC for browser compatibility; fall back to MP4V then AVI.
    # Track which codec we actually use to decide if FFmpeg conversion is needed.
    writer = None
    final_path = out_path
    used_fourcc = ""
    needs_conversion = False

    for suffix, fourcc_str in [
        (".mp4", "avc1"),
        (".mp4", "H264"),
        (".mp4", "mp4v"),
        (".avi", "XVID"),
    ]:
        candidate_path = out_path.with_suffix(suffix)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        candidate = cv2.VideoWriter(str(candidate_path), fourcc, float(max_fps), (w, h))
        if not candidate.isOpened():
            candidate.release()
            continue

        writer = candidate
        final_path = candidate_path
        used_fourcc = fourcc_str
        # mp4v (MPEG-4 Part 2) and XVID are not browser-compatible
        needs_conversion = fourcc_str.lower() in ("mp4v", "xvid")
        break

    if writer is None:
        raise RuntimeError("Unable to open VideoWriter for clip")

    start = time.time()
    writer.write(frame)

    # Best-effort write frames for `seconds`
    while time.time() - start < seconds:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        writer.write(fr)

        # Throttle a bit (don't hammer CPU)
        time.sleep(max(0.0, (1.0 / max_fps) * 0.2))

    writer.release()

    # Post-process with FFmpeg if we used a non-browser-compatible codec
    if needs_conversion and final_path.exists():
        print(f"[worker] Converting {used_fourcc} clip to H.264 for browser compatibility")
        converted_path = final_path.with_stem(final_path.stem + "_h264").with_suffix(".mp4")

        if _convert_to_h264(final_path, converted_path):
            # Replace original with converted version
            # Use the original stem but with .mp4 extension for the final name
            target_path = final_path.parent / (final_path.stem + ".mp4")
            try:
                final_path.unlink()  # Remove original
                converted_path.rename(target_path)
                final_path = target_path
                print(f"[worker] Successfully converted to H.264: {final_path}")
            except Exception as e:
                print(f"[worker] Failed to replace original with converted: {e}")
                # Keep the converted file if rename failed
                if converted_path.exists():
                    final_path = converted_path
        else:
            print(f"[worker] FFmpeg conversion failed, keeping original {used_fourcc} file")
            # Clean up partial converted file if it exists
            if converted_path.exists():
                try:
                    converted_path.unlink()
                except Exception:
                    pass

    return final_path


def _bbox_with_padding(det: Det, frame_shape: tuple[int, int], pad_frac: float = 0.18) -> tuple[int, int, int, int]:
    """
    Expand bbox by pad_frac in each direction, clamped to frame.
    """
    h, w = frame_shape
    bw = det.x2 - det.x1
    bh = det.y2 - det.y1
    pad_x = int(np.ceil(bw * pad_frac))
    pad_y = int(np.ceil(bh * pad_frac))

    x1 = max(0, det.x1 - pad_x)
    y1 = max(0, det.y1 - pad_y)
    x2 = min(w, det.x2 + pad_x)
    y2 = min(h, det.y2 + pad_y)

    return x1, y1, x2, y2


async def _load_individuals_for_matching(db: Any) -> list[tuple[int, np.ndarray, int]]:
    """
    Returns [(id, prototype_vec, visit_count), ...] for individuals that have prototypes.
    """
    rows = (await db.execute(select(Individual))).scalars().all()
    out: list[tuple[int, np.ndarray, int]] = []
    for ind in rows:
        proto = ind.get_prototype()
        if proto is None:
            continue
        out.append((int(ind.id), l2_normalize(proto), int(ind.visit_count)))
    return out


async def _match_or_create_individual(
    db: Any,
    emb: np.ndarray,
    *,
    species_label: str | None,
    match_threshold: float,
    ema_alpha: float,
) -> tuple[int, float]:
    """
    Match embedding to existing Individual prototype, else create new Individual.

    Returns (individual_id, similarity_score)
    """
    emb = l2_normalize(emb)

    candidates = await _load_individuals_for_matching(db)
    if not candidates:
        ind = Individual(name="(unnamed)", visit_count=0, last_seen_at=None, last_species_label=species_label)
        ind.set_prototype(emb)
        ind.visit_count = 1
        ind.last_seen_at = utcnow()
        db.add(ind)
        await db.flush()
        return int(ind.id), 0.0

    best_id = None
    best_dist = 9.0
    best_visits = 0

    for iid, proto, visits in candidates:
        d = cosine_distance(emb, proto)
        if d < best_dist:
            best_dist = d
            best_id = iid
            best_visits = visits

    assert best_id is not None

    sim = float(1.0 - best_dist)

    if best_dist <= match_threshold:
        ind = await db.get(Individual, best_id)
        assert ind is not None
        proto = ind.get_prototype()
        if proto is None:
            ind.set_prototype(emb)
        else:
            # more conservative once there are many visits
            alpha_eff = ema_alpha if best_visits < 5 else min(ema_alpha, 0.05)
            new_proto = update_prototype_ema(l2_normalize(proto), emb, alpha=alpha_eff)
            ind.set_prototype(new_proto)

        ind.visit_count = int(ind.visit_count or 0) + 1
        ind.last_seen_at = utcnow()
        if species_label:
            ind.last_species_label = species_label

        return int(ind.id), sim

    # create new
    ind = Individual(name="(unnamed)", visit_count=0, last_seen_at=None, last_species_label=species_label)
    ind.set_prototype(emb)
    ind.visit_count = 1
    ind.last_seen_at = utcnow()
    db.add(ind)
    await db.flush()
    return int(ind.id), 0.0


async def run_worker() -> None:
    """
    Main loop for the hummingbird monitoring worker.  Requires OpenCV,
    ultralytics and the ClipModel dependencies.  If any of these are not
    available, a ``RuntimeError`` is raised.
    """
    # Force RTSP over TCP for better compatibility (fixes 461 Unsupported Transport)
    # Improve RTSP stability with OpenCV+FFmpeg
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|stimeout;5000000|max_delay;500000|fflags;nobuffer"
    )

    if not (_CV2_AVAILABLE and _YOLO_AVAILABLE):
        raise RuntimeError(
            "OpenCV and ultralytics must be installed to run the worker"
        )

    ensure_dirs()
    await init_async_db()

    # Load models once
    yolo_model_name = os.getenv("HBMON_YOLO_MODEL", "yolo11n.pt")
    yolo = YOLO(yolo_model_name)  # type: ignore[misc]

    # Resolve the class id for 'bird' from the model's names mapping when possible.
    # This keeps things robust if you later use custom-trained weights with different class ordering.
    bird_class_id: int | None = None
    try:
        names = getattr(yolo, 'names', None)
        if isinstance(names, dict):
            for k, v in names.items():
                if str(v).strip().lower() == 'bird':
                    bird_class_id = int(k)
                    break
        elif isinstance(names, (list, tuple)):
            for i, v in enumerate(names):
                if str(v).strip().lower() == 'bird':
                    bird_class_id = int(i)
                    break
    except Exception:
        bird_class_id = None

    if bird_class_id is None:
        bird_class_id = int(os.getenv('HBMON_BIRD_CLASS_ID', str(DEFAULT_BIRD_CLASS_ID)))
        print(f'[worker] Using bird_class_id={bird_class_id} (fallback)')
    else:
        print(f'[worker] Resolved bird_class_id={bird_class_id} from model names')

    clip = ClipModel(device=os.getenv("HBMON_DEVICE", "cpu"))
    env = os.getenv("HBMON_SPECIES_LIST", "").strip()
    if env:
        import re
        labels = [s.strip() for s in re.split(r",\s*", env) if s.strip()]
        if labels:
            clip.set_label_space(labels)

    # Load background image for motion-based filtering (if configured)
    background_img: np.ndarray | None = _load_background_image()
    last_background_check = time.time()

    def _parse_env_bool(value: str) -> bool:
        return value.strip().lower() in ("1", "true", "yes", "y", "on")

    bg_env_overrides: dict[str, object] = {}
    env_bg_enabled = os.getenv("HBMON_BG_SUBTRACTION")
    if env_bg_enabled is not None and env_bg_enabled.strip():
        bg_env_overrides["bg_subtraction_enabled"] = _parse_env_bool(env_bg_enabled)

    env_bg_threshold = os.getenv("HBMON_BG_MOTION_THRESHOLD")
    if env_bg_threshold is not None and env_bg_threshold.strip():
        try:
            bg_env_overrides["bg_motion_threshold"] = int(env_bg_threshold)
        except ValueError:
            print(f"[worker] Invalid HBMON_BG_MOTION_THRESHOLD={env_bg_threshold!r}; ignoring.")

    env_bg_blur = os.getenv("HBMON_BG_MOTION_BLUR")
    if env_bg_blur is not None and env_bg_blur.strip():
        try:
            bg_env_overrides["bg_motion_blur"] = int(env_bg_blur)
        except ValueError:
            print(f"[worker] Invalid HBMON_BG_MOTION_BLUR={env_bg_blur!r}; ignoring.")

    env_bg_overlap = os.getenv("HBMON_BG_MIN_OVERLAP")
    if env_bg_overlap is not None and env_bg_overlap.strip():
        try:
            bg_env_overrides["bg_min_overlap"] = float(env_bg_overlap)
        except ValueError:
            print(f"[worker] Invalid HBMON_BG_MIN_OVERLAP={env_bg_overlap!r}; ignoring.")

    cap: 'cv2.VideoCapture | None' = None  # type: ignore[name-defined]
    last_settings_load = 0.0
    settings: Settings | None = None
    last_bg_settings: tuple[bool, int, int, float] | None = None

    last_trigger = 0.0

    def get_settings() -> Settings:
        nonlocal last_settings_load, settings
        now = time.time()
        if settings is None or (now - last_settings_load) > 3.0:
            settings = load_settings(apply_env_overrides=False)

            # Always honor operational env overrides for RTSP URL and camera name,
            # even when user-tunable settings (thresholds, ROI) come from config.json.
            env_rtsp = os.getenv("HBMON_RTSP_URL")
            if env_rtsp:
                if settings.rtsp_url != env_rtsp:
                    print(f"[worker] Overriding rtsp_url from env: {env_rtsp}")
                settings.rtsp_url = env_rtsp

            env_camera = os.getenv("HBMON_CAMERA_NAME")
            if env_camera:
                if getattr(settings, "camera_name", None) != env_camera:
                    print(f"[worker] Overriding camera_name from env: {env_camera}")
                settings.camera_name = env_camera  # type: ignore[attr-defined]

            if bg_env_overrides:
                if "bg_subtraction_enabled" in bg_env_overrides:
                    settings.bg_subtraction_enabled = bool(bg_env_overrides["bg_subtraction_enabled"])
                if "bg_motion_threshold" in bg_env_overrides:
                    settings.bg_motion_threshold = int(bg_env_overrides["bg_motion_threshold"])
                if "bg_motion_blur" in bg_env_overrides:
                    settings.bg_motion_blur = int(bg_env_overrides["bg_motion_blur"])
                if "bg_min_overlap" in bg_env_overrides:
                    settings.bg_min_overlap = float(bg_env_overrides["bg_min_overlap"])
            last_settings_load = now
        return settings

    while True:
        s = get_settings()
        if not s.rtsp_url:
            print("[worker] HBMON_RTSP_URL not set. Sleeping...")
            time.sleep(2.0)
            continue

        # Ensure capture
        if cap is None or not cap.isOpened():
            print(f"[worker] Opening RTSP: {s.rtsp_url}")
            cap = cv2.VideoCapture(s.rtsp_url)
            # Small buffer helps reduce lag
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            except Exception:
                pass

            # Give it a moment
            time.sleep(0.5)

        ok, frame = cap.read()
        if not ok or frame is None:
            print("[worker] Frame read failed; reconnecting in 1s...")
            try:
                cap.release()
            except Exception:
                pass
            cap = None
            time.sleep(1.0)
            continue

        # --- Debug: prove we are receiving frames even if YOLO never triggers ---
        debug_every = float(os.getenv("HBMON_DEBUG_EVERY_SECONDS", "10"))
        debug_save = os.getenv("HBMON_DEBUG_SAVE_FRAMES", "0") == "1"

        if not hasattr(run_worker, "_last_debug"):
            run_worker._last_debug = 0.0  # type: ignore[attr-defined]

        now_dbg = time.time()
        if now_dbg - run_worker._last_debug > debug_every:  # type: ignore[attr-defined]
            run_worker._last_debug = now_dbg  # type: ignore[attr-defined]
            print(f"[worker] alive frame_shape={frame.shape} rtsp={s.rtsp_url}")

            if debug_save:
                from hbmon.config import media_dir
                p = media_dir() / "debug_latest.jpg"
                _write_jpeg(p, frame)
                print(f"[worker] wrote debug frame {p}")

        # Throttle overall loop (CPU friendly)
        if s.fps_limit and s.fps_limit > 0:
            time.sleep(max(0.0, 1.0 / float(s.fps_limit)))

        # Periodically check for background image updates (every 30 seconds)
        now_bg = time.time()
        if now_bg - last_background_check > 30.0:
            last_background_check = now_bg
            new_bg = _load_background_image()
            if new_bg is not None and background_img is None:
                print("[worker] Background image now available, enabling motion filtering")
                background_img = new_bg
            elif new_bg is None and background_img is not None:
                print("[worker] Background image removed, disabling motion filtering")
                background_img = None
            elif new_bg is not None:
                background_img = new_bg

        bg_enabled, bg_motion_threshold, bg_motion_blur, bg_min_overlap = _sanitize_bg_params(
            enabled=bool(s.bg_subtraction_enabled),
            threshold=int(s.bg_motion_threshold),
            blur=int(s.bg_motion_blur),
            min_overlap=float(s.bg_min_overlap),
        )
        bg_settings = (bg_enabled, bg_motion_threshold, bg_motion_blur, bg_min_overlap)
        if bg_settings != last_bg_settings:
            if background_img is not None and bg_enabled:
                print(f"[worker] Background subtraction enabled: threshold={bg_motion_threshold}, "
                      f"blur={bg_motion_blur}, min_overlap={bg_min_overlap}")
            elif not bg_enabled:
                print("[worker] Background subtraction disabled via config/env.")
            last_bg_settings = bg_settings

        roi_frame, (xoff, yoff) = _apply_roi(frame, s)

        # Compute motion mask for background subtraction (if enabled and background available)
        motion_mask: np.ndarray | None = None
        if bg_enabled and background_img is not None:
            try:
                # Apply same ROI to background image
                bg_roi, _ = _apply_roi(background_img, s)
                motion_mask = _compute_motion_mask(
                    roi_frame,
                    bg_roi,
                    threshold=bg_motion_threshold,
                    blur_size=bg_motion_blur,
                )
            except Exception as e:
                if os.getenv("HBMON_DEBUG_BG", "0") == "1":
                    print(f"[worker] Motion mask error: {e}")
                motion_mask = None

        # YOLO detect birds
        try:
            imgsz = int(os.getenv("HBMON_YOLO_IMGSZ", "1280"))
            results = yolo.predict(
                roi_frame,
                conf=float(s.detect_conf),
                iou=float(s.detect_iou),
                classes=[bird_class_id],
                imgsz=imgsz,
                verbose=False,
            )
        except Exception as e:
            print(f"[worker] YOLO error: {e}")
            time.sleep(0.5)
            continue

        if os.getenv("HBMON_DEBUG_YOLO", "0") == "1":
            r0 = results[0]
            n = 0 if r0.boxes is None else len(r0.boxes)
            print(f"[worker] yolo boxes={n}")

        det = _pick_best_bird_det(
            results,
            int(s.min_box_area),
            bird_class_id,
            motion_mask=motion_mask,
            min_motion_overlap=bg_min_overlap,
        )
        if det is None:
            continue

        # cooldown to avoid repeated triggers for the same visit
        now = time.time()
        if (now - last_trigger) < float(s.cooldown_seconds):
            continue

        # Translate ROI bbox to full-frame coords
        det_full = Det(
            x1=det.x1 + xoff,
            y1=det.y1 + yoff,
            x2=det.x2 + xoff,
            y2=det.y2 + yoff,
            conf=det.conf,
        )

        # Snapshot paths (relative under /media)
        # We save two images: raw (original) and annotated (with bbox + confidence)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        media_paths = _build_observation_media_paths(stamp)
        snap_id = media_paths.observation_uuid
        snap_rel = media_paths.snapshot_rel
        snap_annotated_rel = media_paths.snapshot_annotated_rel
        snap_clip_rel = media_paths.snapshot_clip_rel
        clip_rel = media_paths.clip_rel

        media_root = snapshots_dir().parent  # /media
        snap_path = media_root / snap_rel
        snap_annotated_path = media_root / snap_annotated_rel
        snap_clip_path = media_root / snap_clip_rel
        clip_path = media_root / clip_rel

        # Save both raw and annotated snapshots
        try:
            # Save raw image first
            _write_jpeg(snap_path, frame)
            # Save annotated image with bbox and confidence
            annotated_frame = _draw_bbox(frame, det_full, show_confidence=True)
            _write_jpeg(snap_annotated_path, annotated_frame)
        except Exception as e:
            print(f"[worker] snapshot write failed: {e}")
            # Clean up any partially written snapshot files to avoid orphans
            for path in (snap_path, snap_annotated_path):
                try:
                    if path.exists():
                        path.unlink()
                except Exception as cleanup_err:
                    print(f"[worker] snapshot cleanup failed for {path}: {cleanup_err}")
            continue

        # Record clip (best effort)
        try:
            recorded_path = _record_clip_opencv(cap, clip_path, float(s.clip_seconds), max_fps=20.0)
            clip_rel = str(recorded_path.relative_to(media_root))
        except Exception as e:
            print(f"[worker] clip record failed: {e}")
            # allow observation without clip (use snapshot only)
            # keep video_path pointing to a non-existent file? better to keep empty placeholder
            clip_rel = ""

        # Crop around bbox for CLIP
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = _bbox_with_padding(det_full, (h, w), pad_frac=float(s.crop_padding))
        crop = frame[y1:y2, x1:x2].copy()
        clip_snapshot_rel = ""
        try:
            _write_jpeg(snap_clip_path, crop)
            clip_snapshot_rel = snap_clip_rel
        except Exception as e:
            print(f"[worker] clip snapshot write failed: {e}")

        # Species + embedding
        try:
            raw_species_label, raw_species_prob = clip.predict_species_label_prob(crop)
            emb = clip.encode_embedding(crop)
        except Exception as e:
            print(f"[worker] CLIP error: {e}")
            raw_species_label, raw_species_prob = "Hummingbird (unknown species)", 0.0
            emb = None

        species_label = raw_species_label
        species_prob = float(raw_species_prob)
        if species_prob < float(s.min_species_prob):
            species_label = "Hummingbird (unknown species)"

        # Write DB
        async with async_session_scope() as db:
            individual_id = None
            match_score = 0.0

            if emb is not None:
                individual_id, match_score = await _match_or_create_individual(
                    db,
                    emb,
                    species_label=species_label,
                    match_threshold=float(s.match_threshold),
                    ema_alpha=float(s.ema_alpha),
                )

            bg_active = bool(bg_enabled and background_img is not None)
            snapshots_data = {"annotated_path": snap_annotated_rel}
            if clip_snapshot_rel:
                snapshots_data["clip_path"] = clip_snapshot_rel

            sensitivity_data = {
                "detect_conf": float(s.detect_conf),
                "detect_iou": float(s.detect_iou),
                "min_box_area": int(s.min_box_area),
                "cooldown_seconds": float(s.cooldown_seconds),
                "min_species_prob": float(s.min_species_prob),
                "match_threshold": float(s.match_threshold),
                "ema_alpha": float(s.ema_alpha),
                "crop_padding": float(s.crop_padding),
                "bg_motion_threshold": int(bg_motion_threshold),
                "bg_motion_blur": int(bg_motion_blur),
                "bg_min_overlap": float(bg_min_overlap),
                "bg_subtraction_enabled": bg_active,
                "bg_subtraction_configured": bool(bg_enabled),
                "background_image_available": bool(background_img is not None),
            }
            detection_data = {
                "box_confidence": float(det_full.conf),
                "bbox_xyxy": [int(det_full.x1), int(det_full.y1), int(det_full.x2), int(det_full.y2)],
                "bbox_area": int(det_full.area),
                "bbox_area_ratio": float(_bbox_area_ratio(det_full, (h, w))),
                "bbox_area_ratio_frame": float(_bbox_area_ratio(det_full, (h, w))),
                "bbox_area_ratio_roi": float(_bbox_area_ratio(det, roi_frame.shape[:2])),
                "roi_offset_xy": [int(xoff), int(yoff)],
                "background_subtraction_enabled": bg_active,
                "nms_iou_threshold": float(s.detect_iou),
            }
            identification_data = {
                "individual_id": individual_id,
                "match_score": float(match_score),
                "species_label": raw_species_label,
                "species_prob": float(raw_species_prob),
                "species_label_final": species_label,
                "species_accepted": species_prob >= float(s.min_species_prob),
            }
            extra_data = _build_observation_extra_data(
                observation_uuid=snap_id,
                sensitivity=sensitivity_data,
                detection=detection_data,
                identification=identification_data,
                snapshots=snapshots_data,
            )

            obs = Observation(
                ts=utcnow(),
                camera_name=s.camera_name,
                species_label=species_label,
                species_prob=float(species_prob),
                individual_id=individual_id,
                match_score=float(match_score),
                bbox_x1=int(det_full.x1),
                bbox_y1=int(det_full.y1),
                bbox_x2=int(det_full.x2),
                bbox_y2=int(det_full.y2),
                snapshot_path=snap_rel,
                video_path=clip_rel if clip_rel else "clips/none.mp4",
                extra_json=None,
            )
            obs.set_extra(extra_data)
            db.add(obs)
            await db.flush()  # get obs.id

            if emb is not None:
                e = Embedding(observation_id=int(obs.id), individual_id=individual_id)
                e.set_vec(emb)
                db.add(e)

        last_trigger = time.time()

        print(
            f"[worker] {utcnow().isoformat(timespec='seconds')} "
            f"species={species_label} p={species_prob:.2f} "
            f"ind={individual_id} sim={match_score:.3f} "
            f"bbox=({det_full.x1},{det_full.y1},{det_full.x2},{det_full.y2})"
        )


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
