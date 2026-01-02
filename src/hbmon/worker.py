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
- HBMON_TEMPORAL_WINDOW_FRAMES=5
- HBMON_ARRIVAL_BUFFER_SECONDS=5.0
- HBMON_DEPARTURE_TIMEOUT_SECONDS=2.0
- HBMON_POST_DEPARTURE_BUFFER_SECONDS=3.0
- HBMON_CROP_PADDING=0.05 (padding fraction around bird bbox for CLIP; lower = tighter crop)

Background subtraction extras:
- HBMON_BG_LOG_REJECTED=0 (set to 1 to log motion-rejected candidates)
- HBMON_BG_REJECTED_COOLDOWN_SECONDS=3
- HBMON_BG_REJECTED_SAVE_CLIP=0
- HBMON_BG_REJECTED_MAX_PER_MINUTE=30 (optional cap)
- HBMON_BG_SAVE_MASKS=1
- HBMON_BG_SAVE_MASK_OVERLAY=1
- HBMON_BG_MASK_FORMAT=png
- HBMON_BG_MASK_DOWNSCALE_MAX=0 (0 disables downscale)
"""

from __future__ import annotations

import logging

# Configure logging to show INFO messages (e.g. from ClipModel)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


import asyncio
from collections import deque
from enum import Enum
import uuid
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select  # type: ignore

from hbmon.clip_model import ClipModel
from hbmon.clustering import l2_normalize, update_prototype_ema, cosine_distance
from hbmon.config import (
    Settings,
    background_image_path,
    ensure_dirs,
    env_bool,
    env_int,
    load_settings,
    media_dir,
)
from hbmon.db import async_session_scope, init_async_db
from hbmon.models import Candidate, Embedding, Individual, Observation
from hbmon.observation_tools import extract_video_metadata
from hbmon.yolo_utils import resolve_predict_imgsz

# ---------------------------------------------------------------------------
# Optional heavy dependencies
# ---------------------------------------------------------------------------
# opencv-python and ultralytics are optional; wrap imports so the module can
# still be imported without them.  When missing, functions that rely on
# these packages will raise at runtime.
try:
    import cv2  # type: ignore
    from .recorder import BackgroundRecorder
    _CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    BackgroundRecorder = None  # type: ignore
    _CV2_AVAILABLE = False

try:
    from ultralytics import YOLO  # type: ignore
    _YOLO_AVAILABLE = True
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore
    _YOLO_AVAILABLE = False

try:
    from hbmon.openvino_utils import is_openvino_available, validate_openvino_gpu, force_openvino_gpu_override
    _OPENVINO_UTILS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OPENVINO_UTILS_AVAILABLE = False

    def is_openvino_available() -> bool:
        return False

    def validate_openvino_gpu() -> bool:
        return False

    def force_openvino_gpu_override() -> None:
        pass

class VisitState(Enum):
    IDLE = 0
    RECORDING = 1
    FINALIZING = 2


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
    snapshot_roi_rel: str
    snapshot_background_rel: str
    clip_rel: str


@dataclass(frozen=True)
class CandidateMediaPaths:
    """Relative media paths for a single candidate."""

    candidate_uuid: str
    snapshot_rel: str
    snapshot_annotated_rel: str
    clip_rel: str
    mask_rel: str
    mask_overlay_rel: str


@dataclass
class CandidateItem:
    """Item sent from producer to consumer queue."""
    frame: np.ndarray
    det_full: Det
    det: Det | None = None
    timestamp: float = 0.0
    motion_mask: np.ndarray | None = None
    background_img: np.ndarray | None = None
    settings_snapshot: dict[str, Any] | None = None
    roi_stats: dict[str, Any] | None = None
    bbox_stats: dict[str, Any] | None = None
    bg_active: bool = False
    s: Settings | None = None
    is_rejected: bool = False
    rejected_reason: str | None = None
    video_path: Path | None = None
    model_metadata: dict[str, Any] | None = None  # YOLO and CLIP model information


@dataclass
class FrameEntry:
    """Entry in the temporal voting buffer for multi-frame detection."""
    frame: np.ndarray
    roi_frame: np.ndarray
    timestamp: float
    detections: list[Det]
    motion_mask: np.ndarray | None
    xoff: int
    yoff: int


def _build_observation_media_paths(stamp: str, observation_uuid: str | None = None) -> ObservationMediaPaths:
    """Build the observation media paths with a shared UUID."""

    obs_uuid = observation_uuid or uuid.uuid4().hex
    return ObservationMediaPaths(
        observation_uuid=obs_uuid,
        snapshot_rel=f"snapshots/{stamp}/{obs_uuid}.jpg",
        snapshot_annotated_rel=f"snapshots/{stamp}/{obs_uuid}_annotated.jpg",
        snapshot_clip_rel=f"snapshots/{stamp}/{obs_uuid}_clip.jpg",
        snapshot_roi_rel=f"snapshots/{stamp}/{obs_uuid}_roi.jpg",
        snapshot_background_rel=f"snapshots/{stamp}/{obs_uuid}_background.jpg",
        clip_rel=f"clips/{stamp}/{obs_uuid}.mp4",
    )


def _build_candidate_media_paths(
    stamp: str,
    candidate_uuid: str | None = None,
    *,
    mask_ext: str = "png",
) -> CandidateMediaPaths:
    cand_uuid = candidate_uuid or uuid.uuid4().hex
    return CandidateMediaPaths(
        candidate_uuid=cand_uuid,
        snapshot_rel=f"snapshots/candidates/{stamp}/{cand_uuid}.jpg",
        snapshot_annotated_rel=f"snapshots/candidates/{stamp}/{cand_uuid}_ann.jpg",
        clip_rel=f"clips/candidates/{stamp}/{cand_uuid}.mp4",
        mask_rel=f"masks/candidates/{stamp}/{cand_uuid}_mask.{mask_ext}",
        mask_overlay_rel=f"masks/candidates/{stamp}/{cand_uuid}_overlay.{mask_ext}",
    )


def _build_mask_paths(stamp: str, observation_uuid: str, *, mask_ext: str = "png") -> tuple[str, str]:
    return (
        f"masks/observations/{stamp}/{observation_uuid}_mask.{mask_ext}",
        f"masks/observations/{stamp}/{observation_uuid}_overlay.{mask_ext}",
    )


def _build_observation_extra_data(
    *,
    observation_uuid: str,
    sensitivity: dict[str, Any],
    detection: dict[str, Any],
    identification: dict[str, Any],
    snapshots: dict[str, Any],
    models: dict[str, Any] | None = None,
    video: dict[str, Any] | None = None,
    review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the extra metadata payload for an observation."""

    data: dict[str, Any] = {
        "observation_uuid": observation_uuid,
        "sensitivity": sensitivity,
        "detection": detection,
        "identification": identification,
        "snapshots": snapshots,
        "review": review if review is not None else {"label": None},
    }
    if models is not None:
        data["models"] = models
    if video is not None:
        data["video"] = video
    return data


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


def _motion_overlap_stats(det: "Det", motion_mask: np.ndarray) -> dict[str, float | int]:
    """
    Compute motion overlap stats within the detection bounding box (ROI coords).

    Returns a dict with motion pixel counts and overlap ratio.
    """
    h, w = motion_mask.shape[:2]

    x1 = max(0, min(det.x1, w - 1))
    y1 = max(0, min(det.y1, h - 1))
    x2 = max(x1 + 1, min(det.x2, w))
    y2 = max(y1 + 1, min(det.y2, h))

    if x2 <= x1 or y2 <= y1:
        return {
            "bbox_motion_pixels": 0,
            "bbox_total_pixels": 0,
            "bbox_overlap_ratio": 0.0,
        }

    roi_mask = motion_mask[y1:y2, x1:x2]
    motion_pixels = int(np.count_nonzero(roi_mask))
    total_pixels = int(roi_mask.size)
    overlap_ratio = float(motion_pixels) / max(total_pixels, 1)
    return {
        "bbox_motion_pixels": motion_pixels,
        "bbox_total_pixels": total_pixels,
        "bbox_overlap_ratio": overlap_ratio,
    }


def _roi_motion_stats(motion_mask: np.ndarray) -> dict[str, float | int]:
    """
    Compute motion coverage stats for the full ROI mask.
    """
    motion_pixels = int(np.count_nonzero(motion_mask))
    total_pixels = int(motion_mask.size)
    motion_fraction = float(motion_pixels) / max(total_pixels, 1)
    return {
        "roi_motion_pixels": motion_pixels,
        "roi_total_pixels": total_pixels,
        "roi_motion_fraction": motion_fraction,
    }


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
    stats = _motion_overlap_stats(det, motion_mask)
    if stats["bbox_total_pixels"] == 0:
        return True
    return float(stats["bbox_overlap_ratio"]) >= min_overlap_ratio


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


def _collect_bird_detections(
    results: Any,
    min_box_area: int,
    bird_class_id: int,
) -> list[Det]:
    """
    Collect bird detections from ultralytics results, filtering by min box area.
    """
    if not results:
        return []
    r0 = results[0]
    if r0.boxes is None:
        return []

    boxes = r0.boxes
    if len(boxes) == 0:
        return []

    detections: list[Det] = []
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
            detections.append(d)
        except Exception:
            continue

    return detections


def _select_best_detection(entries: list[tuple[Det, dict[str, float | int]]]) -> tuple[Det, dict[str, float | int]]:
    return max(entries, key=lambda item: (item[0].area, item[0].conf))


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


async def _write_jpeg_async(path: Path, frame_bgr: np.ndarray) -> None:
    """
    Async wrapper for _write_jpeg to enable parallel I/O operations.
    """
    await asyncio.to_thread(_write_jpeg, path, frame_bgr)


def _write_png(path: Path, image: np.ndarray) -> None:
    """
    Write a PNG image to disk.  Requires OpenCV; raises RuntimeError if
    OpenCV is unavailable.
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError("OpenCV (cv2) is not installed; cannot write PNGs")
    assert cv2 is not None  # for type checkers
    _safe_mkdir(path.parent)
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("cv2.imencode failed for png")
    path.write_bytes(buf.tobytes())


def _downscale_shape(height: int, width: int, max_dim: int) -> tuple[int, int] | None:
    if max_dim <= 0:
        return None
    largest = max(height, width)
    if largest <= max_dim:
        return None
    scale = max_dim / float(largest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return new_h, new_w


def _save_motion_mask_images(
    *,
    motion_mask: np.ndarray,
    roi_frame: np.ndarray,
    mask_path: Path,
    overlay_path: Path | None,
    downscale_max: int,
) -> None:
    """
    Save a binary motion mask and optional overlay visualization.
    """
    if not _CV2_AVAILABLE:
        return
    assert cv2 is not None

    mask_uint8 = (motion_mask > 0).astype(np.uint8) * 255
    h, w = mask_uint8.shape[:2]
    target = _downscale_shape(h, w, downscale_max)
    if target:
        target_h, target_w = target
        mask_uint8 = cv2.resize(mask_uint8, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        roi_frame = cv2.resize(roi_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    _write_png(mask_path, mask_uint8)
    if overlay_path is not None:
        overlay = roi_frame.copy()
        color_mask = np.zeros_like(overlay)
        color_mask[:, :, 2] = mask_uint8
        overlay = cv2.addWeighted(overlay, 0.75, color_mask, 0.35, 0)
        _write_png(overlay_path, overlay)
    return


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


def _draw_text_lines(
    frame_bgr: np.ndarray,
    lines: list[str],
    *,
    origin: tuple[int, int] = (12, 24),
    color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """
    Draw multiple text lines at the given origin.
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError("OpenCV (cv2) is not installed; cannot draw text")
    assert cv2 is not None

    annotated = frame_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    x, y = origin
    line_height = 22
    for line in lines:
        (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        cv2.rectangle(
            annotated,
            (x - 4, y - text_h - 6),
            (x + text_w + 4, y + 4),
            color,
            -1,
        )
        cv2.putText(annotated, line, (x, y), font, font_scale, (0, 0, 0), font_thickness)
        y += line_height
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


async def _prepare_crop_and_clip(
    frame: np.ndarray,
    det: Det,
    crop_padding: float,
    snap_clip_path: Path,
    clip: ClipModel | None,
) -> tuple[np.ndarray, tuple[str | None, float], np.ndarray | None]:
    """
    Crop the detection and run CLIP inference for species and embedding.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _bbox_with_padding(det, (h, w), pad_frac=crop_padding)
    crop = frame[y1:y2, x1:x2].copy()
    await _write_jpeg_async(snap_clip_path, crop)

    if clip is None:
        return crop, (None, 0.0), None

    # Run sequentially to avoid OpenVINO "Infer Request is busy" errors
    # Simultaneous calls to the same compiled model on GPU can cause race conditions
    (species_label, species_prob) = await asyncio.to_thread(clip.predict_species_label_prob, crop)
    emb = await asyncio.to_thread(clip.encode_embedding, crop)
    
    return crop, (species_label, float(species_prob)), emb


async def process_candidate_task(item: CandidateItem, clip: ClipModel, media_root: Path, sem: asyncio.Semaphore) -> None:
    """
    Background task to process a single bird candidate.
    """
    try:
        frame = item.frame
        det_full = item.det_full
        motion_mask = item.motion_mask
        background_img = item.background_img
        s = item.s
        if s is None:
            return

        save_masks = env_bool("HBMON_BG_SAVE_MASKS", True)
        save_mask_overlay = env_bool("HBMON_BG_SAVE_MASK_OVERLAY", True)
        mask_format = os.getenv("HBMON_BG_MASK_FORMAT", "png")
        mask_downscale_max = int(os.getenv("HBMON_BG_MASK_DOWNSCALE_MAX", "0"))

        if item.is_rejected:
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            candidate_paths = _build_candidate_media_paths(stamp, mask_ext=mask_format)
            snap_path = media_root / candidate_paths.snapshot_rel
            snap_annotated_path = media_root / candidate_paths.snapshot_annotated_rel
            mask_path = media_root / candidate_paths.mask_rel
            mask_overlay_path = media_root / candidate_paths.mask_overlay_rel
            
            _safe_mkdir(snap_path.parent)
            _safe_mkdir(mask_path.parent)

            annotated_frame = _draw_bbox(frame, det_full, show_confidence=True)
            await asyncio.gather(
                _write_jpeg_async(snap_path, frame),
                _write_jpeg_async(snap_annotated_path, annotated_frame)
            )

            mask_rel = None
            mask_overlay_rel = None
            if save_masks and motion_mask is not None:
                overlay_dest = mask_overlay_path if save_mask_overlay else None
                await asyncio.to_thread(
                    _save_motion_mask_images,
                    motion_mask=motion_mask,
                    roi_frame=frame,
                    mask_path=mask_path,
                    overlay_path=overlay_dest,
                    downscale_max=mask_downscale_max,
                )
                mask_rel = candidate_paths.mask_rel
                mask_overlay_rel = candidate_paths.mask_overlay_rel if save_mask_overlay else None

            candidate_extra = {
                "reason": item.rejected_reason or "motion_rejected",
                "bg": {"active": item.bg_active},
                "motion": item.roi_stats or {},
                "detection": {
                    "confidence": float(det_full.conf),
                    "bbox_xyxy": [int(det_full.x1), int(det_full.y1), int(det_full.x2), int(det_full.y2)],
                    "bbox_area": int(det_full.area),
                }
            }
            if item.bbox_stats:
                candidate_extra["motion"].update(item.bbox_stats)

            async with async_session_scope() as db:
                candidate = Candidate(
                    ts=utcnow(),
                    camera_name=s.camera_name,
                    bbox_x1=int(det_full.x1),
                    bbox_y1=int(det_full.y1),
                    bbox_x2=int(det_full.x2),
                    bbox_y2=int(det_full.y2),
                    snapshot_path=candidate_paths.snapshot_rel,
                    annotated_snapshot_path=candidate_paths.snapshot_annotated_rel,
                    mask_path=mask_rel,
                    mask_overlay_path=mask_overlay_rel,
                    extra_json=None
                )
                candidate.set_extra(candidate_extra)
                db.add(candidate)
            return

        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        media_paths = _build_observation_media_paths(stamp)
        snap_id = media_paths.observation_uuid
        
        snap_path = media_root / media_paths.snapshot_rel
        snap_annotated_path = media_root / media_paths.snapshot_annotated_rel
        snap_clip_path = media_root / media_paths.snapshot_clip_rel
        snap_background_path = media_root / media_paths.snapshot_background_rel
        clip_path = media_root / media_paths.clip_rel

        _safe_mkdir(snap_path.parent)
        _safe_mkdir(clip_path.parent)

        annotated_frame = _draw_bbox(frame, det_full, show_confidence=True)
        roi_save_tasks = [
            _write_jpeg_async(snap_path, frame),
            _write_jpeg_async(snap_annotated_path, annotated_frame)
        ]
        
        # Save ROI snapshot if configured
        has_roi = s.roi is not None
        if has_roi:
            roi_frame, _ = _apply_roi(frame, s)
            roi_save_tasks.append(_write_jpeg_async(media_root / media_paths.snapshot_roi_rel, roi_frame))
        
        await asyncio.gather(*roi_save_tasks)

        clip_rel = ""
        if item.video_path and item.video_path.exists():
            # Use pre-recorded video from full visit capture
            try:
                clip_rel = str(item.video_path.relative_to(media_root))
            except Exception as e:
                logger.warning(f"Failed to resolve video path: {e}")
        else:
            # No video available
            clip_rel = "clips/none.mp4"

        if background_img is not None:
            await _write_jpeg_async(snap_background_path, background_img)

        mask_rel = None
        mask_overlay_rel = None
        if save_masks and motion_mask is not None:
            rel_m, rel_o = _build_mask_paths(stamp, snap_id, mask_ext=mask_format)
            await asyncio.to_thread(
                _save_motion_mask_images,
                motion_mask=motion_mask,
                roi_frame=frame,
                mask_path=media_root / rel_m,
                overlay_path=(media_root / rel_o) if save_mask_overlay else None,
                downscale_max=mask_downscale_max
            )
            mask_rel = rel_m
            mask_overlay_rel = rel_o if save_mask_overlay else None

        crop, (raw_species_label, raw_species_prob), emb = await _prepare_crop_and_clip(
            frame, det_full, float(s.crop_padding), snap_clip_path, clip
        )
        species_label = raw_species_label
        species_prob = float(raw_species_prob)
        if species_prob < float(s.min_species_prob):
            species_label = "Hummingbird (unknown species)"

        async with async_session_scope() as db:
            individual_id, match_score = await _match_or_create_individual(
                db, emb, species_label=species_label,
                match_threshold=float(s.match_threshold),
                ema_alpha=float(s.ema_alpha)
            )

            snapshots_data = {
                "annotated_path": media_paths.snapshot_annotated_rel,
                "clip_path": media_paths.snapshot_clip_rel,
                "roi_path": media_paths.snapshot_roi_rel if has_roi else "",
                "background_path": media_paths.snapshot_background_rel if background_img is not None else ""
            }
            
            # Extract video metadata if video exists
            video_metadata = None
            if item.video_path and item.video_path.exists():
                try:
                    video_metadata = extract_video_metadata(item.video_path)
                except Exception as e:
                    logger.warning(f"Failed to extract video metadata: {e}")
            
            extra_data = _build_observation_extra_data(
                observation_uuid=snap_id,
                sensitivity=item.settings_snapshot or {},
                detection={
                    "box_confidence": float(det_full.conf),
                    "bbox_xyxy": [int(det_full.x1), int(det_full.y1), int(det_full.x2), int(det_full.y2)],
                    "bbox_area": int(det_full.area),
                    "nms_iou_threshold": float(s.detect_iou),
                    "background_subtraction_enabled": item.bg_active,
                },
                identification={
                    "individual_id": individual_id,
                    "match_score": float(match_score),
                    "species_label": raw_species_label,
                    "species_prob": species_prob,
                    "species_label_final": species_label
                },
                snapshots=snapshots_data,
                models=item.model_metadata or {},
                video=video_metadata
            )
            if item.roi_stats or item.bbox_stats:
                extra_data["motion"] = {**(item.roi_stats or {}), **(item.bbox_stats or {})}
            
            obs = Observation(
                ts=utcnow(), camera_name=s.camera_name,
                species_label=species_label, species_prob=species_prob,
                individual_id=individual_id, match_score=match_score,
                bbox_x1=int(det_full.x1), bbox_y1=int(det_full.y1),
                bbox_x2=int(det_full.x2), bbox_y2=int(det_full.y2),
                snapshot_path=media_paths.snapshot_rel,
                video_path=clip_rel or "clips/none.mp4",
                extra_json=None
            )
            obs.set_extra(extra_data)
            db.add(obs)
            await db.flush()
            if emb is not None:
                e = Embedding(observation_id=int(obs.id), individual_id=individual_id)
                e.set_vec(emb)
                db.add(e)
            
            logger.info(f"Saved observation {obs.id} (ind={individual_id})")
            
            if os.getenv("HBMON_DEBUG_VERBOSE") == "1":
                 logger.debug(f"Processing details: individual={individual_id}, species={species_label}({species_prob:.2f}), match_score={match_score:.2f}")

    except Exception as e:
        logger.error(f"task failed: {e}", exc_info=True)
    finally:
        sem.release()


async def processing_dispatcher(queue: asyncio.Queue, clip: ClipModel) -> None:
    """
    Continuously pulls items from the queue and dispatches them to be processed.
    """
    max_tasks = int(os.getenv("HBMON_WORKER_MAX_CONCURRENT_TASKS", "4"))
    sem = asyncio.Semaphore(max_tasks)
    media_root = media_dir()
    logger.info(f"Dispatcher started (max_tasks={max_tasks})")
    while True:
        item = await queue.get()
        await sem.acquire()
        asyncio.create_task(process_candidate_task(item, clip, media_root, sem))
        queue.task_done()


def _load_yolo_model() -> tuple[Any, str]:
    """
    Load YOLO model with optional OpenVINO backend for Intel GPU acceleration.

    Environment variables:
    - HBMON_YOLO_MODEL: Model name (default: yolo11n.pt)
    - HBMON_INFERENCE_BACKEND: Unified backend for both YOLO and CLIP
    - HBMON_YOLO_BACKEND: "pytorch" (default), "openvino-cpu", or "openvino-gpu"
      (overrides HBMON_INFERENCE_BACKEND if set)

    Returns:
        tuple: (Loaded YOLO model, device_label string)
    """
    if YOLO is None and not _YOLO_AVAILABLE:
        raise RuntimeError("ultralytics must be installed to load YOLO models")

    model_name = os.getenv("HBMON_YOLO_MODEL", "yolo11n.pt")
    # Priority: HBMON_YOLO_BACKEND > HBMON_INFERENCE_BACKEND > "pytorch"
    backend = os.getenv("HBMON_YOLO_BACKEND") or os.getenv("HBMON_INFERENCE_BACKEND", "pytorch")
    backend = backend.lower().strip()

    # PyTorch backend (default)
    if backend == "pytorch":
        logger.info(f"Loading YOLO model: {model_name} (PyTorch backend)")
        return YOLO(model_name, task="detect"), "PyTorch"  # type: ignore[misc]

    # OpenVINO backends
    if not is_openvino_available():
        logger.warning("OpenVINO not available, falling back to PyTorch")
        logger.info(f"Loading YOLO model: {model_name} (PyTorch backend)")
        return YOLO(model_name, task="detect"), "PyTorch"  # type: ignore[misc]

    # Check GPU for openvino-gpu backend
    if backend == "openvino-gpu":
        if validate_openvino_gpu():
            logger.info("Enabling OpenVINO GPU override patch...")
            force_openvino_gpu_override()
        else:
            logger.warning("OpenVINO GPU not available, falling back to openvino-cpu")
            backend = "openvino-cpu"

    # Export model to OpenVINO format if needed
    ov_suffix = "_openvino_model"
    model_base = model_name.replace(".pt", "")
    
    # Try to use the centralized OpenVINO cache directory
    ov_cache_dir = os.getenv("OPENVINO_CACHE_DIR")
    if ov_cache_dir:
        ov_model_dir = Path(ov_cache_dir) / "yolo" / f"{model_base}{ov_suffix}"
    else:
        yolo_config_dir = Path(os.getenv("YOLO_CONFIG_DIR", "/data/yolo"))
        ov_model_dir = yolo_config_dir / f"{model_base}{ov_suffix}"
    
    # Ensure parent directory exists for export
    ov_model_dir.parent.mkdir(parents=True, exist_ok=True)

    if not ov_model_dir.exists():
        logger.info(f"Exporting {model_name} to OpenVINO format...")
        yolo = YOLO(model_name, task="detect")  # type: ignore[misc]
        try:
            # The export() method returns the path to the exported model directory.
            # Using dynamic=True allows variable input sizes (e.g. 1088x1920) without crashing OpenVINO.
            exported_model_path = yolo.export(format="openvino", dynamic=True, half=False)
            # Move the exported model to our custom cache directory.
            shutil.move(str(exported_model_path), str(ov_model_dir))
            logger.info("OpenVINO model exported successfully")
        except Exception as e:
            logger.warning(f"OpenVINO export failed: {e}")
            logger.info(f"Loading YOLO model: {model_name} (PyTorch backend)")
            return yolo, "PyTorch"

    # Load OpenVINO model
    device = "GPU" if backend == "openvino-gpu" else "CPU"
    logger.info(f"Loading YOLO model: {ov_model_dir} (OpenVINO {device} backend)")

    try:
        yolo = YOLO(str(ov_model_dir), task="detect")  # type: ignore[misc]
        return yolo, f"OpenVINO-{device}"
    except Exception as e:
        logger.warning(f"Failed to load OpenVINO model: {e}")
        logger.info(f"Loading YOLO model: {model_name} (PyTorch backend)")
        return YOLO(model_name, task="detect"), "PyTorch"  # type: ignore[misc]


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

    # Load YOLO model
    yolo, yolo_device_label = _load_yolo_model()
    
    # Capture YOLO model metadata
    yolo_model_name = os.getenv("HBMON_YOLO_MODEL", "yolo11n.pt")
    yolo_backend = os.getenv("HBMON_YOLO_BACKEND") or os.getenv("HBMON_INFERENCE_BACKEND", "pytorch")

    # Resolve the class id for 'bird' from the model's names mapping when possible.
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
        logger.info(f'Using bird_class_id={bird_class_id} (fallback)')
    else:
        logger.info(f'Resolved bird_class_id={bird_class_id} from model names')

    # Start dispatcher
    queue: asyncio.Queue = asyncio.Queue()
    # Initialize CLIP model with backend selection
    # Priority: HBMON_DEVICE > HBMON_INFERENCE_BACKEND > "cpu"
    clip_backend = os.getenv("HBMON_DEVICE") or os.getenv("HBMON_INFERENCE_BACKEND", "cpu")
    clip = ClipModel(backend=clip_backend)
    
    # Capture CLIP model metadata
    clip_model_name = clip.model_name
    clip_pretrained = clip.pretrained
    clip_backend_label = clip.backend
    
    # Prepare model metadata dict for all observations
    model_metadata = {
        "yolo_model": yolo_model_name,
        "yolo_backend": yolo_backend,
        "yolo_backend_label": yolo_device_label,
        "clip_model": clip_model_name,
        "clip_pretrained": clip_pretrained if clip_pretrained else "",
        "clip_backend": clip_backend_label,
    }
    
    asyncio.create_task(processing_dispatcher(queue, clip))

    env = os.getenv("HBMON_SPECIES_LIST", "").strip()
    if env and clip is not None:
        import re
        labels = [s.strip() for s in re.split(r",\s*", env) if s.strip()]
        if labels:
            clip.set_label_space(labels)

    # Load background image for motion-based filtering (if configured)
    background_img: np.ndarray | None = _load_background_image()
    last_background_check = time.time()

    bg_env_overrides: dict[str, object] = {}
    env_bg_enabled = os.getenv("HBMON_BG_SUBTRACTION")
    if env_bg_enabled is not None and env_bg_enabled.strip():
        bg_env_overrides["bg_subtraction_enabled"] = (env_bg_enabled.strip().lower() in ("1", "true", "yes", "y", "on"))

    env_bg_threshold = os.getenv("HBMON_BG_MOTION_THRESHOLD")
    if env_bg_threshold is not None and env_bg_threshold.strip():
        try:
            bg_env_overrides["bg_motion_threshold"] = int(env_bg_threshold)
        except ValueError:
            pass

    env_bg_blur = os.getenv("HBMON_BG_MOTION_BLUR")
    if env_bg_blur is not None and env_bg_blur.strip():
        try:
            bg_env_overrides["bg_motion_blur"] = int(env_bg_blur)
        except ValueError:
            pass

    env_bg_overlap = os.getenv("HBMON_BG_MIN_OVERLAP")
    if env_bg_overlap is not None and env_bg_overlap.strip():
        try:
            bg_env_overrides["bg_min_overlap"] = float(env_bg_overlap)
        except ValueError:
            pass

    cap: cv2.VideoCapture | None = None  # type: ignore
    settings: Settings | None = load_settings(apply_env_overrides=False)
    last_settings_load = time.time()

    last_debug_log = 0.0
    last_logged_candidate_ts = 0.0
    candidate_log_times: deque[float] = deque()

    # Temporal voting buffer: stores recent frames to catch birds visible for only 1-2 frames
    temporal_window = int(settings.temporal_window_frames if settings else 5)
    frame_buffer: deque[FrameEntry] = deque(maxlen=temporal_window)
    logger.info(f"Temporal voting enabled with window size: {temporal_window} frames")

    def get_settings() -> Settings:
        nonlocal last_settings_load, settings
        now = time.time()
        if settings is None or (now - last_settings_load) > 3.0:
            settings = load_settings(apply_env_overrides=False)
            env_rtsp = os.getenv("HBMON_RTSP_URL")
            if env_rtsp:
                settings.rtsp_url = env_rtsp
            env_camera = os.getenv("HBMON_CAMERA_NAME")
            if env_camera:
                settings.camera_name = env_camera  # type: ignore

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

    consecutive_failures = 0

    logger.info(f"Producer loop started. YOLO backend: {yolo_device_label}")
    
    # Full Visit Capture State
    visit_state = VisitState.IDLE
    visit_recorder: BackgroundRecorder | None = None
    visit_start_ts = 0.0
    visit_last_seen_ts = 0.0
    visit_best_candidate: CandidateItem | None = None
    visit_best_score = 0.0
    last_observation_ts: float | None = None
    
    # Arrival buffer: capture context before detection.
    # Default 5.0s (e.g. 100 frames at 20fps). Tuning: 5.0s is good for "approach" capture.
    arr_sec = float(os.getenv("HBMON_ARRIVAL_BUFFER_SECONDS", "5.0"))
    arrival_window_frames = int(20.0 * arr_sec)
    arrival_buffer: deque[np.ndarray] = deque(maxlen=arrival_window_frames)
    logger.info(f"Arrival buffer initialized: {arrival_window_frames} frames ({arr_sec:.1f}s)")
    while True:
        s = get_settings()
        if not s.rtsp_url:
            await asyncio.sleep(2.0)
            continue

        # Dynamic resizing of arrival buffer
        target_arr_frames = int(20.0 * float(s.arrival_buffer_seconds))
        if arrival_buffer.maxlen != target_arr_frames:
             logger.info(f"Resizing arrival buffer: {arrival_buffer.maxlen} -> {target_arr_frames} frames ({s.arrival_buffer_seconds}s)")
             arrival_buffer = deque(arrival_buffer, maxlen=target_arr_frames)

        if cap is None or not cap.isOpened():
            logger.info(f"Opening RTSP: {s.rtsp_url}")
            cap = cv2.VideoCapture(s.rtsp_url)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            except Exception:
                pass
            await asyncio.sleep(0.5)

        ok, frame = cap.read()
        if not ok or frame is None:
            consecutive_failures += 1
            logger.warning(f"Frame read failed. Reconnecting... (failure {consecutive_failures})")
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
            cap = None
            # Use a shorter, increasing delay for read failures
            await asyncio.sleep(min(0.2 * consecutive_failures, 2.0))
            continue
 
        consecutive_failures = 0

        # ROI & Background
        roi_frame = frame
        xoff, yoff = 0, 0
        if s.roi:
            roi_frame, (xoff, yoff) = _apply_roi(frame, s)

        # Debug Logging
        debug_every = float(os.getenv("HBMON_DEBUG_EVERY_SECONDS", "10"))
        debug_save = env_bool("HBMON_DEBUG_SAVE_FRAMES", False)
        now_dbg = time.time()

        if (now_dbg - last_debug_log) > debug_every:
             logger.info(f"alive q_size={queue.qsize()} rtsp={s.rtsp_url}")
             if debug_save:
                 await _write_jpeg_async(media_dir() / "debug_latest.jpg", frame)
             last_debug_log = now_dbg

        motion_mask = None
        bg_active = False
        if s.bg_subtraction_enabled and background_img is not None:
             bg_roi, _ = _apply_roi(background_img, s)
             motion_mask = _compute_motion_mask(roi_frame, bg_roi, threshold=int(s.bg_motion_threshold), blur_size=int(s.bg_motion_blur))
             bg_active = True

        # YOLO Detect
        try:
             yolo_verbose = (os.getenv("HBMON_DEBUG_VERBOSE") == "1")
             
             # Resolve inference image size using shared utility
             imgsz_env = os.getenv("HBMON_YOLO_IMGSZ", "1088,1920").strip()
             predict_imgsz = resolve_predict_imgsz(imgsz_env, roi_frame.shape)

             t0_yolo = time.perf_counter()
             results = yolo.predict(roi_frame, conf=float(s.detect_conf), iou=float(s.detect_iou), classes=[bird_class_id], imgsz=predict_imgsz, verbose=yolo_verbose)
             dt_yolo = (time.perf_counter() - t0_yolo) * 1000
             
             if yolo_verbose:
                  logger.info(f"YOLO Inference ({yolo_device_label}): {dt_yolo:.2f}ms")
        except Exception:
             logger.error("YOLO inference failed", exc_info=True)
             await asyncio.sleep(0.5)
             continue

        detections = _collect_bird_detections(results, int(s.min_box_area), bird_class_id)
        
        # Update arrival ring buffer (raw frames for video)
        arrival_buffer.append(frame.copy())

        # Store frame in temporal buffer
        frame_buffer.append(FrameEntry(
            frame=frame.copy(),
            roi_frame=roi_frame.copy(),
            timestamp=time.time(),
            detections=detections,
            motion_mask=motion_mask.copy() if motion_mask is not None else None,
            xoff=xoff,
            yoff=yoff
        ))

        # Temporal voting: find best detection across recent frames
        best_entry: FrameEntry | None = None
        best_det: Det | None = None

        for entry in frame_buffer:
            for d in entry.detections:
                if best_det is None or d.conf > best_det.conf:
                    best_det = d
                    best_entry = entry

        if best_entry is None:
            # No detections in window. Use latest frame context to ensure state machine runs.
            if len(frame_buffer) > 0:
                best_entry = frame_buffer[-1]
            else:
                continue

        # Restore context from the best frame
        frame = best_entry.frame
        detections = best_entry.detections
        motion_mask = best_entry.motion_mask
        xoff = best_entry.xoff
        yoff = best_entry.yoff
        bg_active = (motion_mask is not None)

        if os.getenv("HBMON_DEBUG_VERBOSE") == "1":
             logger.debug(f"Found {len(detections)} bird detections.")

        # Check motion overlap
        kept_entries = []
        rejected_entries = []
        if motion_mask is not None:
            for d in detections:
                stats = _motion_overlap_stats(d, motion_mask)
                if float(stats["bbox_overlap_ratio"]) >= float(s.bg_min_overlap):
                    kept_entries.append((d, stats))
                else:
                    rejected_entries.append((d, stats))
        else:
            kept_entries = [(d, {}) for d in detections]

        det = None
        det_stats = {}
        if kept_entries:
            det, det_stats = _select_best_detection(kept_entries)
        elif os.getenv("HBMON_DEBUG_BG") == "1" and rejected_entries:
            # Optionally log why detections were rejected
            for d, stats in rejected_entries:
                logger.debug(f"REJECTED: p={d.conf:.2f} area={d.area} overlap={stats['bbox_overlap_ratio']:.2f} (min={s.bg_min_overlap})")

        # Use timestamp from the selected frame
        assert best_entry is not None
        timestamp = best_entry.timestamp

        if det:
            pass # Replaced by logic below

        # Determine if we have a valid bird
        is_bird_present = (det is not None)
        
        # State Machine Logic
        current_time = timestamp # Use frame timestamp
        
        # 1. Transitions & Actions
        cooldown_seconds = float(s.cooldown_seconds)
        cooldown_active = (
            last_observation_ts is not None
            and cooldown_seconds > 0.0
            and (current_time - last_observation_ts) < cooldown_seconds
        )

        if is_bird_present:
             visit_last_seen_ts = current_time
             
             if visit_state == VisitState.IDLE:
                 if cooldown_active:
                     if os.getenv("HBMON_DEBUG_VERBOSE") == "1":
                         logger.debug("Cooldown active; skipping visit start")
                     continue
                 # START RECORDING
                 visit_state = VisitState.RECORDING
                 visit_start_ts = current_time
                 logger.info(f"Visit STARTED. Detection p={det.conf:.2f}")
                 
                 # Prepare video path
                 visit_video_uuid = uuid.uuid4().hex
                 stamp = time.strftime("%Y%m%d", time.localtime(current_time))
                 video_dir = media_dir() / "clips" / stamp
                 video_dir.mkdir(parents=True, exist_ok=True)
                 visit_video_path = video_dir / f"{visit_video_uuid}.mp4"
                 
                 # Start recorder (approx 20fps)
                 # Videos are stored uncompressed for ML training quality
                 # Compression happens on-the-fly during browser streaming
                 visit_recorder = BackgroundRecorder(
                     visit_video_path, 
                     fps=20.0, 
                     width=frame.shape[1], 
                     height=frame.shape[0]
                 )
                 visit_recorder.start()
                 
                 # Dump arrival buffer
                 for f in arrival_buffer:
                      visit_recorder.feed(f)
                 
                 # Reset best candidate tracking
                 visit_best_candidate = None
                 visit_best_score = -1.0
             
             if visit_state in (VisitState.RECORDING, VisitState.FINALIZING):
                 if visit_state == VisitState.FINALIZING:
                      # Bird returned! Resume recording
                      visit_state = VisitState.RECORDING
                      logger.info("Visit RESUMED")

                 # Update best candidate: prefer large, high-confidence detections
                 # Score is proportional to bounding-box area scaled by confidence.
                 score = det.area * det.conf

                 if score > visit_best_score:
                     visit_best_score = score
                     det_full = Det(x1=det.x1 + xoff, y1=det.y1 + yoff, x2=det.x2 + xoff, y2=det.y2 + yoff, conf=det.conf)
                     
                     visit_best_candidate = CandidateItem(
                         frame=frame.copy(),
                         det_full=det_full,
                         det=det,
                         timestamp=timestamp,
                         motion_mask=motion_mask.copy() if motion_mask is not None else None,
                         background_img=background_img,
                         settings_snapshot={
                              "detect_conf": float(s.detect_conf),
                              "detect_iou": float(s.detect_iou),
                              "min_box_area": int(s.min_box_area),
                              "fps_limit": float(s.fps_limit),
                              "cooldown_seconds": float(s.cooldown_seconds),
                              "bg_subtraction_enabled": bool(s.bg_subtraction_enabled),
                              "bg_subtraction_configured": bool(s.background_image),
                              "background_image_available": background_img is not None,
                          },
                         roi_stats={}, 
                         bbox_stats=det_stats,
                         bg_active=bg_active,
                         s=s,
                         is_rejected=False,
                         video_path=visit_video_path,
                         model_metadata=model_metadata
                     )

        # 2. Continuous Actions (Recording) and Timeouts
        if visit_state in (VisitState.RECORDING, VisitState.FINALIZING):
             if visit_recorder:
                 visit_recorder.feed(frame)
        
             # Check Departure
             if visit_state == VisitState.RECORDING:
                 dep_time = float(s.departure_timeout_seconds)
                 if (current_time - visit_last_seen_ts) > dep_time:
                     visit_state = VisitState.FINALIZING
                     logger.info("Visit FINALIZING (bird left?)")
            
             elif visit_state == VisitState.FINALIZING:
                 dep_time = float(s.departure_timeout_seconds)
                 post_time = float(s.post_departure_buffer_seconds)
                 if (current_time - visit_last_seen_ts) > (dep_time + post_time): # Total timeout
                         # END VISIT
                         visit_state = VisitState.IDLE
                         if visit_recorder:
                              visit_recorder.stop()
                              visit_recorder = None
                         
                         # Process the best candidate
                         if visit_best_candidate:
                              queue.put_nowait(visit_best_candidate)
                              duration = current_time - visit_start_ts
                              logger.info(f"Visit ENDED. Queued best candidate (p={visit_best_candidate.det.conf:.2f}). Duration: {duration:.1f}s")
                              last_observation_ts = current_time
                         else:
                              logger.warning("Visit ended but no candidate found?")
                         
                         frame_buffer.clear()

        # Rejected ?
        log_rejected = env_bool("HBMON_BG_LOG_REJECTED", False)
        if (not det) and log_rejected and rejected_entries:
            # Check rejected cooldown
             rejected_cooldown = float(env_int("HBMON_BG_REJECTED_COOLDOWN_SECONDS", 3))
             max_per_minute = env_int("HBMON_BG_REJECTED_MAX_PER_MINUTE", 30)
             
             if (timestamp - last_logged_candidate_ts >= rejected_cooldown):
                  # Also check max per minute
                  while candidate_log_times and (timestamp - candidate_log_times[0]) > 60.0:
                      candidate_log_times.popleft()
                  
                  if len(candidate_log_times) < max_per_minute:
                      best_rej, rej_stats = _select_best_detection(rejected_entries)
                      det_full_rej = Det(x1=best_rej.x1 + xoff, y1=best_rej.y1 + yoff, x2=best_rej.x2 + xoff, y2=best_rej.y2 + yoff, conf=best_rej.conf)
                      
                      item = CandidateItem(
                          frame=frame.copy(),
                          det_full=det_full_rej,
                          timestamp=timestamp,
                          motion_mask=motion_mask.copy() if motion_mask is not None else None,
                          background_img=background_img,
                          settings_snapshot={}, 
                          roi_stats=_roi_motion_stats(motion_mask) if motion_mask is not None else {},
                          bbox_stats=rej_stats,
                          bg_active=bg_active,
                          s=s,
                          is_rejected=True,
                          rejected_reason="motion_rejected",
                          model_metadata=model_metadata
                      )
                      queue.put_nowait(item)
                      last_logged_candidate_ts = timestamp
                      candidate_log_times.append(timestamp)


        # Update background logic (periodic check)
        now_bg = time.time()
        if now_bg - last_background_check > 30.0:
             last_background_check = now_bg
             new_bg = _load_background_image()
             background_img = new_bg
        
        await asyncio.sleep(0.01)

def main() -> None:
    asyncio.run(run_worker())

if __name__ == "__main__":
    main()
