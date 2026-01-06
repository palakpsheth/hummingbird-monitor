# src/hbmon/annotation_storage.py
"""
Annotation storage helpers for frame-level labeling and hard-negative mining.

This module provides utility functions for:
- Disk layout management under /data/exports/annotations/
- YOLO label file generation
- Manifest read/write for pipeline state
- Hard-negative crop export

Directory Structure:
    /data/exports/annotations/
    ├── frames/{obs_id}/frame_000001.jpg    # Extracted video frames
    ├── labels/{obs_id}/frame_000001.txt    # YOLO-format labels (valid boxes only)
    ├── manifest/{obs_id}.json              # Annotation state + progress
    └── boxes/{obs_id}/frame_000001.json    # All boxes including FP flags

    /data/exports/yolo/
    ├── hard_negatives/                     # Cropped false-positive regions
    └── dataset/                            # Exported YOLO dataset
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Base paths
EXPORTS_BASE = Path(os.environ.get("HBMON_DATA_DIR", "/data")) / "exports"
ANNOTATION_BASE = EXPORTS_BASE / "annotations"
YOLO_BASE = EXPORTS_BASE / "yolo"

# Annotation subdirectories
FRAMES_DIR = ANNOTATION_BASE / "frames"
LABELS_DIR = ANNOTATION_BASE / "labels"
MANIFEST_DIR = ANNOTATION_BASE / "manifest"
BOXES_DIR = ANNOTATION_BASE / "boxes"

# YOLO export directories
HARD_NEGATIVES_DIR = YOLO_BASE / "hard_negatives"
DATASET_DIR = YOLO_BASE / "dataset"


@dataclass
class AnnotationSummary:
    """Summary of annotation progress for an observation."""
    total_frames: int = 0
    reviewed_frames: int = 0
    pending_frames: int = 0
    state: str = "pending"  # pending, preprocessing, in_review, completed
    last_updated: str | None = None


@dataclass
class BoxData:
    """Single bounding box annotation."""
    class_id: int
    x: float  # center x (0-1)
    y: float  # center y (0-1)
    w: float  # width (0-1)
    h: float  # height (0-1)
    is_false_positive: bool = False
    source: str = "auto"
    confidence: float | None = None  # Detector confidence (0-1), null for manual boxes


def ensure_annotation_dirs() -> None:
    """Create all annotation directories if they don't exist."""
    for d in [FRAMES_DIR, LABELS_DIR, MANIFEST_DIR, BOXES_DIR, HARD_NEGATIVES_DIR, DATASET_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _sanitize_obs_id(obs_id: str | int) -> str:
    """Sanitize observation ID to prevent directory traversal."""
    clean = str(obs_id).replace("/", "_").replace("\\", "_").replace("..", "_")
    if not clean or clean.startswith("."):
        raise ValueError(f"Invalid observation ID: {obs_id}")
    return clean


def get_obs_dir(obs_id: str | int) -> Path:
    """Get the base annotation directory for an observation.
    
    This is a convenience function that returns the frames directory,
    which serves as the canonical "observation directory" for cleanup.
    All observation-specific directories (frames/labels/boxes) use the same
    sanitized obs_id, so they can be found via get_frame_dir, get_labels_dir, etc.
    """
    return FRAMES_DIR / _sanitize_obs_id(obs_id)


def get_frame_dir(obs_id: str | int) -> Path:
    """Get the frames directory for an observation."""
    return FRAMES_DIR / _sanitize_obs_id(obs_id)


def get_labels_dir(obs_id: str | int) -> Path:
    """Get the labels directory for an observation."""
    return LABELS_DIR / _sanitize_obs_id(obs_id)


def get_boxes_dir(obs_id: str | int) -> Path:
    """Get the boxes (with FP flags) directory for an observation."""
    return BOXES_DIR / _sanitize_obs_id(obs_id)


def get_frame_path(obs_id: str | int, frame_idx: int) -> Path:
    """Get the path for a specific frame image."""
    return get_frame_dir(obs_id) / f"frame_{frame_idx:06d}.jpg"


def get_label_path(obs_id: str | int, frame_idx: int) -> Path:
    """Get the path for a YOLO label file (valid boxes only)."""
    return get_labels_dir(obs_id) / f"frame_{frame_idx:06d}.txt"


def get_box_json_path(obs_id: str | int, frame_idx: int) -> Path:
    """Get the path for a box JSON file (all boxes with FP flags)."""
    return get_boxes_dir(obs_id) / f"frame_{frame_idx:06d}.json"


def get_manifest_path(obs_id: str | int) -> Path:
    """Get the manifest path for an observation."""
    return MANIFEST_DIR / f"{_sanitize_obs_id(obs_id)}.json"


def save_yolo_label(obs_id: str | int, frame_idx: int, boxes: list[BoxData]) -> Path | None:
    """
    Save a YOLO-format label file with valid boxes only.

    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    Each value is normalized to 0-1.

    Args:
        obs_id: Observation identifier
        frame_idx: Frame index
        boxes: List of BoxData objects

    Returns:
        Path to the saved label file, or None if no valid boxes
    """
    # Filter out false positives
    valid_boxes = [b for b in boxes if not b.is_false_positive]

    label_path = get_label_path(obs_id, frame_idx)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    if not valid_boxes:
        # Write empty file for true negative (no bird present)
        label_path.write_text("")
        return label_path

    lines = []
    for box in valid_boxes:
        # YOLO format: class x_center y_center width height
        line = f"{box.class_id} {box.x:.6f} {box.y:.6f} {box.w:.6f} {box.h:.6f}"
        lines.append(line)

    label_path.write_text("\n".join(lines) + "\n")
    return label_path


def save_box_json(obs_id: str | int, frame_idx: int, boxes: list[BoxData], bird_present: bool = True) -> Path:
    """
    Save all boxes (including false positives) as JSON for reference.

    Args:
        obs_id: Observation identifier
        frame_idx: Frame index
        boxes: List of BoxData objects
        bird_present: Whether a bird is present in this frame

    Returns:
        Path to the saved JSON file
    """
    json_path = get_box_json_path(obs_id, frame_idx)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "frame_index": frame_idx,
        "bird_present": bird_present,
        "boxes": [asdict(box) for box in boxes],
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }

    json_path.write_text(json.dumps(data, indent=2))
    return json_path


def load_box_json(obs_id: str | int, frame_idx: int) -> dict[str, Any] | None:
    """Load box JSON for a frame, returning None if not found."""
    json_path = get_box_json_path(obs_id, frame_idx)
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load box JSON {json_path}: {e}")
        return None


def load_manifest(obs_id: str | int) -> dict[str, Any] | None:
    """Load the annotation manifest for an observation."""
    manifest_path = get_manifest_path(obs_id)
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load manifest {manifest_path}: {e}")
        return None


def save_manifest(obs_id: str | int, data: dict[str, Any]) -> Path:
    """Save the annotation manifest for an observation."""
    manifest_path = get_manifest_path(obs_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(data, indent=2))
    return manifest_path


def sync_db_to_manifest(obs_id: str | int, summary: AnnotationSummary | None) -> Path:
    """
    Update the manifest with the current DB annotation summary.

    Args:
        obs_id: Observation identifier
        summary: Current annotation summary from DB, or None to reset

    Returns:
        Path to the updated manifest
    """
    existing = load_manifest(obs_id) or {}
    # If summary is None (reset case), use a default pending state
    if summary is None:
        summary = AnnotationSummary(state="pending")
    existing.update(asdict(summary))
    existing["synced_at"] = datetime.utcnow().isoformat() + "Z"
    return save_manifest(obs_id, existing)


def get_false_positive_boxes(obs_id: str | int) -> list[tuple[int, BoxData]]:
    """
    Get all false-positive boxes for an observation.

    Returns:
        List of (frame_idx, BoxData) tuples for all FP boxes
    """
    boxes_dir = get_boxes_dir(obs_id)
    if not boxes_dir.exists():
        return []

    fp_boxes = []
    for json_file in sorted(boxes_dir.glob("frame_*.json")):
        try:
            data = json.loads(json_file.read_text())
            frame_idx = data.get("frame_index", 0)
            for box_data in data.get("boxes", []):
                if box_data.get("is_false_positive", False):
                    box = BoxData(
                        class_id=box_data.get("class_id", 0),
                        x=box_data.get("x", 0),
                        y=box_data.get("y", 0),
                        w=box_data.get("w", 0),
                        h=box_data.get("h", 0),
                        is_false_positive=True,
                        source=box_data.get("source", "auto"),
                    )
                    fp_boxes.append((frame_idx, box))
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.warning(f"Failed to parse {json_file}: {e}")
            continue

    return fp_boxes


def export_hard_negative_crop(
    obs_id: str | int,
    frame_idx: int,
    box: BoxData,
    frame_path: Path,
    margin: float = 0.1,
) -> Path | None:
    """
    Crop a false-positive region from a frame for hard-negative mining.

    Args:
        obs_id: Observation identifier
        frame_idx: Frame index
        box: BoxData for the false-positive region
        frame_path: Path to the source frame image
        margin: Extra margin around the box (fraction of box size)

    Returns:
        Path to the saved crop, or None if cropping failed
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available for hard-negative cropping")
        return None

    if not frame_path.exists():
        logger.warning(f"Frame not found: {frame_path}")
        return None

    img = cv2.imread(str(frame_path))
    if img is None:
        logger.warning(f"Failed to read frame: {frame_path}")
        return None

    h, w = img.shape[:2]

    # Convert normalized coords to pixel coords
    cx, cy = box.x * w, box.y * h
    bw, bh = box.w * w, box.h * h

    # Add margin
    bw_m = bw * (1 + margin)
    bh_m = bh * (1 + margin)

    # Calculate crop bounds
    x1 = max(0, int(cx - bw_m / 2))
    y1 = max(0, int(cy - bh_m / 2))
    x2 = min(w, int(cx + bw_m / 2))
    y2 = min(h, int(cy + bh_m / 2))

    if x2 <= x1 or y2 <= y1:
        logger.warning(f"Invalid crop bounds for {obs_id} frame {frame_idx}")
        return None

    crop = img[y1:y2, x1:x2]

    # Save to hard negatives directory
    crop_dir = HARD_NEGATIVES_DIR / _sanitize_obs_id(obs_id)
    crop_dir.mkdir(parents=True, exist_ok=True)

    crop_name = f"frame_{frame_idx:06d}_box_{int(box.x * 1000)}_{int(box.y * 1000)}.jpg"
    crop_path = crop_dir / crop_name

    cv2.imwrite(str(crop_path), crop)
    return crop_path


def count_annotation_files(obs_id: str | int) -> dict[str, int]:
    """
    Count annotation files for an observation.

    Returns:
        Dict with counts: frames, labels, boxes
    """
    return {
        "frames": len(list(get_frame_dir(obs_id).glob("*.jpg"))) if get_frame_dir(obs_id).exists() else 0,
        "labels": len(list(get_labels_dir(obs_id).glob("*.txt"))) if get_labels_dir(obs_id).exists() else 0,
        "boxes": len(list(get_boxes_dir(obs_id).glob("*.json"))) if get_boxes_dir(obs_id).exists() else 0,
    }
