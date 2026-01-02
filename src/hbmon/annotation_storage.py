# src/hbmon/annotation_storage.py
"""
Helpers for managing annotation storage on disk.

Directory layout (rooted at ``/data/exports/annotations`` by default):

- frames/{obs_id}/frame_000001.jpg
- labels/{obs_id}/frame_000001.txt
- boxes/{obs_id}/frame_000001.json
- manifest/{obs_id}.json

These helpers provide path sanitation, manifest read/write helpers, and
basic writers for YOLO labels and box JSON payloads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from hbmon.config import data_dir


def _normalize_observation_id(observation_id: int | str) -> str:
    """Normalize observation IDs to a safe directory name."""
    text = str(observation_id).strip()
    if not text.isdigit():
        raise ValueError(f"Invalid observation id: {observation_id!r}")
    return text


def annotations_root(base_dir: Path | None = None) -> Path:
    """Return the root directory for annotations."""
    base = base_dir if base_dir is not None else data_dir()
    return base / "exports" / "annotations"


def frames_dir(observation_id: int | str, base_dir: Path | None = None) -> Path:
    """Return the directory holding extracted frames for an observation."""
    obs = _normalize_observation_id(observation_id)
    return annotations_root(base_dir) / "frames" / obs


def labels_dir(observation_id: int | str, base_dir: Path | None = None) -> Path:
    """Return the directory holding YOLO labels for an observation."""
    obs = _normalize_observation_id(observation_id)
    return annotations_root(base_dir) / "labels" / obs


def boxes_dir(observation_id: int | str, base_dir: Path | None = None) -> Path:
    """Return the directory holding raw box JSON for an observation."""
    obs = _normalize_observation_id(observation_id)
    return annotations_root(base_dir) / "boxes" / obs


def manifest_path(observation_id: int | str, base_dir: Path | None = None) -> Path:
    """Return the manifest path for an observation."""
    obs = _normalize_observation_id(observation_id)
    return annotations_root(base_dir) / "manifest" / f"{obs}.json"


def ensure_annotation_dirs(observation_id: int | str, base_dir: Path | None = None) -> None:
    """Create the annotation directory structure for an observation."""
    frames_dir(observation_id, base_dir).mkdir(parents=True, exist_ok=True)
    labels_dir(observation_id, base_dir).mkdir(parents=True, exist_ok=True)
    boxes_dir(observation_id, base_dir).mkdir(parents=True, exist_ok=True)
    manifest_path(observation_id, base_dir).parent.mkdir(parents=True, exist_ok=True)


def read_manifest(observation_id: int | str, base_dir: Path | None = None) -> dict[str, Any] | None:
    """Load the annotation manifest JSON, returning None if missing/invalid."""
    path = manifest_path(observation_id, base_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_manifest(
    observation_id: int | str,
    manifest: dict[str, Any],
    base_dir: Path | None = None,
) -> Path:
    """Write the annotation manifest JSON and return the file path."""
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a dictionary.")
    path = manifest_path(observation_id, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True)
    path.write_text(payload + "\n", encoding="utf-8")
    return path


def write_yolo_labels(label_path: Path, boxes: Iterable[dict[str, Any]]) -> None:
    """Write YOLO label lines for the provided boxes."""
    lines: list[str] = []
    for box in boxes:
        class_id = int(box.get("class_id", 0))
        x = float(box.get("x", 0.0))
        y = float(box.get("y", 0.0))
        w = float(box.get("w", 0.0))
        h = float(box.get("h", 0.0))
        lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_boxes_json(box_path: Path, boxes: list[dict[str, Any]]) -> None:
    """Write raw box JSON including false-positive tags."""
    payload = json.dumps(boxes, indent=2, sort_keys=True)
    box_path.parent.mkdir(parents=True, exist_ok=True)
    box_path.write_text(payload + "\n", encoding="utf-8")
