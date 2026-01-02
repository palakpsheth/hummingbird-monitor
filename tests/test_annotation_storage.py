from __future__ import annotations

import json
from pathlib import Path

import pytest

from hbmon.annotation_storage import (
    annotations_root,
    boxes_dir,
    ensure_annotation_dirs,
    frames_dir,
    labels_dir,
    manifest_path,
    read_manifest,
    write_boxes_json,
    write_manifest,
    write_yolo_labels,
)


def test_annotation_paths_use_safe_observation_ids(tmp_path: Path) -> None:
    base = tmp_path / "data"
    obs_id = 123
    assert frames_dir(obs_id, base).name == str(obs_id)
    assert labels_dir(obs_id, base).name == str(obs_id)
    assert boxes_dir(obs_id, base).name == str(obs_id)
    assert manifest_path(obs_id, base).name == f"{obs_id}.json"


def test_annotation_paths_reject_invalid_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        frames_dir("../bad", tmp_path)


def test_manifest_roundtrip(tmp_path: Path) -> None:
    base = tmp_path / "data"
    manifest = {"total_frames": 5, "reviewed_frames": 2}
    path = write_manifest(7, manifest, base)
    assert path.exists()
    loaded = read_manifest(7, base)
    assert loaded == manifest


def test_read_manifest_returns_none_when_invalid(tmp_path: Path) -> None:
    base = tmp_path / "data"
    path = manifest_path(9, base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not-json", encoding="utf-8")
    assert read_manifest(9, base) is None


def test_read_manifest_returns_none_for_non_dict(tmp_path: Path) -> None:
    base = tmp_path / "data"
    path = manifest_path(10, base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]", encoding="utf-8")
    assert read_manifest(10, base) is None


def test_write_yolo_labels(tmp_path: Path) -> None:
    label_path = tmp_path / "labels" / "frame_000001.txt"
    boxes = [{"class_id": 0, "x": 0.5, "y": 0.5, "w": 0.25, "h": 0.4}]
    write_yolo_labels(label_path, boxes)
    text = label_path.read_text(encoding="utf-8").strip()
    assert text.startswith("0 0.500000 0.500000 0.250000 0.400000")


def test_write_boxes_json(tmp_path: Path) -> None:
    box_path = tmp_path / "boxes" / "frame_000001.json"
    boxes = [{"class_id": 0, "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4, "is_false_positive": True}]
    write_boxes_json(box_path, boxes)
    loaded = json.loads(box_path.read_text(encoding="utf-8"))
    assert loaded == boxes


def test_write_manifest_rejects_non_dict(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_manifest(11, ["bad"], tmp_path)


def test_annotations_root_defaults_to_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path))
    root = annotations_root()
    assert root == tmp_path / "exports" / "annotations"


def test_ensure_annotation_dirs_creates_layout(tmp_path: Path) -> None:
    ensure_annotation_dirs(42, base_dir=tmp_path)
    assert (tmp_path / "exports" / "annotations" / "frames" / "42").exists()
    assert (tmp_path / "exports" / "annotations" / "labels" / "42").exists()
    assert (tmp_path / "exports" / "annotations" / "boxes" / "42").exists()
    assert (tmp_path / "exports" / "annotations" / "manifest").exists()
