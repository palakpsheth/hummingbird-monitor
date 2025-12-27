"""
Additional tests for worker helper functions.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hbmon.config import Roi, Settings
from hbmon.worker import (
    Det,
    _apply_roi,
    _bbox_area_ratio,
    _bbox_with_padding,
    _build_observation_extra_data,
    _build_observation_media_paths,
    _convert_to_h264,
    _detection_overlaps_motion,
    _format_bbox_label,
    _sanitize_bg_params,
)


def test_build_observation_media_paths_with_fixed_uuid():
    paths = _build_observation_media_paths("20240101", observation_uuid="abc123")
    assert paths.observation_uuid == "abc123"
    assert paths.snapshot_rel.endswith("snapshots/20240101/abc123.jpg")
    assert paths.snapshot_annotated_rel.endswith("snapshots/20240101/abc123_annotated.jpg")
    assert paths.snapshot_clip_rel.endswith("snapshots/20240101/abc123_clip.jpg")
    assert paths.clip_rel.endswith("clips/20240101/abc123.mp4")


def test_build_observation_extra_data_defaults_review():
    data = _build_observation_extra_data(
        observation_uuid="obs-1",
        sensitivity={"detect_conf": 0.3},
        detection={"bbox_area": 500},
        identification={"species": "Anna's"},
        snapshots={"raw": "snap.jpg"},
        review=None,
    )
    assert data["observation_uuid"] == "obs-1"
    assert data["review"] == {"label": None}


def test_apply_roi_clamps_and_returns_offset():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    settings = Settings()
    settings.roi = Roi(x1=0.1, y1=0.2, x2=0.6, y2=0.8)
    roi_frame, (x_off, y_off) = _apply_roi(frame, settings)
    assert roi_frame.shape[0] == 60
    assert roi_frame.shape[1] == 100
    assert (x_off, y_off) == (20, 20)


def test_sanitize_bg_params_normalizes_values():
    enabled, threshold, blur, overlap = _sanitize_bg_params(
        enabled=True,
        threshold=300,
        blur=4,
        min_overlap=-0.5,
    )
    assert enabled is True
    assert threshold == 255
    assert blur == 5
    assert overlap == 0.0


def test_detection_overlaps_motion_threshold():
    motion_mask = np.zeros((10, 10), dtype=np.uint8)
    motion_mask[2:6, 2:6] = 255
    det = Det(x1=0, y1=0, x2=5, y2=5, conf=0.9)
    assert _detection_overlaps_motion(det, motion_mask, min_overlap_ratio=0.1)
    assert not _detection_overlaps_motion(det, motion_mask, min_overlap_ratio=0.9)


def test_bbox_helpers_format_and_ratio():
    det = Det(x1=0, y1=0, x2=10, y2=5, conf=0.42)
    assert _format_bbox_label(det) == "0.42 | 50px^2"
    assert _bbox_area_ratio(det, (10, 10)) == 0.5
    assert _bbox_area_ratio(det, (0, 0)) == 0.0


def test_bbox_with_padding_clamps():
    det = Det(x1=5, y1=5, x2=10, y2=10, conf=0.3)
    x1, y1, x2, y2 = _bbox_with_padding(det, (12, 12), pad_frac=0.5)
    assert (x1, y1) == (2, 2)
    assert (x2, y2) == (12, 12)


def test_convert_to_h264_handles_missing_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setattr("hbmon.worker._ffmpeg_available", lambda: False)
    result = _convert_to_h264(tmp_path / "in.mp4", tmp_path / "out.mp4")
    assert result is False


def test_convert_to_h264_success(monkeypatch, tmp_path):
    monkeypatch.setattr("hbmon.worker._ffmpeg_available", lambda: True)
    output_path = tmp_path / "out.mp4"
    output_path.write_bytes(b"ok")

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr("hbmon.worker.subprocess.run", fake_run)
    assert _convert_to_h264(tmp_path / "in.mp4", output_path)
