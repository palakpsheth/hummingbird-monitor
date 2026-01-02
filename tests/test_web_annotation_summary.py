from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from hbmon.annotation_storage import write_manifest
from hbmon.models import Observation
from hbmon.web import _annotation_summary_for_observation


def test_annotation_summary_defaults() -> None:
    obs = Observation(id=1, snapshot_path="snap.jpg", video_path="clip.mp4")
    obs.ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    summary = _annotation_summary_for_observation(obs)
    assert summary["total_frames"] == 0
    assert summary["state"] == "available"


def test_annotation_summary_from_extra_json() -> None:
    obs = Observation(id=2, snapshot_path="snap.jpg", video_path="clip.mp4")
    obs.set_extra(
        {
            "annotation_summary": {
                "total_frames": 5,
                "reviewed_frames": 2,
                "pending_frames": 3,
                "state": "in_review",
                "last_updated": "2025-01-02T00:00:00Z",
            }
        }
    )
    summary = _annotation_summary_for_observation(obs)
    assert summary["reviewed_frames"] == 2
    assert summary["pending_frames"] == 3
    assert summary["state"] == "in_review"


def test_annotation_summary_uses_observation_timestamp() -> None:
    obs = Observation(id=3, snapshot_path="snap.jpg", video_path="clip.mp4")
    obs.ts = datetime(2025, 1, 3, tzinfo=timezone.utc)
    obs.set_extra({"annotation_summary": {"total_frames": 1, "reviewed_frames": 0}})
    summary = _annotation_summary_for_observation(obs)
    assert summary["last_updated_utc"] == "2025-01-03T00:00:00Z"


def test_annotation_summary_defaults_when_timestamp_missing() -> None:
    obs = Observation(id=4, snapshot_path="snap.jpg", video_path="clip.mp4")
    obs.ts = None
    summary = _annotation_summary_for_observation(obs)
    assert summary["last_updated_utc"].endswith("Z")


def test_annotation_summary_invalid_last_updated_falls_back() -> None:
    obs = Observation(id=5, snapshot_path="snap.jpg", video_path="clip.mp4")
    obs.ts = datetime(2025, 1, 5, tzinfo=timezone.utc)
    obs.set_extra({"annotation_summary": {"last_updated": "not-a-date"}})
    summary = _annotation_summary_for_observation(obs)
    assert summary["last_updated_utc"] == "2025-01-05T00:00:00Z"


def test_annotation_summary_uses_manifest_when_missing_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path))
    manifest = {
        "total_frames": 4,
        "reviewed_frames": 1,
        "pending_frames": 3,
        "state": "in_review",
        "last_updated": "2025-01-06T00:00:00Z",
    }
    write_manifest(6, manifest, tmp_path)
    obs = Observation(id=6, snapshot_path="snap.jpg", video_path="clip.mp4")
    obs.ts = datetime(2025, 1, 6, tzinfo=timezone.utc)
    summary = _annotation_summary_for_observation(obs)
    assert summary["reviewed_frames"] == 1
    assert summary["pending_frames"] == 3
    assert summary["state"] == "in_review"
    assert summary["last_updated_utc"] == "2025-01-06T00:00:00Z"
