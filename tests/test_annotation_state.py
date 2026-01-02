from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from hbmon.annotation_state import (
    ANNOTATION_STATES,
    normalize_annotation_summary,
    persist_annotation_summary,
    update_observation_annotation_summary,
)
from hbmon.models import Observation


def test_normalize_annotation_summary_defaults_state() -> None:
    summary = normalize_annotation_summary(total_frames=0, reviewed_frames=0)
    assert summary["state"] == "available"
    assert summary["pending_frames"] == 0


def test_normalize_annotation_summary_completed() -> None:
    summary = normalize_annotation_summary(total_frames=3, reviewed_frames=3)
    assert summary["state"] == "completed"
    assert summary["pending_frames"] == 0


def test_normalize_annotation_summary_in_review() -> None:
    summary = normalize_annotation_summary(total_frames=5, reviewed_frames=2)
    assert summary["state"] == "in_review"
    assert summary["pending_frames"] == 3


def test_normalize_annotation_summary_preprocessing() -> None:
    summary = normalize_annotation_summary(total_frames=5, reviewed_frames=0)
    assert summary["state"] == "preprocessing"


def test_normalize_annotation_summary_respects_state() -> None:
    summary = normalize_annotation_summary(total_frames=1, reviewed_frames=0, state="in_review")
    assert summary["state"] == "in_review"


def test_normalize_annotation_summary_clamps_counts() -> None:
    summary = normalize_annotation_summary(total_frames=2, reviewed_frames=5, pending_frames=10)
    assert summary["reviewed_frames"] == 2
    assert summary["pending_frames"] == 0


def test_normalize_annotation_summary_timestamp_format() -> None:
    ts = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    summary = normalize_annotation_summary(total_frames=0, reviewed_frames=0, last_updated_at=ts)
    assert summary["last_updated"].endswith("Z")


def test_update_observation_annotation_summary() -> None:
    obs = Observation(id=1, snapshot_path="snap.jpg", video_path="clip.mp4")
    summary = normalize_annotation_summary(total_frames=2, reviewed_frames=1)
    extra = update_observation_annotation_summary(obs, summary)
    assert extra["annotation_summary"]["reviewed_frames"] == 1
    assert extra["annotation_summary"]["state"] in ANNOTATION_STATES


def test_update_observation_annotation_summary_requires_merge_extra() -> None:
    with pytest.raises(TypeError):
        update_observation_annotation_summary(object(), {})


def test_persist_annotation_summary_writes_manifest(tmp_path: Path) -> None:
    obs = Observation(id=3, snapshot_path="snap.jpg", video_path="clip.mp4")
    summary = normalize_annotation_summary(total_frames=1, reviewed_frames=0)
    persist_annotation_summary(obs, obs.id, summary, base_dir=tmp_path)
    manifest = tmp_path / "exports" / "annotations" / "manifest" / f"{obs.id}.json"
    assert manifest.exists()
