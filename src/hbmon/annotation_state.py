# src/hbmon/annotation_state.py
"""
Annotation state helpers.

Provides normalization logic for annotation progress tracking, along with helpers
to persist summary metadata to observation ``extra_json`` and annotation
manifests on disk.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from hbmon.annotation_storage import write_manifest

ANNOTATION_STATES = ("available", "preprocessing", "in_review", "completed")


def _as_utc_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize_annotation_summary(
    *,
    total_frames: int,
    reviewed_frames: int,
    pending_frames: int | None = None,
    state: str | None = None,
    last_updated_at: datetime | None = None,
) -> dict[str, Any]:
    """
    Normalize annotation summary values and infer state when missing.

    Returns a dict with ``total_frames``, ``reviewed_frames``, ``pending_frames``,
    ``state``, and ``last_updated`` (UTC Z string).
    """
    total = max(int(total_frames), 0)
    reviewed = max(int(reviewed_frames), 0)
    if total > 0:
        reviewed = min(reviewed, total)

    if pending_frames is None:
        pending = max(total - reviewed, 0)
    else:
        pending = max(int(pending_frames), 0)
        if total > 0:
            pending = min(pending, total - reviewed)

    normalized_state = state if state in ANNOTATION_STATES else None
    if normalized_state is None:
        if total == 0:
            normalized_state = "available"
        elif pending == 0:
            normalized_state = "completed"
        elif reviewed == 0:
            normalized_state = "preprocessing"
        else:
            normalized_state = "in_review"

    updated_at = last_updated_at or datetime.now(timezone.utc)

    return {
        "total_frames": total,
        "reviewed_frames": reviewed,
        "pending_frames": pending,
        "state": normalized_state,
        "last_updated": _as_utc_z(updated_at),
    }


def update_observation_annotation_summary(observation: Any, summary: dict[str, Any]) -> dict[str, Any]:
    """
    Merge annotation summary into an observation's extra metadata.

    The observation must expose ``merge_extra`` to persist into ``extra_json``.
    """
    if not hasattr(observation, "merge_extra"):
        raise TypeError("Observation must support merge_extra().")
    return observation.merge_extra({"annotation_summary": summary})


def write_annotation_manifest(
    observation_id: int | str,
    summary: dict[str, Any],
    base_dir: Any | None = None,
) -> None:
    """
    Persist annotation summary to the manifest on disk.
    """
    write_manifest(observation_id, summary, base_dir)


def persist_annotation_summary(
    observation: Any,
    observation_id: int | str,
    summary: dict[str, Any],
    base_dir: Any | None = None,
) -> dict[str, Any]:
    """
    Persist annotation summary to observation metadata and disk manifest.
    """
    merged = update_observation_annotation_summary(observation, summary)
    write_annotation_manifest(observation_id, summary, base_dir)
    return merged
