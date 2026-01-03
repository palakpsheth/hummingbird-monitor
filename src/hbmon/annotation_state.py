# src/hbmon/annotation_state.py
"""
Annotation state machine for pipeline workflow.

This module defines the annotation workflow states and provides helper functions
for state transitions, summary computation, and pipeline gating logic.

States:
    - pending: Observation has no annotation activity
    - preprocessing: Frames are being extracted and/or auto-annotated
    - in_review: Frames are ready for manual review
    - completed: All frames have been reviewed

Pipeline Rules:
    - Training is BLOCKED if any observation is "partial" (some frames reviewed, some not)
    - Training is ALLOWED if all observations are either "completed" or "pending"
    - Pending observations are EXCLUDED from the training dataset
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AnnotationState(Enum):
    """Annotation workflow states."""
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    IN_REVIEW = "in_review"
    COMPLETED = "completed"


@dataclass
class AnnotationProgress:
    """Progress summary for an observation's annotations."""
    observation_id: int
    state: AnnotationState
    total_frames: int
    reviewed_frames: int
    pending_frames: int
    last_updated: str | None = None
    is_partial: bool = False  # True if some (but not all) frames reviewed

    @property
    def progress_percent(self) -> float:
        """Calculate review progress percentage."""
        if self.total_frames == 0:
            return 0.0
        return (self.reviewed_frames / self.total_frames) * 100.0


def determine_state(total_frames: int, reviewed_frames: int) -> AnnotationState:
    """
    Determine the annotation state based on frame counts.
    
    Args:
        total_frames: Total number of extracted frames
        reviewed_frames: Number of frames that have been reviewed
        
    Returns:
        The appropriate AnnotationState
    """
    if total_frames == 0:
        return AnnotationState.PENDING
    elif reviewed_frames == 0:
        return AnnotationState.IN_REVIEW  # Frames exist but none reviewed
    elif reviewed_frames < total_frames:
        return AnnotationState.IN_REVIEW  # Partial review
    else:
        return AnnotationState.COMPLETED  # All frames reviewed


def is_partial(total_frames: int, reviewed_frames: int) -> bool:
    """
    Check if an observation has partial annotations (some but not all reviewed).
    
    Partial observations block the training pipeline.
    """
    return total_frames > 0 and 0 < reviewed_frames < total_frames


def can_include_in_training(state: AnnotationState) -> bool:
    """
    Check if an observation can be included in training dataset.
    
    Only completed observations are included.
    """
    return state == AnnotationState.COMPLETED


def parse_annotation_summary(extra: dict[str, Any] | None) -> AnnotationProgress | None:
    """
    Parse annotation summary from observation extra_json.
    
    Args:
        extra: The observation's extra_json dict
        
    Returns:
        AnnotationProgress if summary exists, None otherwise
    """
    if not extra or not isinstance(extra, dict):
        return None
    
    ann = extra.get("annotation_summary")
    if not ann or not isinstance(ann, dict):
        return None
    
    total = ann.get("total_frames", 0) or 0
    reviewed = ann.get("reviewed_frames", 0) or 0
    pending = ann.get("pending_frames", 0) or total - reviewed
    state_str = ann.get("state", "pending")
    last_updated = ann.get("last_updated")
    
    try:
        state = AnnotationState(state_str)
    except ValueError:
        state = determine_state(total, reviewed)
    
    return AnnotationProgress(
        observation_id=0,  # Caller should set this
        state=state,
        total_frames=total,
        reviewed_frames=reviewed,
        pending_frames=pending,
        last_updated=last_updated,
        is_partial=is_partial(total, reviewed),
    )


def compute_pipeline_eligibility(
    observations: list[AnnotationProgress],
) -> tuple[bool, list[int], list[int], str]:
    """
    Compute whether the training pipeline can be triggered.
    
    Args:
        observations: List of annotation progress for all observations
        
    Returns:
        Tuple of:
        - can_train: Whether training can be started
        - included_ids: IDs of observations to include in training
        - excluded_ids: IDs of observations to exclude (pending)
        - reason: Human-readable explanation
    """
    included_ids = []
    excluded_ids = []
    partial_ids = []
    
    for obs in observations:
        if obs.state == AnnotationState.COMPLETED:
            included_ids.append(obs.observation_id)
        elif obs.is_partial:
            partial_ids.append(obs.observation_id)
        else:
            excluded_ids.append(obs.observation_id)
    
    if partial_ids:
        return (
            False,
            [],
            [],
            f"Training blocked: {len(partial_ids)} observation(s) have partial annotations. "
            f"Complete or reset them before training.",
        )
    
    if not included_ids:
        return (
            False,
            [],
            excluded_ids,
            "Training blocked: No completed observations available for training.",
        )
    
    return (
        True,
        included_ids,
        excluded_ids,
        f"Ready for training: {len(included_ids)} observation(s) included, "
        f"{len(excluded_ids)} pending observation(s) excluded.",
    )


def create_annotation_summary_dict(
    total_frames: int,
    reviewed_frames: int,
) -> dict[str, Any]:
    """
    Create an annotation_summary dict for storing in extra_json.
    
    Args:
        total_frames: Total number of frames
        reviewed_frames: Number of reviewed frames
        
    Returns:
        Dict suitable for merge_extra()
    """
    pending = total_frames - reviewed_frames
    state = determine_state(total_frames, reviewed_frames)
    
    return {
        "annotation_summary": {
            "total_frames": total_frames,
            "reviewed_frames": reviewed_frames,
            "pending_frames": pending,
            "state": state.value,
            "last_updated": datetime.now(timezone.utc).isoformat() + "Z",
        }
    }


def transition_to_preprocessing(obs_id: int) -> dict[str, Any]:
    """Create summary dict for preprocessing state."""
    return {
        "annotation_summary": {
            "total_frames": 0,
            "reviewed_frames": 0,
            "pending_frames": 0,
            "state": AnnotationState.PREPROCESSING.value,
            "last_updated": datetime.now(timezone.utc).isoformat() + "Z",
        }
    }


def transition_to_in_review(obs_id: int, total_frames: int) -> dict[str, Any]:
    """Create summary dict for in_review state after preprocessing."""
    return {
        "annotation_summary": {
            "total_frames": total_frames,
            "reviewed_frames": 0,
            "pending_frames": total_frames,
            "state": AnnotationState.IN_REVIEW.value,
            "last_updated": datetime.now(timezone.utc).isoformat() + "Z",
        }
    }
