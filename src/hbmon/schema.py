# src/hbmon/schema.py
"""
Pydantic schemas for hbmon API + template context.

These schemas are used for:
- JSON endpoints (ROI, health, etc.)
- Normalizing ROI submissions
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class RoiIn(BaseModel):
    """
    ROI submitted from the UI (normalized coords 0..1).
    """
    x1: float = Field(..., ge=0.0, le=1.0)
    y1: float = Field(..., ge=0.0, le=1.0)
    x2: float = Field(..., ge=0.0, le=1.0)
    y2: float = Field(..., ge=0.0, le=1.0)

    @field_validator("x2")
    @classmethod
    def _x_order(cls, v: float, info) -> float:
        # pydantic v2: info.data has previous fields
        x1 = float(info.data.get("x1", 0.0))
        if v <= x1:
            raise ValueError("x2 must be > x1")
        return v

    @field_validator("y2")
    @classmethod
    def _y_order(cls, v: float, info) -> float:
        y1 = float(info.data.get("y1", 0.0))
        if v <= y1:
            raise ValueError("y2 must be > y1")
        return v


class RoiOut(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class HealthOut(BaseModel):
    ok: bool
    version: str
    db_ok: bool
    last_observation_utc: str | None = None
    rtsp_url: str | None = None


class ObservationOut(BaseModel):
    id: int
    ts_utc: str
    camera_name: str | None

    species_label: str
    species_prob: float

    individual_id: int | None
    match_score: float

    snapshot_path: str
    video_path: str

    bbox_xyxy: tuple[int, int, int, int] | None = None
    extra: dict[str, Any] | None = None


class IndividualOut(BaseModel):
    id: int
    name: str
    visit_count: int
    created_utc: str
    last_seen_utc: str | None
    last_species_label: str | None = None
