# src/hbmon/schema.py
"""
Pydantic schemas for hbmon API + template context.

These schemas are used for:
- JSON endpoints (ROI, health, etc.)
- Normalizing ROI submissions
"""

from __future__ import annotations

"""
Pydantic schemas for hbmon API.

This module attempts to import Pydantic for schema validation.  If Pydantic
is not installed, it falls back to using dataclasses that mimic the
attributes of the Pydantic models but do not perform validation.  This
ensures that unit tests can run in environments without Pydantic installed.

The ``_PYDANTIC_AVAILABLE`` flag indicates whether Pydantic is available.
"""

from typing import Any, Optional

try:
    from pydantic import BaseModel, Field, field_validator  # type: ignore
    _PYDANTIC_AVAILABLE = True
except Exception:  # pragma: no cover
    BaseModel = None  # type: ignore
    def Field(default=..., **kwargs):
        return default  # type: ignore
    def field_validator(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator
    _PYDANTIC_AVAILABLE = False
    from dataclasses import dataclass as _dataclass


if _PYDANTIC_AVAILABLE:
    class RoiIn(BaseModel):
        """
        ROI submitted from the UI (normalized coords 0..1).  Uses Pydantic to
        enforce bounds and ordering on the coordinates when available.
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

    class SystemLoad(BaseModel):
        cpu: float
        mem: float
        gpu_intel: float | None = None
        gpu_nvidia: float | None = None

    class HealthOut(BaseModel):
        ok: bool
        version: str
        db_ok: bool
        last_observation_utc: str | None = None
        rtsp_url: str | None = None
        system_load: SystemLoad | None = None

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
else:
    # -------------------------------------------------------------------------
    # Dataclass-based stubs when Pydantic is unavailable
    # -------------------------------------------------------------------------
    @_dataclass
    class RoiIn:
        """
        Dataclass version of the ROI input schema used when Pydantic is not
        installed.  Coordinates are accepted without validation; users
        consuming this class should perform their own checks if necessary.
        """
        x1: float
        y1: float
        x2: float
        y2: float

    @_dataclass
    class RoiOut:
        x1: float
        y1: float
        x2: float
        y2: float

    @_dataclass
    class SystemLoad:
        cpu: float
        mem: float
        gpu_intel: Optional[float] = None
        gpu_nvidia: Optional[float] = None

    @_dataclass
    class HealthOut:
        ok: bool
        version: str
        db_ok: bool
        last_observation_utc: Optional[str] = None
        rtsp_url: Optional[str] = None
        system_load: Optional[SystemLoad] = None

    @_dataclass
    class ObservationOut:
        id: int
        ts_utc: str
        camera_name: Optional[str]
        species_label: str
        species_prob: float
        individual_id: Optional[int]
        match_score: float
        snapshot_path: str
        video_path: str
        bbox_xyxy: Optional[tuple[int, int, int, int]] = None
        extra: Optional[dict[str, Any]] = None

    @_dataclass
    class IndividualOut:
        id: int
        name: str
        visit_count: int
        created_utc: str
        last_seen_utc: Optional[str]
        last_species_label: Optional[str] = None
