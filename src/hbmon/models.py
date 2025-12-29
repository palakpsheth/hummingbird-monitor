# src/hbmon/models.py
"""
SQLAlchemy ORM models for hbmon.

Tables:
- individuals: 1 row per inferred hummingbird individual (cluster)
- observations: 1 row per detection event (snapshot + clip + labels)
- candidates: motion-rejected detections for review/labeling
- embeddings: optional, stores per-observation embedding vectors (compressed blob)

Rationale:
- We keep a prototype embedding on the Individual row for fast matching.
- We optionally store per-observation embeddings for split-review + debugging.
  (Can be disabled in worker code if desired.)
"""

from __future__ import annotations

"""
Models for the hummingbird monitor.

This module attempts to gracefully handle the absence of SQLAlchemy so that it
can be imported in environments where that dependency is not installed.  When
SQLAlchemy is present, full ORM models are defined; when it is missing, a
lightweight set of dataclass-based stubs is provided.  The stubs implement
enough of the API for unit tests that do not rely on a real database.  Any
attempt to use database features without SQLAlchemy will raise a clear
``RuntimeError`` at runtime.

The ``_SQLALCHEMY_AVAILABLE`` flag below can be inspected to determine if
SQLAlchemy is available.
"""

import json
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional

import numpy as np

try:
    # Attempt to import SQLAlchemy.  If it is unavailable this import will
    # raise ImportError and we will fall back to stubs.
    from sqlalchemy import (
        DateTime,
        Float,
        ForeignKey,
        Integer,
        LargeBinary,
        String,
        Text,
        UniqueConstraint,
        Index,
    )
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship  # type: ignore
    _SQLALCHEMY_AVAILABLE = True
except Exception:  # pragma: no cover - executed when SQLAlchemy missing
    DateTime = Float = ForeignKey = Integer = LargeBinary = String = Text = UniqueConstraint = Index = None  # type: ignore
    DeclarativeBase = object  # type: ignore
    Mapped = mapped_column = relationship = None  # type: ignore
    _SQLALCHEMY_AVAILABLE = False


__all__ = [
    "_SQLALCHEMY_AVAILABLE",
    "Base",
    "Individual",
    "Observation",
    "Candidate",
    "Embedding",
    "_pack_embedding",
    "_unpack_embedding",
]


# ----------------------------
# Base
# ----------------------------


def _deep_merge(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """
    Return a new dict containing a deep merge of ``base`` and ``new``.
    Nested dicts are merged recursively; non-dict values overwrite existing ones.
    Neither input dict is mutated.
    """
    merged: dict[str, Any] = dict(base)
    for k, v in new.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _extract_review_label(extra: Any) -> str | None:
    """
    Pull the review label from ``extra`` if present.

    Args:
        extra: Parsed extra metadata (expected dict).

    Returns:
        The review label string or ``None`` if missing/invalid.
    """
    if not extra or not isinstance(extra, dict):
        return None
    review = extra.get("review")
    if not isinstance(review, dict):
        return None
    label = review.get("label")
    if not label:
        return None
    label_value = str(label)
    if label_value == "false_negative":
        return "unknown"
    return label_value

# ---------------------------------------------------------------------------
# SQLAlchemy base or stub
# ---------------------------------------------------------------------------
if _SQLALCHEMY_AVAILABLE:
    class Base(DeclarativeBase):
        """Base class for SQLAlchemy ORM models."""
        pass
else:
    class Base:
        """
        Fallback base class used when SQLAlchemy is unavailable.  This class
        exists solely to allow type checking and attribute access without
        raising ImportError at import time.  It should not be used for any
        database operations.  Methods that rely on SQLAlchemy will raise
        ``RuntimeError`` at runtime.
        """
        pass


def utcnow() -> datetime:
    """Return the current UTC timestamp.  Separated into its own function for
    easier testing/mocking.
    """
    return datetime.now(timezone.utc)


def _to_utc(dt: datetime) -> datetime:
    """
    Normalize a datetime to UTC, assuming naive values are already UTC.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ----------------------------
# Utilities for embedding blobs
# ----------------------------

def _pack_embedding(vec: np.ndarray) -> bytes:
    """
    Pack an embedding vector to a compressed bytes blob.
    Assumes float32 1D array.
    """
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    raw = v.tobytes(order="C")
    return zlib.compress(raw, level=6)


def _unpack_embedding(blob: bytes) -> np.ndarray:
    raw = zlib.decompress(blob)
    arr = np.frombuffer(raw, dtype=np.float32)
    return arr



# ---------------------------------------------------------------------------
# ORM models or stubs
# ---------------------------------------------------------------------------

if _SQLALCHEMY_AVAILABLE:
    # -----------------------------------------------------------------------
    # SQLAlchemy-backed models
    # -----------------------------------------------------------------------
    class Individual(Base):
        __tablename__ = "individuals"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

        # User-editable
        name: Mapped[str] = mapped_column(String(128), nullable=False, default="(unnamed)")

        # Stats
        visit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
        created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
        last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

        # Prototype embedding (compressed). Null until first embedding assigned.
        prototype_blob: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

        # Optional label hint (not authoritative)
        last_species_label: Mapped[str | None] = mapped_column(String(128), nullable=True)

        # Relationships
        observations: Mapped[list["Observation"]] = relationship(
            back_populates="individual",
            cascade="all, delete-orphan",
            passive_deletes=True,
        )

        # ---------- Convenience ----------

        def set_prototype(self, vec: np.ndarray) -> None:
            self.prototype_blob = _pack_embedding(vec)

        def get_prototype(self) -> np.ndarray | None:
            if self.prototype_blob is None:
                return None
            return _unpack_embedding(self.prototype_blob)

        @property
        def last_seen_utc(self) -> str | None:
            if self.last_seen_at is None:
                return None
            return _to_utc(self.last_seen_at).isoformat(timespec="seconds").replace("+00:00", "Z")


    class Observation(Base):
        __tablename__ = "observations"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

        # When it happened (UTC)
        ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True, default=utcnow)

        # Camera info
        camera_name: Mapped[str | None] = mapped_column(String(128), nullable=True)

        # Species prediction
        species_label: Mapped[str] = mapped_column(String(128), nullable=False, default="Hummingbird (unknown species)")
        species_prob: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

        # Individual match
        individual_id: Mapped[int] = mapped_column(
            Integer,
            ForeignKey("individuals.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        )
        match_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)  # similarity (1 - dist)

        # Detection bounding box (pixel coords in original frame)
        bbox_x1: Mapped[int | None] = mapped_column(Integer, nullable=True)
        bbox_y1: Mapped[int | None] = mapped_column(Integer, nullable=True)
        bbox_x2: Mapped[int | None] = mapped_column(Integer, nullable=True)
        bbox_y2: Mapped[int | None] = mapped_column(Integer, nullable=True)

        # Media (relative to /media mount)
        snapshot_path: Mapped[str] = mapped_column(String(512), nullable=False)
        video_path: Mapped[str] = mapped_column(String(512), nullable=False)

        # Extra JSON metadata (e.g., detector outputs)
        extra_json: Mapped[str | None] = mapped_column(Text, nullable=True)

        # Relationship
        individual: Mapped["Individual | None"] = relationship(back_populates="observations")

        # ---------- Convenience ----------

        @property
        def ts_utc(self) -> str:
            return _to_utc(self.ts).isoformat(timespec="seconds").replace("+00:00", "Z")

        @property
        def bbox_xyxy(self) -> tuple[int, int, int, int] | None:
            if None in (self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2):
                return None
            return (int(self.bbox_x1), int(self.bbox_y1), int(self.bbox_x2), int(self.bbox_y2))

        @property
        def bbox_str(self) -> str | None:
            b = self.bbox_xyxy
            if b is None:
                return None
            return f"{b[0]},{b[1]},{b[2]},{b[3]}"

        def set_extra(self, d: dict[str, Any]) -> None:
            self.extra_json = json.dumps(d, sort_keys=True)

        def get_extra(self) -> dict[str, Any] | None:
            if not self.extra_json:
                return None
            try:
                obj = json.loads(self.extra_json)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return None
            return None

        def merge_extra(self, updates: dict[str, Any]) -> dict[str, Any]:
            """
            Deep-merge ``updates`` into existing extra metadata and persist.
            Nested dicts are merged recursively; other values overwrite.
            """
            base_dict: dict[str, Any] = self.get_extra() or {}
            if not isinstance(base_dict, dict):
                base_dict = {}
            merged = _deep_merge(base_dict, updates)
            self.set_extra(merged)
            return merged

        @property
        def review_label(self) -> str | None:
            return _extract_review_label(self.get_extra())


    class Candidate(Base):
        __tablename__ = "candidates"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

        # When it happened (UTC)
        ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True, default=utcnow)

        # Camera info
        camera_name: Mapped[str | None] = mapped_column(String(128), nullable=True)

        # Detection bounding box (pixel coords in original frame)
        bbox_x1: Mapped[int | None] = mapped_column(Integer, nullable=True)
        bbox_y1: Mapped[int | None] = mapped_column(Integer, nullable=True)
        bbox_x2: Mapped[int | None] = mapped_column(Integer, nullable=True)
        bbox_y2: Mapped[int | None] = mapped_column(Integer, nullable=True)

        # Media (relative to /media mount)
        snapshot_path: Mapped[str] = mapped_column(String(512), nullable=False)
        annotated_snapshot_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
        mask_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
        mask_overlay_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
        clip_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

        # Extra JSON metadata (e.g., detector outputs)
        extra_json: Mapped[str | None] = mapped_column(Text, nullable=True)

        @property
        def ts_utc(self) -> str:
            return _to_utc(self.ts).isoformat(timespec="seconds").replace("+00:00", "Z")

        @property
        def bbox_xyxy(self) -> tuple[int, int, int, int] | None:
            if None in (self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2):
                return None
            return (int(self.bbox_x1), int(self.bbox_y1), int(self.bbox_x2), int(self.bbox_y2))

        def set_extra(self, d: dict[str, Any]) -> None:
            self.extra_json = json.dumps(d, sort_keys=True)

        def get_extra(self) -> dict[str, Any] | None:
            if not self.extra_json:
                return None
            try:
                obj = json.loads(self.extra_json)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return None
            return None

        def merge_extra(self, updates: dict[str, Any]) -> dict[str, Any]:
            """
            Deep-merge ``updates`` into existing extra metadata and persist.
            Nested dicts are merged recursively; other values overwrite.
            """
            base_dict: dict[str, Any] = self.get_extra() or {}
            if not isinstance(base_dict, dict):
                base_dict = {}
            merged = _deep_merge(base_dict, updates)
            self.set_extra(merged)
            return merged


    class Embedding(Base):
        """
        Optional per-observation embeddings (for debugging/split tools).

        These can grow the DB; you may choose to not store them, but the schema
        supports it.
        """
        __tablename__ = "embeddings"
        __table_args__ = (
            UniqueConstraint("observation_id", name="uq_embedding_observation"),
            Index("ix_embeddings_individual_id", "individual_id"),
        )

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

        observation_id: Mapped[int] = mapped_column(
            Integer,
            ForeignKey("observations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )

        # Redundant for convenience queries
        individual_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

        # Compressed float32 bytes
        embedding_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

        # Light metadata
        created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

        def set_vec(self, vec: np.ndarray) -> None:
            self.embedding_blob = _pack_embedding(vec)

        def get_vec(self) -> np.ndarray:
            return _unpack_embedding(self.embedding_blob)
else:
    # -----------------------------------------------------------------------
    # Dataclass-based stubs when SQLAlchemy is unavailable
    # -----------------------------------------------------------------------
    @dataclass
    class Individual(Base):
        """
        Lightweight stand-in for the ``Individual`` model used when SQLAlchemy
        is unavailable.  Provides the same public API as the ORM model but does
        not persist anything.  Relationships are represented as simple lists.
        """
        id: int
        name: str = "(unnamed)"
        visit_count: int = 0
        created_at: datetime = field(default_factory=utcnow)
        last_seen_at: Optional[datetime] = None
        prototype_blob: Optional[bytes] = None
        last_species_label: Optional[str] = None
        observations: List["Observation"] = field(default_factory=list)

        def set_prototype(self, vec: np.ndarray) -> None:
            self.prototype_blob = _pack_embedding(vec)

        def get_prototype(self) -> Optional[np.ndarray]:
            if self.prototype_blob is None:
                return None
            return _unpack_embedding(self.prototype_blob)

        @property
        def last_seen_utc(self) -> Optional[str]:
            if self.last_seen_at is None:
                return None
            return _to_utc(self.last_seen_at).isoformat(timespec="seconds").replace("+00:00", "Z")


    @dataclass
    class Observation(Base):
        """
        Lightweight stand-in for the ``Observation`` model used when SQLAlchemy
        is unavailable.  This class mimics the attribute names of the ORM
        version but stores data in plain Python fields.
        """
        id: int
        ts: datetime = field(default_factory=utcnow)
        camera_name: Optional[str] = None
        species_label: str = "Hummingbird (unknown species)"
        species_prob: float = 0.0
        individual_id: Optional[int] = None
        match_score: float = 0.0
        bbox_x1: Optional[int] = None
        bbox_y1: Optional[int] = None
        bbox_x2: Optional[int] = None
        bbox_y2: Optional[int] = None
        snapshot_path: str = ""
        video_path: str = ""
        extra_json: Optional[str] = None
        individual: Optional[Individual] = None

        @property
        def ts_utc(self) -> str:
            return _to_utc(self.ts).isoformat(timespec="seconds").replace("+00:00", "Z")

        @property
        def bbox_xyxy(self) -> Optional[tuple[int, int, int, int]]:
            if None in (self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2):
                return None
            return (int(self.bbox_x1), int(self.bbox_y1), int(self.bbox_x2), int(self.bbox_y2))

        @property
        def bbox_str(self) -> Optional[str]:
            b = self.bbox_xyxy
            if b is None:
                return None
            return f"{b[0]},{b[1]},{b[2]},{b[3]}"

        def set_extra(self, d: dict[str, Any]) -> None:
            self.extra_json = json.dumps(d, sort_keys=True)

        def get_extra(self) -> Optional[dict[str, Any]]:
            if not self.extra_json:
                return None
            try:
                obj = json.loads(self.extra_json)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return None
            return None

        def merge_extra(self, updates: dict[str, Any]) -> dict[str, Any]:
            """
            Deep-merge ``updates`` into existing extra metadata and persist.
            Nested dicts are merged recursively; other values overwrite.
            """
            base_dict: dict[str, Any] = self.get_extra() or {}
            if not isinstance(base_dict, dict):
                base_dict = {}
            merged = _deep_merge(base_dict, updates)
            self.set_extra(merged)
            return merged

        @property
        def review_label(self) -> str | None:
            return _extract_review_label(self.get_extra())


    @dataclass
    class Candidate(Base):
        """
        Lightweight stand-in for the ``Candidate`` model used when SQLAlchemy
        is unavailable.
        """
        id: int
        ts: datetime = field(default_factory=utcnow)
        camera_name: Optional[str] = None
        bbox_x1: Optional[int] = None
        bbox_y1: Optional[int] = None
        bbox_x2: Optional[int] = None
        bbox_y2: Optional[int] = None
        snapshot_path: str = ""
        annotated_snapshot_path: Optional[str] = None
        mask_path: Optional[str] = None
        mask_overlay_path: Optional[str] = None
        clip_path: Optional[str] = None
        extra_json: Optional[str] = None

        @property
        def ts_utc(self) -> str:
            return _to_utc(self.ts).isoformat(timespec="seconds").replace("+00:00", "Z")

        @property
        def bbox_xyxy(self) -> Optional[tuple[int, int, int, int]]:
            if None in (self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2):
                return None
            return (int(self.bbox_x1), int(self.bbox_y1), int(self.bbox_x2), int(self.bbox_y2))

        def set_extra(self, d: dict[str, Any]) -> None:
            self.extra_json = json.dumps(d, sort_keys=True)

        def get_extra(self) -> Optional[dict[str, Any]]:
            if not self.extra_json:
                return None
            try:
                obj = json.loads(self.extra_json)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return None
            return None

        def merge_extra(self, updates: dict[str, Any]) -> dict[str, Any]:
            """
            Deep-merge ``updates`` into existing extra metadata and persist.
            Nested dicts are merged recursively; other values overwrite.
            """
            base_dict: dict[str, Any] = self.get_extra() or {}
            if not isinstance(base_dict, dict):
                base_dict = {}
            merged = _deep_merge(base_dict, updates)
            self.set_extra(merged)
            return merged


    @dataclass
    class Embedding(Base):
        """
        Lightweight stand-in for the ``Embedding`` model used when SQLAlchemy is
        unavailable.  Stores embeddings in-memory only.
        """
        id: int
        observation_id: int
        individual_id: Optional[int] = None
        embedding_blob: bytes = b""
        created_at: datetime = field(default_factory=utcnow)

        def set_vec(self, vec: np.ndarray) -> None:
            self.embedding_blob = _pack_embedding(vec)

        def get_vec(self) -> np.ndarray:
            return _unpack_embedding(self.embedding_blob)
