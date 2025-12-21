# src/hbmon/models.py
"""
SQLAlchemy ORM models for hbmon.

Tables:
- individuals: 1 row per inferred hummingbird individual (cluster)
- observations: 1 row per detection event (snapshot + clip + labels)
- embeddings: optional, stores per-observation embedding vectors (compressed blob)

Rationale:
- We keep a prototype embedding on the Individual row for fast matching.
- We optionally store per-observation embeddings for split-review + debugging.
  (Can be disabled in worker code if desired.)
"""

from __future__ import annotations

import json
import zlib
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import (
    Boolean,
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
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ----------------------------
# Base
# ----------------------------

class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


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


# ----------------------------
# ORM Models
# ----------------------------

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
        return self.last_seen_at.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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
        return self.ts.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

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
