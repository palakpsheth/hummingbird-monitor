"""
Tests for schema module validation.

These tests cover the Pydantic schemas and their validation logic.
"""

from __future__ import annotations

import pytest

import hbmon.schema as schema


class TestRoiIn:
    """Tests for the RoiIn schema."""

    def test_roi_in_valid(self):
        """Test creating a valid RoiIn."""
        roi = schema.RoiIn(x1=0.1, y1=0.2, x2=0.8, y2=0.9)

        assert roi.x1 == 0.1
        assert roi.y1 == 0.2
        assert roi.x2 == 0.8
        assert roi.y2 == 0.9

    def test_roi_in_full_frame(self):
        """Test RoiIn covering full frame."""
        roi = schema.RoiIn(x1=0.0, y1=0.0, x2=1.0, y2=1.0)

        assert roi.x1 == 0.0
        assert roi.x2 == 1.0

    @pytest.mark.skipif(
        not getattr(schema, "_PYDANTIC_AVAILABLE", True),
        reason="Pydantic not available for validation tests"
    )
    def test_roi_in_x2_must_be_greater_than_x1(self):
        """Test that x2 must be greater than x1."""
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            schema.RoiIn(x1=0.5, y1=0.1, x2=0.3, y2=0.9)

    @pytest.mark.skipif(
        not getattr(schema, "_PYDANTIC_AVAILABLE", True),
        reason="Pydantic not available for validation tests"
    )
    def test_roi_in_y2_must_be_greater_than_y1(self):
        """Test that y2 must be greater than y1."""
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            schema.RoiIn(x1=0.1, y1=0.8, x2=0.9, y2=0.5)

    @pytest.mark.skipif(
        not getattr(schema, "_PYDANTIC_AVAILABLE", True),
        reason="Pydantic not available for validation tests"
    )
    def test_roi_in_coordinates_in_range(self):
        """Test that coordinates must be in [0, 1]."""
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            schema.RoiIn(x1=-0.1, y1=0.0, x2=0.5, y2=0.5)

    @pytest.mark.skipif(
        not getattr(schema, "_PYDANTIC_AVAILABLE", True),
        reason="Pydantic not available for validation tests"
    )
    def test_roi_in_coordinates_above_one(self):
        """Test that coordinates must be in [0, 1]."""
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            schema.RoiIn(x1=0.0, y1=0.0, x2=1.5, y2=0.5)


class TestRoiOut:
    """Tests for the RoiOut schema."""

    def test_roi_out_creation(self):
        """Test creating RoiOut."""
        roi = schema.RoiOut(x1=0.1, y1=0.2, x2=0.8, y2=0.9)

        assert roi.x1 == 0.1
        assert roi.y1 == 0.2
        assert roi.x2 == 0.8
        assert roi.y2 == 0.9


class TestHealthOut:
    """Tests for the HealthOut schema."""

    def test_health_out_basic(self):
        """Test creating basic HealthOut."""
        health = schema.HealthOut(
            ok=True,
            version="0.1.0",
            db_ok=True,
        )

        assert health.ok is True
        assert health.version == "0.1.0"
        assert health.db_ok is True
        assert health.last_observation_utc is None
        assert health.rtsp_url is None

    def test_health_out_with_optional_fields(self):
        """Test HealthOut with optional fields."""
        health = schema.HealthOut(
            ok=True,
            version="0.1.0",
            db_ok=True,
            last_observation_utc="2024-01-15T12:00:00Z",
            rtsp_url="rtsp://test:554/stream",
        )

        assert health.last_observation_utc == "2024-01-15T12:00:00Z"
        assert health.rtsp_url == "rtsp://test:554/stream"


class TestObservationOut:
    """Tests for the ObservationOut schema."""

    def test_observation_out_basic(self):
        """Test creating basic ObservationOut."""
        obs = schema.ObservationOut(
            id=1,
            ts_utc="2024-01-15T12:00:00Z",
            camera_name="hummingbirdcam",
            species_label="Anna's Hummingbird",
            species_prob=0.85,
            individual_id=5,
            match_score=0.92,
            snapshot_path="snapshots/2024-01-15/abc.jpg",
            video_path="clips/2024-01-15/abc.mp4",
        )

        assert obs.id == 1
        assert obs.ts_utc == "2024-01-15T12:00:00Z"
        assert obs.species_label == "Anna's Hummingbird"
        assert obs.species_prob == 0.85

    def test_observation_out_with_bbox(self):
        """Test ObservationOut with bbox."""
        obs = schema.ObservationOut(
            id=1,
            ts_utc="2024-01-15T12:00:00Z",
            camera_name="hummingbirdcam",
            species_label="Anna's Hummingbird",
            species_prob=0.85,
            individual_id=None,
            match_score=0.0,
            snapshot_path="test.jpg",
            video_path="test.mp4",
            bbox_xyxy=(10, 20, 100, 200),
        )

        assert obs.bbox_xyxy == (10, 20, 100, 200)

    def test_observation_out_with_extra(self):
        """Test ObservationOut with extra metadata."""
        extra_data = {"detection": {"confidence": 0.9}}
        obs = schema.ObservationOut(
            id=1,
            ts_utc="2024-01-15T12:00:00Z",
            camera_name="hummingbirdcam",
            species_label="Anna's Hummingbird",
            species_prob=0.85,
            individual_id=None,
            match_score=0.0,
            snapshot_path="test.jpg",
            video_path="test.mp4",
            extra=extra_data,
        )

        assert obs.extra == extra_data


class TestIndividualOut:
    """Tests for the IndividualOut schema."""

    def test_individual_out_basic(self):
        """Test creating basic IndividualOut."""
        ind = schema.IndividualOut(
            id=1,
            name="Ruby",
            visit_count=42,
            created_utc="2024-01-01T00:00:00Z",
            last_seen_utc="2024-01-15T12:00:00Z",
        )

        assert ind.id == 1
        assert ind.name == "Ruby"
        assert ind.visit_count == 42
        assert ind.created_utc == "2024-01-01T00:00:00Z"
        assert ind.last_seen_utc == "2024-01-15T12:00:00Z"

    def test_individual_out_with_species(self):
        """Test IndividualOut with species label."""
        ind = schema.IndividualOut(
            id=1,
            name="Ruby",
            visit_count=42,
            created_utc="2024-01-01T00:00:00Z",
            last_seen_utc="2024-01-15T12:00:00Z",
            last_species_label="Anna's Hummingbird",
        )

        assert ind.last_species_label == "Anna's Hummingbird"

    def test_individual_out_null_last_seen(self):
        """Test IndividualOut with null last_seen."""
        ind = schema.IndividualOut(
            id=1,
            name="(unnamed)",
            visit_count=0,
            created_utc="2024-01-01T00:00:00Z",
            last_seen_utc=None,
        )

        assert ind.last_seen_utc is None


class TestPydanticAvailability:
    """Tests for checking Pydantic availability flag."""

    def test_pydantic_available_flag_exists(self):
        """Test that _PYDANTIC_AVAILABLE flag is defined."""
        assert hasattr(schema, "_PYDANTIC_AVAILABLE")
        assert isinstance(schema._PYDANTIC_AVAILABLE, bool)
