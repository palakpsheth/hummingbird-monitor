"""
Tests for models module utility functions.

These tests cover helper functions and dataclass behaviors in models.py.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

import hbmon.models as models


class TestDeepMerge:
    """Tests for the _deep_merge function."""

    def test_deep_merge_simple(self):
        """Test simple dict merge."""
        base = {"a": 1, "b": 2}
        new = {"b": 3, "c": 4}

        result = models._deep_merge(base, new)

        assert result == {"a": 1, "b": 3, "c": 4}
        # Original dicts should not be modified
        assert base == {"a": 1, "b": 2}
        assert new == {"b": 3, "c": 4}

    def test_deep_merge_nested(self):
        """Test nested dict merge."""
        base = {
            "top": 1,
            "nested": {"a": 1, "b": 2},
        }
        new = {
            "nested": {"b": 3, "c": 4},
            "extra": 5,
        }

        result = models._deep_merge(base, new)

        assert result == {
            "top": 1,
            "nested": {"a": 1, "b": 3, "c": 4},
            "extra": 5,
        }

    def test_deep_merge_empty_base(self):
        """Test merging into empty dict."""
        result = models._deep_merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_deep_merge_empty_new(self):
        """Test merging empty dict."""
        result = models._deep_merge({"a": 1}, {})
        assert result == {"a": 1}

    def test_deep_merge_overwrite_non_dict(self):
        """Test that non-dict values are overwritten."""
        base = {"key": {"nested": 1}}
        new = {"key": "string_value"}

        result = models._deep_merge(base, new)
        assert result == {"key": "string_value"}

    def test_deep_merge_deeply_nested(self):
        """Test deeply nested merge."""
        base = {"l1": {"l2": {"l3": {"a": 1}}}}
        new = {"l1": {"l2": {"l3": {"b": 2}}}}

        result = models._deep_merge(base, new)
        assert result == {"l1": {"l2": {"l3": {"a": 1, "b": 2}}}}


class TestExtractReviewLabel:
    """Tests for the _extract_review_label function."""

    def test_extract_review_label_none(self):
        """Test with None input."""
        assert models._extract_review_label(None) is None

    def test_extract_review_label_not_dict(self):
        """Test with non-dict input."""
        assert models._extract_review_label("string") is None
        assert models._extract_review_label([1, 2, 3]) is None
        assert models._extract_review_label(123) is None

    def test_extract_review_label_no_review(self):
        """Test with dict missing review section."""
        assert models._extract_review_label({"other": "data"}) is None

    def test_extract_review_label_review_not_dict(self):
        """Test with review section that is not a dict."""
        assert models._extract_review_label({"review": "string"}) is None

    def test_extract_review_label_no_label(self):
        """Test with review dict missing label."""
        assert models._extract_review_label({"review": {"other": "data"}}) is None

    def test_extract_review_label_empty_label(self):
        """Test with empty label."""
        assert models._extract_review_label({"review": {"label": ""}}) is None

    def test_extract_review_label_valid(self):
        """Test with valid review label."""
        result = models._extract_review_label({
            "review": {"label": "true_positive"}
        })
        assert result == "true_positive"

    def test_extract_review_label_maps_false_negative(self):
        """Test that legacy false negative labels map to unknown."""
        result = models._extract_review_label({
            "review": {"label": "false_negative"}
        })
        assert result == "unknown"


class TestPackUnpackEmbedding:
    """Tests for the _pack_embedding and _unpack_embedding functions."""

    def test_pack_unpack_roundtrip(self):
        """Test that packing and unpacking preserves data."""
        original = np.array([1.0, 2.0, -3.5, 0.0], dtype=np.float32)

        packed = models._pack_embedding(original)
        assert isinstance(packed, bytes)
        assert len(packed) > 0

        unpacked = models._unpack_embedding(packed)
        np.testing.assert_array_almost_equal(unpacked, original)

    def test_pack_large_vector(self):
        """Test packing a large embedding vector."""
        original = np.random.randn(512).astype(np.float32)

        packed = models._pack_embedding(original)
        unpacked = models._unpack_embedding(packed)

        np.testing.assert_array_almost_equal(unpacked, original)

    def test_pack_converts_to_float32(self):
        """Test that pack converts to float32."""
        original = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        packed = models._pack_embedding(original)
        unpacked = models._unpack_embedding(packed)

        assert unpacked.dtype == np.float32

    def test_pack_flattens_multidimensional(self):
        """Test that pack flattens multi-dimensional arrays."""
        original = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        packed = models._pack_embedding(original)
        unpacked = models._unpack_embedding(packed)

        expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(unpacked, expected)


class TestToUtc:
    """Tests for the _to_utc function."""

    def test_to_utc_naive_datetime(self):
        """Test that naive datetime is treated as UTC."""
        naive = datetime(2024, 1, 15, 12, 0, 0)
        result = models._to_utc(naive)

        assert result.tzinfo == timezone.utc
        assert result.year == 2024
        assert result.hour == 12

    def test_to_utc_already_utc(self):
        """Test that UTC datetime is unchanged."""
        utc_dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = models._to_utc(utc_dt)

        assert result == utc_dt

    def test_to_utc_other_timezone(self):
        """Test conversion from other timezone to UTC."""
        from datetime import timedelta

        # Create a datetime in UTC+5
        tz_plus5 = timezone(timedelta(hours=5))
        dt = datetime(2024, 1, 15, 17, 0, 0, tzinfo=tz_plus5)

        result = models._to_utc(dt)

        assert result.tzinfo == timezone.utc
        assert result.hour == 12  # 17 - 5 = 12


class TestUtcnow:
    """Tests for the utcnow function in models."""

    def test_utcnow_returns_utc(self):
        """Test that utcnow returns a UTC datetime."""
        result = models.utcnow()

        assert isinstance(result, datetime)
        assert result.tzinfo == timezone.utc


class TestObservationStub:
    """Tests for the Observation dataclass stub."""

    def test_observation_ts_utc(self):
        """Test Observation.ts_utc property."""
        obs = models.Observation(
            id=1,
            ts=datetime(2024, 1, 15, 12, 30, 45, tzinfo=timezone.utc),
            snapshot_path="test.jpg",
            video_path="test.mp4",
        )

        assert obs.ts_utc == "2024-01-15T12:30:45Z"

    def test_observation_bbox_xyxy_all_set(self):
        """Test Observation.bbox_xyxy with all coordinates set."""
        obs = models.Observation(
            id=1,
            ts=datetime.now(timezone.utc),
            snapshot_path="test.jpg",
            video_path="test.mp4",
            bbox_x1=10,
            bbox_y1=20,
            bbox_x2=100,
            bbox_y2=200,
        )

        assert obs.bbox_xyxy == (10, 20, 100, 200)

    def test_observation_bbox_xyxy_partial(self):
        """Test Observation.bbox_xyxy with partial coordinates."""
        obs = models.Observation(
            id=1,
            ts=datetime.now(timezone.utc),
            snapshot_path="test.jpg",
            video_path="test.mp4",
            bbox_x1=10,
            bbox_y1=20,
            # x2, y2 are None
        )

        assert obs.bbox_xyxy is None

    def test_observation_bbox_str(self):
        """Test Observation.bbox_str property."""
        obs = models.Observation(
            id=1,
            ts=datetime.now(timezone.utc),
            snapshot_path="test.jpg",
            video_path="test.mp4",
            bbox_x1=10,
            bbox_y1=20,
            bbox_x2=100,
            bbox_y2=200,
        )

        assert obs.bbox_str == "10,20,100,200"

    def test_observation_bbox_str_none(self):
        """Test Observation.bbox_str when bbox is None."""
        obs = models.Observation(
            id=1,
            ts=datetime.now(timezone.utc),
            snapshot_path="test.jpg",
            video_path="test.mp4",
        )

        assert obs.bbox_str is None

    def test_observation_set_get_extra(self):
        """Test Observation.set_extra and get_extra."""
        obs = models.Observation(
            id=1,
            ts=datetime.now(timezone.utc),
            snapshot_path="test.jpg",
            video_path="test.mp4",
        )

        data = {"key": "value", "nested": {"a": 1}}
        obs.set_extra(data)

        result = obs.get_extra()
        assert result == data

    def test_observation_get_extra_invalid_json(self):
        """Test Observation.get_extra with invalid JSON."""
        obs = models.Observation(
            id=1,
            ts=datetime.now(timezone.utc),
            snapshot_path="test.jpg",
            video_path="test.mp4",
            extra_json="not valid json",
        )

        assert obs.get_extra() is None

    def test_observation_get_extra_non_dict(self):
        """Test Observation.get_extra with non-dict JSON."""
        obs = models.Observation(
            id=1,
            ts=datetime.now(timezone.utc),
            snapshot_path="test.jpg",
            video_path="test.mp4",
            extra_json='[1, 2, 3]',
        )

        assert obs.get_extra() is None

    def test_observation_merge_extra(self):
        """Test Observation.merge_extra for deep merge."""
        obs = models.Observation(
            id=1,
            ts=datetime.now(timezone.utc),
            snapshot_path="test.jpg",
            video_path="test.mp4",
        )

        obs.set_extra({"a": 1, "nested": {"b": 2}})
        result = obs.merge_extra({"nested": {"c": 3}, "d": 4})

        assert result == {"a": 1, "nested": {"b": 2, "c": 3}, "d": 4}
        assert obs.get_extra() == result

    def test_observation_review_label(self):
        """Test Observation.review_label property."""
        obs = models.Observation(
            id=1,
            ts=datetime.now(timezone.utc),
            snapshot_path="test.jpg",
            video_path="test.mp4",
        )

        obs.set_extra({"review": {"label": "true_positive"}})
        assert obs.review_label == "true_positive"

    def test_observation_review_label_none(self):
        """Test Observation.review_label when no label."""
        obs = models.Observation(
            id=1,
            ts=datetime.now(timezone.utc),
            snapshot_path="test.jpg",
            video_path="test.mp4",
        )

        assert obs.review_label is None


class TestIndividualStub:
    """Tests for the Individual dataclass stub."""

    def test_individual_set_get_prototype(self):
        """Test Individual.set_prototype and get_prototype."""
        ind = models.Individual(id=1)

        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ind.set_prototype(vec)

        result = ind.get_prototype()
        np.testing.assert_array_almost_equal(result, vec)

    def test_individual_get_prototype_none(self):
        """Test Individual.get_prototype when no prototype set."""
        ind = models.Individual(id=1)
        assert ind.get_prototype() is None

    def test_individual_last_seen_utc(self):
        """Test Individual.last_seen_utc property."""
        ind = models.Individual(
            id=1,
            last_seen_at=datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc),
        )

        assert ind.last_seen_utc == "2024-01-15T12:30:00Z"

    def test_individual_last_seen_utc_none(self):
        """Test Individual.last_seen_utc when not set."""
        ind = models.Individual(id=1)
        assert ind.last_seen_utc is None


class TestEmbeddingStub:
    """Tests for the Embedding dataclass stub."""

    def test_embedding_set_get_vec(self):
        """Test Embedding.set_vec and get_vec."""
        emb = models.Embedding(id=1, observation_id=1)

        vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        emb.set_vec(vec)

        result = emb.get_vec()
        np.testing.assert_array_almost_equal(result, vec)
