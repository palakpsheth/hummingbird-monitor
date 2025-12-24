"""
Additional tests for clustering module functions.

These tests cover the remaining clustering utilities and data structures.
"""

from __future__ import annotations

import numpy as np
import pytest

import hbmon.clustering as clustering


class TestEnvHelpers:
    """Tests for environment variable helper functions."""

    def test_get_env_float_with_value(self, monkeypatch):
        """Test _get_env_float returns value when set."""
        monkeypatch.setenv("TEST_FLOAT", "3.14")
        result = clustering._get_env_float("TEST_FLOAT", 1.0)
        assert result == 3.14

    def test_get_env_float_missing(self, monkeypatch):
        """Test _get_env_float returns default when missing."""
        monkeypatch.delenv("TEST_FLOAT_MISSING", raising=False)
        result = clustering._get_env_float("TEST_FLOAT_MISSING", 2.5)
        assert result == 2.5

    def test_get_env_float_empty(self, monkeypatch):
        """Test _get_env_float returns default when empty."""
        monkeypatch.setenv("TEST_FLOAT_EMPTY", "")
        result = clustering._get_env_float("TEST_FLOAT_EMPTY", 1.5)
        assert result == 1.5

    def test_get_env_float_invalid(self, monkeypatch):
        """Test _get_env_float returns default on invalid value."""
        monkeypatch.setenv("TEST_FLOAT_INVALID", "not_a_float")
        result = clustering._get_env_float("TEST_FLOAT_INVALID", 1.0)
        assert result == 1.0

    def test_get_env_int_with_value(self, monkeypatch):
        """Test _get_env_int returns value when set."""
        monkeypatch.setenv("TEST_INT", "42")
        result = clustering._get_env_int("TEST_INT", 10)
        assert result == 42

    def test_get_env_int_missing(self, monkeypatch):
        """Test _get_env_int returns default when missing."""
        monkeypatch.delenv("TEST_INT_MISSING", raising=False)
        result = clustering._get_env_int("TEST_INT_MISSING", 5)
        assert result == 5

    def test_get_env_int_empty(self, monkeypatch):
        """Test _get_env_int returns default when empty."""
        monkeypatch.setenv("TEST_INT_EMPTY", "")
        result = clustering._get_env_int("TEST_INT_EMPTY", 3)
        assert result == 3

    def test_get_env_int_invalid(self, monkeypatch):
        """Test _get_env_int returns default on invalid value."""
        monkeypatch.setenv("TEST_INT_INVALID", "not_an_int")
        result = clustering._get_env_int("TEST_INT_INVALID", 7)
        assert result == 7


class TestMatchResult:
    """Tests for the MatchResult dataclass."""

    def test_match_result_creation(self):
        """Test creating a MatchResult."""
        result = clustering.MatchResult(
            matched=True,
            individual_id=5,
            distance=0.15,
            similarity=0.85,
        )

        assert result.matched is True
        assert result.individual_id == 5
        assert result.distance == 0.15
        assert result.similarity == 0.85


class TestIndividualState:
    """Tests for the IndividualState dataclass."""

    def test_individual_state_creation(self):
        """Test creating an IndividualState."""
        proto = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        state = clustering.IndividualState(
            individual_id=1,
            prototype=proto,
            visit_count=5,
        )

        assert state.individual_id == 1
        np.testing.assert_array_equal(state.prototype, proto)
        assert state.visit_count == 5
        assert state.pending is None

    def test_ensure_pending_creates_list(self):
        """Test ensure_pending creates empty list."""
        proto = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        state = clustering.IndividualState(individual_id=1, prototype=proto)

        assert state.pending is None
        state.ensure_pending()
        assert state.pending == []

    def test_ensure_pending_idempotent(self):
        """Test ensure_pending is idempotent."""
        proto = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        state = clustering.IndividualState(individual_id=1, prototype=proto)

        state.ensure_pending()
        state.pending.append(np.zeros(3))
        state.ensure_pending()

        assert len(state.pending) == 1

    def test_add_pending_appends(self):
        """Test add_pending appends embeddings."""
        proto = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        state = clustering.IndividualState(individual_id=1, prototype=proto)

        emb1 = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 0.5, 0.5], dtype=np.float32)

        state.add_pending(emb1)
        state.add_pending(emb2)

        assert len(state.pending) == 2

    def test_add_pending_drops_oldest_when_full(self, monkeypatch):
        """Test add_pending drops oldest when buffer is full."""
        monkeypatch.setattr(clustering, "MAX_PENDING", 3)

        proto = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        state = clustering.IndividualState(individual_id=1, prototype=proto)

        for i in range(5):
            state.add_pending(np.array([float(i), 0.0, 0.0], dtype=np.float32))

        assert len(state.pending) == 3
        # First two should have been dropped
        assert state.pending[0][0] == 2.0


class TestChooseIndividual:
    """Tests for the choose_individual function."""

    def test_choose_individual_empty_list(self):
        """Test choose_individual returns None for empty list."""
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = clustering.choose_individual(emb, [])

        assert result is None

    def test_choose_individual_single_match(self):
        """Test choose_individual matches single individual."""
        proto = clustering.l2_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        state = clustering.IndividualState(individual_id=1, prototype=proto)

        emb = clustering.l2_normalize(np.array([0.9, 0.1, 0.0], dtype=np.float32))
        result = clustering.choose_individual(emb, [state], threshold=0.5)

        assert result is not None
        assert result.matched is True
        assert result.individual_id == 1

    def test_choose_individual_no_match(self):
        """Test choose_individual with no match."""
        proto = clustering.l2_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        state = clustering.IndividualState(individual_id=1, prototype=proto)

        # Completely different direction
        emb = clustering.l2_normalize(np.array([-1.0, 0.0, 0.0], dtype=np.float32))
        result = clustering.choose_individual(emb, [state], threshold=0.1)

        assert result is not None
        assert result.matched is False


class TestUpdatePrototypeEma:
    """Tests for the update_prototype_ema function."""

    def test_update_prototype_ema_small_alpha(self):
        """Test EMA update with small alpha."""
        proto = clustering.l2_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        emb = clustering.l2_normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))

        result = clustering.update_prototype_ema(proto, emb, alpha=0.1)

        # Result should be mostly in original direction
        assert result[0] > result[1]
        # Should be normalized
        assert np.isclose(np.linalg.norm(result), 1.0)

    def test_update_prototype_ema_large_alpha(self):
        """Test EMA update with large alpha."""
        proto = clustering.l2_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        emb = clustering.l2_normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))

        result = clustering.update_prototype_ema(proto, emb, alpha=0.9)

        # Result should be mostly in new direction
        assert result[1] > result[0]


class TestSplitSuggestion:
    """Tests for the SplitSuggestion dataclass."""

    def test_split_suggestion_not_ok(self):
        """Test SplitSuggestion with ok=False."""
        sugg = clustering.SplitSuggestion(
            ok=False,
            reason="not enough samples",
        )

        assert sugg.ok is False
        assert sugg.labels is None
        assert sugg.centroid_a is None

    def test_split_suggestion_ok(self):
        """Test SplitSuggestion with ok=True."""
        ca = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        cb = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        sugg = clustering.SplitSuggestion(
            ok=True,
            reason="clusters found",
            labels=["A", "B", "A"],
            centroid_a=ca,
            centroid_b=cb,
        )

        assert sugg.ok is True
        assert sugg.labels == ["A", "B", "A"]


class TestSuggestSplitTwoGroups:
    """Tests for the suggest_split_two_groups function."""

    def test_suggest_split_not_enough_samples(self):
        """Test suggest_split with too few samples."""
        embs = [np.random.randn(10).astype(np.float32) for _ in range(5)]
        result = clustering.suggest_split_two_groups(embs, min_samples=10)

        assert result.ok is False
        assert "samples" in result.reason.lower()

    def test_suggest_split_with_separable_clusters(self):
        """Test suggest_split with clearly separable clusters."""
        # Skip if sklearn not available
        if clustering.KMeans is None:
            pytest.skip("sklearn not available")

        # Create two clearly separable groups
        group_a = [clustering.l2_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32) + np.random.randn(3) * 0.05)
                   for _ in range(10)]
        group_b = [clustering.l2_normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32) + np.random.randn(3) * 0.05)
                   for _ in range(10)]

        embs = group_a + group_b
        result = clustering.suggest_split_two_groups(embs, min_samples=12)

        assert result.ok is True
        assert result.labels is not None
        assert len(result.labels) == 20
        assert result.centroid_a is not None
        assert result.centroid_b is not None


class TestAssignToCentroid:
    """Tests for the assign_to_centroid function."""

    def test_assign_to_centroid_a(self):
        """Test assignment to centroid A."""
        ca = clustering.l2_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        cb = clustering.l2_normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))

        emb = clustering.l2_normalize(np.array([0.9, 0.1, 0.0], dtype=np.float32))
        result = clustering.assign_to_centroid(emb, ca, cb)

        assert result == "A"

    def test_assign_to_centroid_b(self):
        """Test assignment to centroid B."""
        ca = clustering.l2_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        cb = clustering.l2_normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))

        emb = clustering.l2_normalize(np.array([0.1, 0.9, 0.0], dtype=np.float32))
        result = clustering.assign_to_centroid(emb, ca, cb)

        assert result == "B"

    def test_assign_to_centroid_ties_to_a(self):
        """Test that ties go to A."""
        ca = clustering.l2_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        cb = clustering.l2_normalize(np.array([-1.0, 0.0, 0.0], dtype=np.float32))

        # Equidistant from both
        emb = clustering.l2_normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        result = clustering.assign_to_centroid(emb, ca, cb)

        assert result == "A"  # Ties go to A


class TestL2Normalize:
    """Additional tests for l2_normalize function."""

    def test_l2_normalize_returns_unit_vector(self):
        """Test that l2_normalize returns a unit vector."""
        vec = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        result = clustering.l2_normalize(vec)

        assert np.isclose(np.linalg.norm(result), 1.0)

    def test_l2_normalize_zero_vector(self):
        """Test l2_normalize with near-zero vector."""
        vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        result = clustering.l2_normalize(vec)

        # Should not produce NaN due to eps
        assert not np.any(np.isnan(result))


class TestCosineMetrics:
    """Additional tests for cosine similarity/distance."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        a = clustering.l2_normalize(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = a.copy()

        assert np.isclose(clustering.cosine_similarity(a, b), 1.0)

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        assert np.isclose(clustering.cosine_similarity(a, b), 0.0)

    def test_cosine_distance_opposite(self):
        """Test cosine distance of opposite vectors."""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        assert np.isclose(clustering.cosine_distance(a, b), 2.0)
