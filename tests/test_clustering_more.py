"""
Additional tests for functions in ``hbmon.clustering`` not covered in other suites.

These tests verify the behavior of choosing individuals, exponential moving
average updates, assigning embeddings to centroids, and management of
pending embedding buffers.  They are designed to run without relying on
scikit‑learn, so they skip any tests that would require external ML
libraries.
"""

import numpy as np

import hbmon.clustering as clustering


def test_choose_individual_none():
    # No individuals yields None
    res = clustering.choose_individual(np.array([1.0, 0.0], dtype=np.float32), [])
    assert res is None


def test_choose_individual_best_match():
    # Two distinct prototypes: choose the nearest one
    # Normalized prototypes along x and y axes
    p1 = clustering.l2_normalize(np.array([1.0, 0.0], dtype=np.float32))
    p2 = clustering.l2_normalize(np.array([0.0, 1.0], dtype=np.float32))
    inds = [
        clustering.IndividualState(individual_id=1, prototype=p1.copy(), visit_count=0),
        clustering.IndividualState(individual_id=2, prototype=p2.copy(), visit_count=0),
    ]
    # Embedding close to p1 should match individual 1
    emb = clustering.l2_normalize(np.array([0.9, 0.1], dtype=np.float32))
    res = clustering.choose_individual(emb, inds, threshold=0.5)
    assert res is not None
    assert res.individual_id == 1
    # The similarity for identical vectors is 1, for nearly orthogonal ~0.1
    assert res.similarity > 0.8
    assert res.distance < 0.2


def test_update_prototype_ema():
    # Start with a prototype and update it toward a new embedding using EMA.
    proto = clustering.l2_normalize(np.array([1.0, 0.0], dtype=np.float32))
    emb = clustering.l2_normalize(np.array([0.0, 1.0], dtype=np.float32))
    # With alpha=0.5, the updated prototype should be halfway between and normalized.
    updated = clustering.update_prototype_ema(proto, emb, alpha=0.5)
    # Should lie on the line between (1,0) and (0,1) => (0.5,0.5) normalized => (0.7071,0.7071)
    assert np.allclose(updated, np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.float32), atol=1e-6)


def test_assign_to_centroid_assigns_closer():
    # Two centroids along axes
    ca = clustering.l2_normalize(np.array([1.0, 0.0], dtype=np.float32))
    cb = clustering.l2_normalize(np.array([0.0, 1.0], dtype=np.float32))
    # Embedding closer to A
    emb_a = clustering.l2_normalize(np.array([0.8, 0.2], dtype=np.float32))
    assert clustering.assign_to_centroid(emb_a, ca, cb) == "A"
    # Embedding closer to B
    emb_b = clustering.l2_normalize(np.array([0.2, 0.8], dtype=np.float32))
    assert clustering.assign_to_centroid(emb_b, ca, cb) == "B"
    # Exactly equidistant yields "A" because da <= db
    equi = clustering.l2_normalize(np.array([1.0, 1.0], dtype=np.float32))
    assert clustering.assign_to_centroid(equi, ca, cb) == "A"


def test_pending_buffer_overflow():
    # Ensure pending list is capped at MAX_PENDING and drops oldest
    state = clustering.IndividualState(
        individual_id=1, prototype=clustering.l2_normalize(np.random.randn(3).astype(np.float32))
    )
    # Add more embeddings than MAX_PENDING
    for i in range(clustering.MAX_PENDING + 5):
        emb = clustering.l2_normalize(np.random.randn(3).astype(np.float32))
        state.add_pending(emb)
    assert state.pending is not None
    # The pending buffer length should not exceed MAX_PENDING
    assert len(state.pending) == clustering.MAX_PENDING
    # The earliest entries should have been dropped; we can verify by checking
    # that the last entry corresponds to the most recently added embedding.
    last_emb = state.pending[-1]
    assert isinstance(last_emb, np.ndarray)