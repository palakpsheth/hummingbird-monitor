# tests/test_reid.py
from __future__ import annotations

import numpy as np


def test_reid_assigns_same_individual_when_close():
    """
    Spec: embedding-based re-ID assigns to existing individual if cosine distance is below threshold.
    """
    from hbmon.reid.assign import ReIdentifier  # type: ignore

    # Two embeddings very close
    e1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e2 = np.array([0.99, 0.01, 0.0], dtype=np.float32)

    reid = ReIdentifier(distance_threshold=0.1)

    # first time: creates individual 1
    ind1 = reid.assign(embedding=e1)
    # second time: should match
    ind2 = reid.assign(embedding=e2)

    assert ind1 == ind2


def test_reid_creates_new_individual_when_far():
    from hbmon.reid.assign import ReIdentifier  # type: ignore

    e1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    reid = ReIdentifier(distance_threshold=0.1)

    ind1 = reid.assign(embedding=e1)
    ind2 = reid.assign(embedding=e2)

    assert ind1 != ind2
