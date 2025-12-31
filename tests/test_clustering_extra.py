import numpy as np
import pytest
from hbmon.clustering import update_prototype_ema, cosine_distance, l2_normalize

def test_l2_normalize_zero():
    v = np.zeros(128)
    normed = l2_normalize(v)
    assert np.all(normed == 0)

def test_update_prototype_ema_none():
    # Test with None prototype
    new_vec = np.random.rand(128)
    res = update_prototype_ema(None, new_vec, alpha=0.1)
    # Should return normalized new_vec
    assert np.allclose(np.linalg.norm(res), 1.0)
    assert np.allclose(res, l2_normalize(new_vec))

def test_cosine_distance_zero():
    v1 = np.zeros(128)
    v2 = np.random.rand(128)
    # Distance to zero vector should be 1.0 (no similarity)
    assert cosine_distance(v1, v2) == 1.0
    assert cosine_distance(v2, v1) == 1.0

def test_update_prototype_ema_shapes():
    p = np.random.rand(128)
    v = np.random.rand(64) # Mismatched
    with pytest.raises(ValueError):
        update_prototype_ema(p, v)
