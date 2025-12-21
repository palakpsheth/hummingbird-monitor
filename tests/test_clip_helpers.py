"""
Additional unit tests for helper functions in ``hbmon.clip_model``.

These tests focus on the lightweight utilities that do not require any
heavy machine‑learning dependencies.  Functions like ``_softmax`` and
``crop_bgr`` can be exercised with simple numpy arrays.  The PIL
conversion in ``_ensure_rgb_pil`` is only tested when Pillow is
installed; otherwise the test is skipped.
"""

import numpy as np
import pytest

import hbmon.clip_model as clip


def test_softmax_properties():
    # Simple array: max should yield highest prob and sum should be 1.0
    x = np.array([2.0, 1.0, 0.1], dtype=np.float64)
    probs = clip._softmax(x)
    # Probabilities sum to 1
    assert np.allclose(probs.sum(), 1.0, atol=1e-12)
    # Index of max input corresponds to max probability
    assert int(np.argmax(probs)) == 0
    # Monotonic ordering: larger input gives larger probability
    # Because softmax is strictly increasing on individual elements
    assert probs[0] > probs[1] > probs[2]

    # Test with negative and large values to ensure numerical stability
    y = np.array([-10.0, 0.0, 10.0], dtype=np.float64)
    p2 = clip._softmax(y)
    # The largest value dominates
    assert p2[2] > 0.999 and p2[0] < 1e-4


def test_crop_bgr_clamps_bounds():
    # Create a 4x4 image with a simple gradient and 3 channels
    img = np.arange(4 * 4 * 3, dtype=np.uint8).reshape((4, 4, 3))
    # Bounding box partially outside image should be clamped to valid region
    # Negative values and values beyond width/height are clamped
    cropped = clip.crop_bgr(img, (-5, -5, 5, 5))
    # The resulting crop should span the entire original image
    assert cropped.shape == img.shape
    # BBox with x2 <= x1 should return original image
    cropped2 = clip.crop_bgr(img, (3, 3, 2, 2))
    assert np.array_equal(cropped2, img)


def test_ensure_rgb_pil_conversion(monkeypatch):
    # If Pillow is unavailable, skip this test
    if clip.Image is None:
        pytest.skip("Pillow not installed; skipping _ensure_rgb_pil test")
    # Construct a small BGR image and verify conversion to PIL
    # The BGR pixel [B,G,R] should become [R,G,B] in the RGB image
    bgr = np.zeros((1, 1, 3), dtype=np.uint8)
    bgr[0, 0] = [1, 2, 3]  # B=1, G=2, R=3
    pil_img = clip._ensure_rgb_pil(bgr)
    assert pil_img.mode == "RGB"
    # The pixel value in PIL should be (R,G,B)=(3,2,1)
    assert pil_img.getpixel((0, 0)) == (3, 2, 1)