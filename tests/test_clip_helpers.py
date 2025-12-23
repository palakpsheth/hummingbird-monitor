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


def test_get_env_returns_value(monkeypatch):
    """Test _get_env returns environment variable value when set."""
    monkeypatch.setenv("TEST_CLIP_VAR", "test_value")
    result = clip._get_env("TEST_CLIP_VAR", "default")
    assert result == "test_value"


def test_get_env_returns_default_when_missing(monkeypatch):
    """Test _get_env returns default when variable is missing."""
    monkeypatch.delenv("MISSING_VAR", raising=False)
    result = clip._get_env("MISSING_VAR", "default_value")
    assert result == "default_value"


def test_get_env_returns_default_when_empty(monkeypatch):
    """Test _get_env returns default when variable is empty string."""
    monkeypatch.setenv("EMPTY_VAR", "")
    result = clip._get_env("EMPTY_VAR", "default")
    assert result == "default"


def test_get_env_returns_default_when_whitespace(monkeypatch):
    """Test _get_env returns default when variable is whitespace only."""
    monkeypatch.setenv("WHITESPACE_VAR", "   ")
    result = clip._get_env("WHITESPACE_VAR", "default")
    assert result == "default"


def test_softmax_numerical_stability_large_values():
    """Test softmax handles very large values without overflow."""
    x = np.array([1000.0, 1001.0, 1002.0], dtype=np.float64)
    probs = clip._softmax(x)
    # Should not produce NaN or Inf
    assert not np.any(np.isnan(probs))
    assert not np.any(np.isinf(probs))
    assert np.isclose(probs.sum(), 1.0)


def test_softmax_with_negative_values():
    """Test softmax works with all negative values."""
    x = np.array([-5.0, -2.0, -1.0], dtype=np.float64)
    probs = clip._softmax(x)
    assert np.isclose(probs.sum(), 1.0)
    # Less negative -> higher probability
    assert probs[2] > probs[1] > probs[0]


def test_crop_bgr_returns_original_when_bbox_none():
    """Test crop_bgr returns original when bbox is None."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    result = clip.crop_bgr(img, None)
    assert np.array_equal(result, img)


def test_crop_bgr_valid_crop():
    """Test crop_bgr with valid bounding box."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[25:75, 50:150, 0] = 255  # Mark region blue
    result = clip.crop_bgr(img, (50, 25, 150, 75))
    assert result.shape == (50, 100, 3)
    assert result[0, 0, 0] == 255  # Should contain the blue region


def test_ensure_rgb_pil_raises_on_none():
    """Test _ensure_rgb_pil raises ValueError on None input."""
    with pytest.raises(ValueError, match="None"):
        clip._ensure_rgb_pil(None)


def test_ensure_rgb_pil_raises_on_wrong_shape():
    """Test _ensure_rgb_pil raises ValueError on wrong image shape."""
    # Grayscale image (2D array)
    gray = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="HxWx3"):
        clip._ensure_rgb_pil(gray)


def test_ensure_rgb_pil_raises_on_wrong_channels():
    """Test _ensure_rgb_pil raises ValueError on wrong channel count."""
    # RGBA image (4 channels)
    rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="HxWx3"):
        clip._ensure_rgb_pil(rgba)


def test_ensure_rgb_pil_handles_float_image():
    """Test _ensure_rgb_pil handles float images by clipping values."""
    if clip.Image is None:
        pytest.skip("Pillow not installed")
    float_img = np.zeros((10, 10, 3), dtype=np.float32)
    float_img[:, :, 1] = 128.5
    pil_img = clip._ensure_rgb_pil(float_img)
    assert pil_img.mode == "RGB"


def test_species_prediction_frozen():
    """Test SpeciesPrediction dataclass is frozen (immutable)."""
    pred = clip.SpeciesPrediction(
        label="Test",
        probability=0.5,
        probs={"Test": 0.5},
        logits={"Test": 1.0},
    )
    # Attempting to modify should raise
    with pytest.raises(Exception):
        pred.label = "Changed"


def test_default_species_list():
    """Test DEFAULT_SPECIES is defined and contains expected entries."""
    assert hasattr(clip, "DEFAULT_SPECIES")
    assert isinstance(clip.DEFAULT_SPECIES, list)
    assert "Anna's Hummingbird" in clip.DEFAULT_SPECIES
    assert "Rufous Hummingbird" in clip.DEFAULT_SPECIES
    assert "Hummingbird (unknown species)" in clip.DEFAULT_SPECIES


def test_default_prompts_match_species():
    """Test DEFAULT_PROMPTS has entries for all species in DEFAULT_SPECIES."""
    for species in clip.DEFAULT_SPECIES:
        assert species in clip.DEFAULT_PROMPTS
        assert isinstance(clip.DEFAULT_PROMPTS[species], list)
        assert len(clip.DEFAULT_PROMPTS[species]) > 0