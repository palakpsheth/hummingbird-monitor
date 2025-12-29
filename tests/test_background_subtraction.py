"""
Tests for background subtraction functions in hbmon.worker.

These tests verify the motion detection and detection filtering logic
used for background subtraction.
"""

import numpy as np
import pytest

# Import the worker module to access the background subtraction functions.
# Note: These functions require OpenCV, which may not be available in all test environments.
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# Skip all tests in this module if OpenCV is not available
pytestmark = pytest.mark.skipif(not _CV2_AVAILABLE, reason="OpenCV not available")


@pytest.fixture
def sample_background():
    """Create a simple solid gray background image."""
    # 100x100 gray image (BGR format)
    return np.full((100, 100, 3), 128, dtype=np.uint8)


@pytest.fixture
def sample_frame_with_motion(sample_background):
    """Create a frame with a white square (simulating motion) on the background."""
    frame = sample_background.copy()
    # Add a white 20x20 square at position (40, 40) to simulate motion
    frame[40:60, 40:60] = 255
    return frame


@pytest.fixture
def sample_frame_no_motion(sample_background):
    """Create a frame identical to the background (no motion)."""
    return sample_background.copy()


class TestComputeMotionMask:
    """Tests for the _compute_motion_mask function."""

    def test_identical_frames_produce_no_motion(self, sample_background):
        """When frame matches background exactly, mask should be all zeros."""
        from hbmon.worker import _compute_motion_mask

        mask = _compute_motion_mask(sample_background, sample_background.copy())
        
        # Should have no motion detected
        assert mask.shape == sample_background.shape[:2]
        assert np.count_nonzero(mask) == 0

    def test_different_frames_produce_motion(self, sample_background, sample_frame_with_motion):
        """When frame differs from background, mask should show motion areas."""
        from hbmon.worker import _compute_motion_mask

        mask = _compute_motion_mask(sample_frame_with_motion, sample_background)
        
        # Should detect motion where the white square is
        assert mask.shape == sample_background.shape[:2]
        assert np.count_nonzero(mask) > 0

    def test_resizes_background_to_match_frame(self, sample_background):
        """Background is resized if dimensions don't match the frame."""
        from hbmon.worker import _compute_motion_mask

        # Create a larger frame
        larger_frame = np.full((200, 200, 3), 128, dtype=np.uint8)
        
        # Should not raise, and should return mask matching frame size
        mask = _compute_motion_mask(larger_frame, sample_background)
        assert mask.shape == (200, 200)

    def test_threshold_affects_sensitivity(self, sample_background):
        """Higher threshold requires more significant pixel difference."""
        from hbmon.worker import _compute_motion_mask

        # Frame with slight change (pixel value 140 vs 128 = 12 difference)
        frame_slight_change = sample_background.copy()
        frame_slight_change[40:60, 40:60] = 140

        # Low threshold should detect the change
        mask_low = _compute_motion_mask(frame_slight_change, sample_background, threshold=5)
        
        # High threshold should not detect the change
        mask_high = _compute_motion_mask(frame_slight_change, sample_background, threshold=50)
        
        assert np.count_nonzero(mask_low) > np.count_nonzero(mask_high)

    def test_even_blur_size_is_normalized_to_odd(self, sample_background, sample_frame_with_motion):
        """Even blur_size values should be normalized to odd (no error)."""
        from hbmon.worker import _compute_motion_mask

        # Should not raise error with even blur size
        mask = _compute_motion_mask(sample_frame_with_motion, sample_background, blur_size=4)
        assert mask.shape == sample_background.shape[:2]

    def test_zero_blur_size_skips_blur(self, sample_background, sample_frame_with_motion):
        """blur_size=0 should skip Gaussian blur."""
        from hbmon.worker import _compute_motion_mask

        # Should not raise error
        mask = _compute_motion_mask(sample_frame_with_motion, sample_background, blur_size=0)
        assert mask.shape == sample_background.shape[:2]


class TestDetectionOverlapsMotion:
    """Tests for the _detection_overlaps_motion function."""

    def test_detection_in_motion_area_returns_true(self):
        """Detection overlapping motion area should return True."""
        from hbmon.worker import _detection_overlaps_motion, Det

        # Create a mask with motion in center
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255  # Motion in center

        # Detection box covering the motion area
        det = Det(x1=35, y1=35, x2=65, y2=65, conf=0.9)
        
        assert _detection_overlaps_motion(det, mask, min_overlap_ratio=0.15) is True

    def test_detection_outside_motion_area_returns_false(self):
        """Detection outside motion area should return False."""
        from hbmon.worker import _detection_overlaps_motion, Det

        # Create a mask with motion in center
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255  # Motion in center

        # Detection box in corner, no overlap with motion
        det = Det(x1=0, y1=0, x2=20, y2=20, conf=0.9)
        
        assert _detection_overlaps_motion(det, mask, min_overlap_ratio=0.15) is False

    def test_partial_overlap_respects_threshold(self):
        """Partial overlap should be compared against min_overlap_ratio."""
        from hbmon.worker import _detection_overlaps_motion, Det

        # Create a mask with motion in small area
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[45:55, 45:55] = 255  # Small 10x10 motion area

        # Detection box much larger than motion area (50x50)
        det = Det(x1=25, y1=25, x2=75, y2=75, conf=0.9)
        
        # Motion covers 100 pixels, detection covers 2500 pixels = 4% overlap
        # Should return False with 15% threshold
        assert _detection_overlaps_motion(det, mask, min_overlap_ratio=0.15) is False
        
        # Should return True with 1% threshold
        assert _detection_overlaps_motion(det, mask, min_overlap_ratio=0.01) is True

    def test_detection_at_image_boundary_is_clamped(self):
        """Detection coordinates outside image bounds should be clamped."""
        from hbmon.worker import _detection_overlaps_motion, Det

        mask = np.ones((100, 100), dtype=np.uint8) * 255  # All motion

        # Detection extends beyond image boundaries
        det = Det(x1=-10, y1=-10, x2=110, y2=110, conf=0.9)
        
        # Should not raise error and should return True (motion everywhere)
        assert _detection_overlaps_motion(det, mask, min_overlap_ratio=0.15) is True

    def test_zero_area_detection_is_clamped_to_minimum(self):
        """Edge case: zero-area detection should be clamped to at least 1x1 pixel."""
        from hbmon.worker import _detection_overlaps_motion, Det

        # Create a mask with no motion
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        # Detection with x2 == x1 and y2 == y1 (zero area)
        det = Det(x1=50, y1=50, x2=50, y2=50, conf=0.9)
        
        # The function clamps to ensure at least 1x1 pixel area.
        # With no motion in the mask, the overlap is 0%, so it should return False.
        result = _detection_overlaps_motion(det, mask, min_overlap_ratio=0.15)
        assert result is False
        
        # With motion at the detection point, it should return True
        mask[50, 50] = 255
        result = _detection_overlaps_motion(det, mask, min_overlap_ratio=0.15)
        assert result is True


class TestMotionStats:
    """Tests for motion overlap/stat helpers."""

    def test_motion_overlap_stats(self):
        """Overlap stats should count motion pixels inside bbox."""
        from hbmon.worker import _motion_overlap_stats, Det

        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:10, 5:10] = 255
        det = Det(x1=0, y1=0, x2=10, y2=10, conf=0.5)

        stats = _motion_overlap_stats(det, mask)
        assert stats["bbox_motion_pixels"] == 25
        assert stats["bbox_total_pixels"] == 100
        assert stats["bbox_overlap_ratio"] == 0.25

    def test_roi_motion_stats(self):
        """ROI motion stats should compute fraction of active pixels."""
        from hbmon.worker import _roi_motion_stats

        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:2, 0:5] = 255  # 10 pixels
        stats = _roi_motion_stats(mask)

        assert stats["roi_motion_pixels"] == 10
        assert stats["roi_total_pixels"] == 100
        assert stats["roi_motion_fraction"] == 0.1


class TestLoadBackgroundImage:
    """Tests for the _load_background_image function."""

    def test_returns_none_when_no_background_configured(self, monkeypatch, tmp_path):
        """Should return None when background image doesn't exist."""
        import hbmon.config as config
        from hbmon.worker import _load_background_image

        monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
        config.ensure_dirs()
        
        result = _load_background_image()
        assert result is None

    def test_loads_existing_background_image(self, monkeypatch, tmp_path):
        """Should load and return background image when it exists."""
        import hbmon.config as config
        from hbmon.worker import _load_background_image

        monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
        config.ensure_dirs()
        
        # Create a test background image
        bg_path = config.background_image_path()
        test_img = np.full((50, 50, 3), 100, dtype=np.uint8)
        cv2.imwrite(str(bg_path), test_img)
        
        result = _load_background_image()
        
        assert result is not None
        assert result.shape == (50, 50, 3)

    def test_returns_none_for_invalid_image(self, monkeypatch, tmp_path):
        """Should return None if image file is corrupted/invalid."""
        import hbmon.config as config
        from hbmon.worker import _load_background_image

        monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
        config.ensure_dirs()
        
        # Create an invalid image file
        bg_path = config.background_image_path()
        bg_path.write_text("not an image")
        
        result = _load_background_image()
        assert result is None
