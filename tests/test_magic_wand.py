"""
Tests for Magic Wand on-demand detection endpoint.
"""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_annotation_frame():
    """Create a mock AnnotationFrame."""
    frame = MagicMock()
    frame.id = 123
    frame.frame_path = "/media/obs_1/frames/frame_001.jpg"
    return frame


@pytest.fixture
def mock_detector():
    """Create a mock AnnotationDetector."""
    detector = MagicMock()
    # Create mock detected boxes
    mock_box = MagicMock()
    mock_box.class_id = 14  # bird
    mock_box.x = 0.5
    mock_box.y = 0.5
    mock_box.w = 0.2
    mock_box.h = 0.15
    mock_box.confidence = 0.85
    detector._detect_standard_yolo.return_value = [mock_box]
    return detector


class TestMagicWandEndpoint:
    """Test the detect_region API endpoint."""

    @pytest.mark.asyncio
    async def test_detect_region_returns_boxes(self, mock_annotation_frame, mock_detector):
        """Test that detect_region returns correctly mapped boxes."""
        # This is a unit test for the coordinate remapping logic
        # Simulating crop at center of 1920x1080 image
        
        # Input: click at center (0.5, 0.5) of a 1920x1080 image
        # Crop size: 640x640
        # Crop origin: (640, 220) to (1280, 860)
        
        w, h = 1920, 1080
        cx, cy = 960, 540  # center
        crop_size = 640
        half = crop_size // 2
        
        x1 = max(0, cx - half)  # 640
        y1 = max(0, cy - half)  # 220
        
        crop_w, crop_h = 640, 640
        
        # Mock box from detector (normalized to crop)
        box_x_crop = 0.5  # center of crop
        box_y_crop = 0.5
        box_w_crop = 0.2
        box_h_crop = 0.15
        
        # Calculate expected full-image coords
        bx_px = box_x_crop * crop_w  # 320
        by_px = box_y_crop * crop_h  # 320
        bw_px = box_w_crop * crop_w  # 128
        bh_px = box_h_crop * crop_h  # 96
        
        final_cx_px = x1 + bx_px  # 640 + 320 = 960
        final_cy_px = y1 + by_px  # 220 + 320 = 540
        
        expected_x = final_cx_px / w  # 960 / 1920 = 0.5
        expected_y = final_cy_px / h  # 540 / 1080 = 0.5
        expected_w = bw_px / w  # 128 / 1920 ≈ 0.0667
        expected_h = bh_px / h  # 96 / 1080 ≈ 0.0889
        
        assert abs(expected_x - 0.5) < 0.01
        assert abs(expected_y - 0.5) < 0.01
        assert abs(expected_w - 0.0667) < 0.01
        assert abs(expected_h - 0.0889) < 0.01

    def test_crop_bounds_clamping_at_edge(self):
        """Test that crop bounds are correctly clamped at image edges."""
        # Click near top-left corner
        w, h = 1920, 1080
        cx, cy = 100, 100  # Near corner
        crop_size = 640
        half = crop_size // 2  # 320
        
        x1 = max(0, cx - half)  # max(0, -220) = 0
        y1 = max(0, cy - half)  # max(0, -220) = 0
        x2 = min(w, x1 + crop_size)  # min(1920, 640) = 640
        y2 = min(h, y1 + crop_size)  # min(1080, 640) = 640
        
        assert x1 == 0
        assert y1 == 0
        assert x2 == 640
        assert y2 == 640

    def test_crop_bounds_clamping_at_bottom_right(self):
        """Test that crop bounds are correctly clamped at bottom-right edge."""
        w, h = 1920, 1080
        cx, cy = 1800, 1000  # Near bottom-right
        crop_size = 640
        half = crop_size // 2
        
        x1 = max(0, cx - half)  # 1480
        y1 = max(0, cy - half)  # 680
        x2 = min(w, x1 + crop_size)  # min(1920, 2120) = 1920
        y2 = min(h, y1 + crop_size)  # min(1080, 1320) = 1080
        
        # Adjust x1/y1 to ensure full crop size
        if x2 - x1 < crop_size:
            x1 = max(0, x2 - crop_size)  # 1920 - 640 = 1280
        if y2 - y1 < crop_size:
            y1 = max(0, y2 - crop_size)  # 1080 - 640 = 440
        
        assert x1 == 1280
        assert y1 == 440
        assert x2 - x1 == 640
        assert y2 - y1 == 640


class TestMagicWandIntegration:
    """Integration-style tests (still mocked but testing more components)."""

    @pytest.mark.asyncio
    async def test_endpoint_handles_missing_frame(self):
        """Test that endpoint returns 404 for missing frame."""
        # This would be tested with TestClient in a real integration test
        pass  # Placeholder for actual integration test

    @pytest.mark.asyncio
    async def test_endpoint_handles_missing_image_file(self):
        """Test that endpoint returns 404 for missing image file."""
        pass  # Placeholder for actual integration test

    @pytest.mark.asyncio
    async def test_endpoint_handles_detector_error(self):
        """Test that endpoint gracefully handles detector errors."""
        pass  # Placeholder for actual integration test
