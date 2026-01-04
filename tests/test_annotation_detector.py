
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Mock sahi modules before import
@pytest.fixture(autouse=True)
def mock_sahi_modules():
    with patch.dict("sys.modules", {
        "sahi": MagicMock(),
        "sahi.predict": MagicMock(),
        "sahi.utils": MagicMock(),
        "sahi.utils.yolov8": MagicMock(),
    }):
        yield

def test_sahi_integration_logic():
    """Test SAHI usage in detect_frame."""
    
    # Mock return values
    mock_prediction = MagicMock()
    mock_prediction.bbox.minx = 100
    mock_prediction.bbox.miny = 100
    mock_prediction.bbox.maxx = 200
    mock_prediction.bbox.maxy = 200
    mock_prediction.score.value = 0.95
    mock_prediction.category.id = 0
    mock_prediction.category.name = "bird"
    
    mock_result = MagicMock()
    mock_result.object_prediction_list = [mock_prediction]
    
    with patch("hbmon.annotation_detector.AnnotationDetector._load_yolo"), \
         patch("hbmon.annotation_detector.AnnotationDetector._load_sam"), \
         patch("sahi.predict.get_sliced_prediction", return_value=mock_result) as mock_get_sliced:
        
        # Configure env
        with patch.dict(os.environ, {
            "HBMON_ANNOTATION_USE_SAHI": "1",
            "HBMON_ANNOTATION_YOLO_MODEL": "yolo_mock.pt",
            "HBMON_ANNOTATION_USE_SAM": "0", # Disable SAM download
        }):
            from hbmon.annotation_detector import AnnotationDetector
            detector = AnnotationDetector()
            
            # Mock internal YOLO for name resolution
            mock_desc = MagicMock()
            mock_desc.names = {0: "bird"}
            detector._yolo = mock_desc
            
            # Initialize
            detector.initialize()
            
            # Test frame
            frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
            
            # Run detection
            boxes = detector.detect_frame(frame)
            
            # Assertions
            assert len(boxes) == 1
            assert boxes[0].source == "sahi-auto"
            # 100-200 in 1000x1000 => 0.1-0.2
            # Center: 0.15, Width: 0.1
            assert abs(boxes[0].x - 0.15) < 1e-5
            assert abs(boxes[0].y - 0.15) < 1e-5
            
            # Verify sliced prediction called
            mock_get_sliced.assert_called_once()
            
def test_sahi_disabled():
    """Test SAHI disabled falls back to standard."""
    with patch("hbmon.annotation_detector.AnnotationDetector._detect_standard_yolo") as mock_standard:
        mock_standard.return_value = []
        
        # Patch the global constant since it's read at import time
        with patch("hbmon.annotation_detector.ANNOTATION_USE_SAHI", False):
            from hbmon.annotation_detector import AnnotationDetector
            # Also verify explicit init arg overrides
            detector = AnnotationDetector(use_sahi=False)
            detector.initialize() # mocks load yolo internally
            
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            detector.detect_frame(frame)
            
            mock_standard.assert_called_once()
