
import os
import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import MagicMock, patch

# Define paths to test images relative to this test file
TEST_DATA_DIR = Path(__file__).parent / "integration" / "test_data" / "sahi"
TP_IMAGE = TEST_DATA_DIR / "tp_obs_53.jpg"
FP_IMAGE = TEST_DATA_DIR / "fp_obs_184.jpg"

@pytest.fixture
def mock_sahi_infra():
    """Mock SAHI modules to prevent ImportErrors and avoid real inference."""
    predict_mock = MagicMock()
    
    # Create the result structure
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
    
    predict_mock.return_value = mock_result
    
    sahi_module = MagicMock()
    sahi_predict_module = MagicMock()
    sahi_predict_module.get_sliced_prediction = predict_mock
    
    # Setup mocks for export/AutoDetectionModel
    sahi_auto_model_module = MagicMock()
    sahi_auto_model_module.AutoDetectionModel = MagicMock()
    
    with patch.dict("sys.modules", {
        "sahi": sahi_module,
        "sahi.predict": sahi_predict_module,
        "sahi.auto_model": sahi_auto_model_module,
    }):
        yield predict_mock

def load_image(path: Path) -> np.ndarray:
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        pytest.fail(f"Failed to load image: {path}")
    return img

def test_sahi_detection_tp_obs_53(mock_sahi_infra):
    """Test SAHI pipeline with True Positive image (Obs 53)."""
    frame = load_image(TP_IMAGE)
    mock_get_sliced = mock_sahi_infra
    
    with patch("hbmon.annotation_detector.AnnotationDetector._load_yolo"), \
         patch("hbmon.annotation_detector.AnnotationDetector._load_sam"), \
         patch("hbmon.annotation_detector.AnnotationDetector._load_sahi_model", return_value=MagicMock()):  # Must not return None
        
        # We explicitly enable SAHI via constructor args and ensure env vars are set
        # But patching env vars is good practice too
        with patch.dict(os.environ, {
            "HBMON_SAHI_SLICE_HEIGHT": "640",
            "HBMON_SAHI_SLICE_WIDTH": "640",
            "HBMON_SAHI_OVERLAP_RATIO": "0.2"
        }):
            from hbmon.annotation_detector import AnnotationDetector
            
            # Note: We must pass use_sahi=True to override any default behavior
            # Also use_sam=False to skip SAM logic for this specific test
            detector = AnnotationDetector(use_sahi=True, use_sam=False)
            
            # Setup YOLO check names (used in SAHI result filtering)
            mock_yolo = MagicMock()
            mock_yolo.names = {0: "bird"}
            detector._yolo = mock_yolo
            
            # Run detection
            boxes = detector.detect_frame(frame)
            
            # Verify SAHI was called
            mock_get_sliced.assert_called_once()
            
            # Verify arguments
            call_args = mock_get_sliced.call_args
            assert call_args is not None
            
            # Arg 0 is frame, but converted to RGB in _detect_with_sahi.
            # We can verify it's a numpy array of same shape.
            passed_img = call_args[0][0]
            assert passed_img.shape == frame.shape
            
            # Verify config
            kwargs = call_args[1]
            assert kwargs["slice_height"] == 640
            assert kwargs["slice_width"] == 640
            assert kwargs["overlap_height_ratio"] == 0.2
            
            # Verify boxes parsed
            assert len(boxes) == 1
            assert boxes[0].source == "sahi-auto"

def test_sahi_detection_fp_obs_184(mock_sahi_infra):
    """Test SAHI pipeline with False Positive image (Obs 184)."""
    frame = load_image(FP_IMAGE)
    mock_get_sliced = mock_sahi_infra
    
    with patch("hbmon.annotation_detector.AnnotationDetector._load_yolo"), \
         patch("hbmon.annotation_detector.AnnotationDetector._load_sam"), \
         patch("hbmon.annotation_detector.AnnotationDetector._load_sahi_model", return_value=MagicMock()):
        
        from hbmon.annotation_detector import AnnotationDetector
        detector = AnnotationDetector(use_sahi=True, use_sam=False)
        
        mock_yolo = MagicMock()
        mock_yolo.names = {0: "bird"}
        detector._yolo = mock_yolo
        
        boxes = detector.detect_frame(frame)
        
        mock_get_sliced.assert_called_once()
        assert len(boxes) == 1
        assert boxes[0].source == "sahi-auto"
