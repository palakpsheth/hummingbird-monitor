import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from pathlib import Path
from hbmon.config import Settings, Roi
from hbmon.worker import _prepare_crop_and_clip, Det, _apply_roi

@pytest.mark.anyio
async def test_prepare_crop_and_clip_runs_inference():
    """Test that _prepare_crop_and_clip calls CLIP model when provided."""
    # Setup
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    det = Det(x1=10, y1=10, x2=50, y2=50, conf=0.9)
    clip_mock = MagicMock()
    clip_mock.predict_species_label_prob.return_value = ("Test Bird", 0.95)
    clip_mock.encode_embedding.return_value = np.zeros(512)
    
    # Run
    # mocking _bbox_with_padding (assuming it works, but verifying call flow)
    # mocking _write_jpeg to avoid FS ops
    with patch("hbmon.worker._bbox_with_padding", return_value=(10, 10, 50, 50)), \
         patch("hbmon.worker._write_jpeg") as mock_write:
         
        crop, (label, prob), emb = await _prepare_crop_and_clip(frame, det, 0.0, Path("dump.jpg"), clip_mock)
    
    # Assert
    assert label == "Test Bird"
    assert prob == 0.95
    assert emb is not None
    clip_mock.predict_species_label_prob.assert_called_once()
    mock_write.assert_called_once()

@pytest.mark.anyio
async def test_prepare_crop_and_clip_no_clip():
    """Test graceful fallback when CLIP model is None."""
    # Setup
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    det = Det(x1=10, y1=10, x2=50, y2=50, conf=0.9)
    
    with patch("hbmon.worker._bbox_with_padding", return_value=(10, 10, 50, 50)), \
         patch("hbmon.worker._write_jpeg") as mock_write:
        crop, (label, prob), emb = await _prepare_crop_and_clip(frame, det, 0.0, Path("dump.jpg"), None)

    assert label is None
    assert prob == 0.0
    assert emb is None
    mock_write.assert_called_once()

def test_settings_roi_usage_regression():
    """
    Regression test for TypeError: object of type 'Roi' has no len().
    Verifies that we can check existence of s.roi and use it validation-free.
    """
    s = Settings(roi=Roi(0, 0, 1, 1))
    
    # Verify the bug fix: simple boolean check passes
    assert s.roi
    
    # Verify the legacy bug: len() fails - effectively testing that Roi is indeed the object that caused the issue
    with pytest.raises(TypeError):
        len(s.roi)
        
    # Verify usage in _apply_roi works
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    roi_frame, offset = _apply_roi(frame, s)
    # With full ROI (0..1), it should return roughly the whole frame (clamped)
    # The mocks/logic in _apply_roi will clamp and slice.
    assert roi_frame.shape == (100, 100, 3)
    assert offset == (0, 0)
