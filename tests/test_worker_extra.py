import numpy as np
from unittest.mock import MagicMock
from hbmon.worker import (
    Det,
    _sanitize_bg_params,
    _compute_motion_mask,
    _motion_overlap_stats,
    _pick_best_bird_det,
    _draw_bbox,
    _draw_text_lines,
    _build_observation_media_paths,
    _build_candidate_media_paths,
    _bbox_area_ratio,
    _downscale_shape,
)

def test_sanitize_bg_params():
    # Test valid
    assert _sanitize_bg_params(enabled=True, threshold=30, blur=5, min_overlap=0.15) == (True, 30, 5, 0.15)
    # Test clamping
    assert _sanitize_bg_params(enabled=True, threshold=300, blur=4, min_overlap=-0.1) == (True, 255, 5, 0.0)
    assert _sanitize_bg_params(enabled=True, threshold=-5, blur=1, min_overlap=1.5) == (True, 0, 1, 1.0)

def test_compute_motion_mask():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bg = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add a white box in frame
    frame[40:60, 40:60] = 255
    
    mask = _compute_motion_mask(frame, bg, threshold=50, blur_size=3)
    assert mask.shape == (100, 100)
    assert np.count_nonzero(mask) > 0
    # Center should be non-zero
    assert mask[50, 50] == 255

def test_motion_overlap_stats():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255 # 20x20 = 400 pixels
    det = Det(x1=45, y1=45, x2=55, y2=55, conf=0.9) # 10x10 inside mask
    
    stats = _motion_overlap_stats(det, mask)
    assert stats["bbox_motion_pixels"] == 100
    assert stats["bbox_total_pixels"] == 100
    assert stats["bbox_overlap_ratio"] == 1.0
    
    # Det completely outside
    det_out = Det(x1=0, y1=0, x2=10, y2=10, conf=0.8)
    stats_out = _motion_overlap_stats(det_out, mask)
    assert stats_out["bbox_motion_pixels"] == 0
    assert stats_out["bbox_overlap_ratio"] == 0.0

def test_pick_best_bird_det():
    class MockBox:
        def __init__(self, cls, conf, xyxy):
            self.cls = MagicMock()
            self.cls.item.return_value = cls
            self.conf = MagicMock()
            self.conf.item.return_value = conf
            self.xyxy = [MagicMock()]
            self.xyxy[0].detach().cpu().numpy.return_value = np.array(xyxy)

    class MockResult:
        def __init__(self, boxes):
            self.boxes = boxes

    # 14 is bird
    boxes = [
        MockBox(14, 0.9, [10, 10, 30, 30]), # Area 400
        MockBox(14, 0.8, [10, 10, 40, 40]), # Area 900 -> Should win
        MockBox(0, 0.95, [0, 0, 50, 50]),   # Not a bird
    ]
    results = [MockResult(boxes)]
    
    best = _pick_best_bird_det(results, min_box_area=100, bird_class_id=14)
    assert best is not None
    assert best.area == 900
    assert best.conf == 0.8

def test_pick_best_bird_det_no_results():
    assert _pick_best_bird_det([], 100, 14) is None
    assert _pick_best_bird_det([MagicMock(boxes=None)], 100, 14) is None

def test_draw_bbox():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    det = Det(x1=10, y1=10, x2=50, y2=50, conf=0.9)
    annotated = _draw_bbox(frame, det)
    assert annotated.shape == (100, 100, 3)
    assert not np.array_equal(annotated, frame)

def test_draw_text_lines():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    lines = ["Line 1", "Line 2"]
    annotated = _draw_text_lines(frame, lines)
    assert annotated.shape == (100, 100, 3)
    assert not np.array_equal(annotated, frame)

def test_build_paths():
    obs = _build_observation_media_paths("20230101")
    assert "snapshots/20230101/" in obs.snapshot_rel
    assert obs.observation_uuid is not None
    
    cand = _build_candidate_media_paths("20230101")
    assert "snapshots/candidates/20230101/" in cand.snapshot_rel

def test_bbox_area_ratio():
    det = Det(x1=10, y1=10, x2=20, y2=20, conf=0.9) # Area 100
    assert _bbox_area_ratio(det, (100, 100)) == 0.01
    assert _bbox_area_ratio(det, (0, 0)) == 0.0

def test_downscale_shape():
    assert _downscale_shape(1000, 2000, 1000) == (500, 1000)
    assert _downscale_shape(500, 500, 1000) is None
    assert _downscale_shape(500, 500, 0) is None
