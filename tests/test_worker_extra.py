import numpy as np
import pytest
from hbmon.worker import (
    Det,
    _motion_overlap_stats,
    _detection_overlaps_motion,
    _pick_best_bird_det,
    _collect_bird_detections,
    _roi_motion_stats,
    _downscale_shape
)

def test_motion_overlap_stats_clamping():
    # Det outside image: clamps to [h-1:h, w-1:w] which is 1x1 area
    mask = np.zeros((100, 100))
    det = Det(110, 110, 120, 120, 0.5)
    stats = _motion_overlap_stats(det, mask)
    assert stats["bbox_total_pixels"] == 1
    assert stats["bbox_overlap_ratio"] == 0.0

def test_detection_overlaps_motion_clamped():
    mask = np.zeros((10, 10))
    det = Det(11, 11, 12, 12, 0.5)
    # Clamps to last pixel, which is empty. 0.0 < 0.15
    assert _detection_overlaps_motion(det, mask) is False

def test_roi_motion_stats():
    mask = np.zeros((10, 10))
    mask[0, 0] = 255
    stats = _roi_motion_stats(mask)
    assert stats["roi_motion_pixels"] == 1
    assert stats["roi_total_pixels"] == 100
    assert stats["roi_motion_fraction"] == 0.01

def test_downscale_shape():
    assert _downscale_shape(100, 200, 50) == (25, 50)
    assert _downscale_shape(100, 200, 300) is None
    assert _downscale_shape(100, 200, 0) is None

class Detach:
    def __init__(self, val): self.val = val
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return np.array(self.val)

class FakeBox:
    def __init__(self, d, cls_id):
        self.cls = np.array([cls_id])
        self.conf = np.array([d.conf])
        self.xyxy = [Detach([d.x1, d.y1, d.x2, d.y2])]

class FakeBoxes:
    def __init__(self, boxes):
        self.boxes = boxes
    def __iter__(self):
        return iter(self.boxes)
    def __len__(self):
        return len(self.boxes)

class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes

def test_pick_best_bird_det_motion_filter():
    # Mock ultralytics results
    box1 = Det(10, 10, 20, 20, 0.9) # Bird 1
    box2 = Det(30, 30, 60, 60, 0.8) # Bird 2 (larger)
    
    b1 = FakeBox(box1, 14)
    b2 = FakeBox(box2, 14)
    
    results = [FakeResult(FakeBoxes([b1, b2]))]
    
    # No motion mask: picks larger area (b2)
    best = _pick_best_bird_det(results, 10, 14)
    assert best is not None
    assert best.x1 == 30
    
    # With motion mask: mask only has motion at b1
    mask = np.zeros((100, 100))
    mask[10:20, 10:20] = 255
    best = _pick_best_bird_det(results, 10, 14, motion_mask=mask)
    assert best is not None
    assert best.x1 == 10

def test_collect_bird_detections_mismatch_cls():
    results = [FakeResult(FakeBoxes([FakeBox(Det(0,0,10,10,0.9), 15)]))] # 15 != 14
    dets = _collect_bird_detections(results, 5, 14)
    assert len(dets) == 0
