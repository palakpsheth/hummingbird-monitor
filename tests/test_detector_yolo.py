# tests/test_detector_yolo.py
from __future__ import annotations

import numpy as np


def test_detector_filters_by_confidence(monkeypatch):
    """
    Spec: detector returns structured detections and filters low-confidence boxes.
    """
    # Example expected API:
    #   from hbmon.detect.yolo import YoloDetector
    #   det = YoloDetector(model=...)
    #   det.detect(frame) -> list[Detection]
    from hbmon.detect.yolo import YoloDetector  # type: ignore

    class DummyModel:
        def __call__(self, frame):
            # Return a pretend result format your wrapper knows how to parse.
            return [
                {"xyxy": [10, 10, 50, 50], "conf": 0.9, "cls": "bird"},
                {"xyxy": [0, 0, 5, 5], "conf": 0.2, "cls": "bird"},
            ]

    det = YoloDetector(model=DummyModel(), conf_threshold=0.5)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = det.detect(frame)

    assert len(out) == 1
    assert out[0].label in {"bird", "hummingbird"}
    assert out[0].conf >= 0.5
    x1, y1, x2, y2 = out[0].bbox_xyxy
    assert x2 > x1 and y2 > y1


def test_detector_empty(monkeypatch):
    from hbmon.detect.yolo import YoloDetector  # type: ignore

    class DummyModel:
        def __call__(self, frame):
            return []

    det = YoloDetector(model=DummyModel(), conf_threshold=0.5)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    assert det.detect(frame) == []
