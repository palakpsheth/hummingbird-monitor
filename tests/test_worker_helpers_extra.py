"""
Additional tests for worker helper functions.
"""

from __future__ import annotations

from types import SimpleNamespace
import subprocess

import numpy as np
import pytest

from hbmon.config import Roi, Settings
from hbmon.worker import (
    Det,
    _apply_roi,
    _bbox_area_ratio,
    _bbox_with_padding,
    _build_observation_extra_data,
    _build_observation_media_paths,
    _build_candidate_media_paths,
    _build_mask_paths,
    _convert_to_h264,
    _downscale_shape,
    _detection_overlaps_motion,
    _motion_overlap_stats,
    _roi_motion_stats,
    _compute_motion_mask,
    _pick_best_bird_det,
    _collect_bird_detections,
    _select_best_detection,
    _format_bbox_label,
    _ffmpeg_available,
    _load_background_image,
    _write_png,
    _safe_mkdir,
    _save_motion_mask_images,
    _write_jpeg,
    _sanitize_bg_params,
    _draw_bbox,
    _draw_text_lines,
)


def test_build_observation_media_paths_with_fixed_uuid():
    paths = _build_observation_media_paths("20240101", observation_uuid="abc123")
    assert paths.observation_uuid == "abc123"
    assert paths.snapshot_rel.endswith("snapshots/20240101/abc123.jpg")
    assert paths.snapshot_annotated_rel.endswith("snapshots/20240101/abc123_annotated.jpg")
    assert paths.snapshot_clip_rel.endswith("snapshots/20240101/abc123_clip.jpg")
    assert paths.clip_rel.endswith("clips/20240101/abc123.mp4")


def test_build_candidate_media_paths_with_fixed_uuid():
    paths = _build_candidate_media_paths("20240101", candidate_uuid="cand123", mask_ext="png")
    assert paths.candidate_uuid == "cand123"
    assert paths.snapshot_rel.endswith("snapshots/candidates/20240101/cand123.jpg")
    assert paths.snapshot_annotated_rel.endswith("snapshots/candidates/20240101/cand123_ann.jpg")
    assert paths.clip_rel.endswith("clips/candidates/20240101/cand123.mp4")
    assert paths.mask_rel.endswith("masks/candidates/20240101/cand123_mask.png")
    assert paths.mask_overlay_rel.endswith("masks/candidates/20240101/cand123_overlay.png")


def test_build_mask_paths():
    mask_rel, overlay_rel = _build_mask_paths("20240101", "obs123", mask_ext="png")
    assert mask_rel.endswith("masks/observations/20240101/obs123_mask.png")
    assert overlay_rel.endswith("masks/observations/20240101/obs123_overlay.png")


def test_build_observation_extra_data_defaults_review():
    data = _build_observation_extra_data(
        observation_uuid="obs-1",
        sensitivity={"detect_conf": 0.3},
        detection={"bbox_area": 500},
        identification={"species": "Anna's"},
        snapshots={"raw": "snap.jpg"},
        review=None,
    )
    assert data["observation_uuid"] == "obs-1"
    assert data["review"] == {"label": None}


def test_apply_roi_clamps_and_returns_offset():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    settings = Settings()
    settings.roi = Roi(x1=0.1, y1=0.2, x2=0.6, y2=0.8)
    roi_frame, (x_off, y_off) = _apply_roi(frame, settings)
    assert roi_frame.shape[0] == 60
    assert roi_frame.shape[1] == 100
    assert (x_off, y_off) == (20, 20)


def test_downscale_shape_limits():
    assert _downscale_shape(10, 5, 0) is None
    assert _downscale_shape(10, 5, 12) is None
    assert _downscale_shape(10, 5, 4) == (4, 2)


def test_sanitize_bg_params_normalizes_values():
    enabled, threshold, blur, overlap = _sanitize_bg_params(
        enabled=True,
        threshold=300,
        blur=4,
        min_overlap=-0.5,
    )
    assert enabled is True
    assert threshold == 255
    assert blur == 5
    assert overlap == 0.0


def test_detection_overlaps_motion_threshold():
    motion_mask = np.zeros((10, 10), dtype=np.uint8)
    motion_mask[2:6, 2:6] = 255
    det = Det(x1=0, y1=0, x2=5, y2=5, conf=0.9)
    assert _detection_overlaps_motion(det, motion_mask, min_overlap_ratio=0.1)
    assert not _detection_overlaps_motion(det, motion_mask, min_overlap_ratio=0.9)


def test_bbox_helpers_format_and_ratio():
    det = Det(x1=0, y1=0, x2=10, y2=5, conf=0.42)
    assert _format_bbox_label(det) == "0.42 | 50px^2"
    assert _bbox_area_ratio(det, (10, 10)) == 0.5
    assert _bbox_area_ratio(det, (0, 0)) == 0.0


def test_bbox_with_padding_clamps():
    det = Det(x1=5, y1=5, x2=10, y2=10, conf=0.3)
    x1, y1, x2, y2 = _bbox_with_padding(det, (12, 12), pad_frac=0.5)
    assert (x1, y1) == (2, 2)
    assert (x2, y2) == (12, 12)


def test_convert_to_h264_handles_missing_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setattr("hbmon.worker._ffmpeg_available", lambda: False)
    result = _convert_to_h264(tmp_path / "in.mp4", tmp_path / "out.mp4")
    assert result is False


def test_ffmpeg_available(monkeypatch):
    monkeypatch.setattr("hbmon.worker.shutil.which", lambda name: None)
    assert _ffmpeg_available() is False

    monkeypatch.setattr("hbmon.worker.shutil.which", lambda name: "/usr/bin/ffmpeg")
    assert _ffmpeg_available() is True


def test_convert_to_h264_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr("hbmon.worker._ffmpeg_available", lambda: True)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1)

    monkeypatch.setattr("hbmon.worker.subprocess.run", fake_run)
    assert not _convert_to_h264(tmp_path / "in.mp4", tmp_path / "out.mp4")


def test_convert_to_h264_success(monkeypatch, tmp_path):
    monkeypatch.setattr("hbmon.worker._ffmpeg_available", lambda: True)
    output_path = tmp_path / "out.mp4"
    output_path.write_bytes(b"ok")

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr("hbmon.worker.subprocess.run", fake_run)
    assert _convert_to_h264(tmp_path / "in.mp4", output_path)


def test_draw_bbox_requires_cv2(monkeypatch):
    monkeypatch.setattr("hbmon.worker._CV2_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="OpenCV"):
        _draw_bbox(np.zeros((2, 2, 3), dtype=np.uint8), Det(x1=0, y1=0, x2=1, y2=1, conf=0.5))


def test_draw_text_lines_requires_cv2(monkeypatch):
    monkeypatch.setattr("hbmon.worker._CV2_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="OpenCV"):
        _draw_text_lines(np.zeros((2, 2, 3), dtype=np.uint8), ["line"])


def test_draw_helpers_with_stubbed_cv2(monkeypatch):
    import hbmon.worker as worker

    class RecorderCV2:
        FONT_HERSHEY_SIMPLEX = 1

        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple]] = []

        def rectangle(self, *args, **kwargs):
            self.calls.append(("rectangle", args))
            return None

        def putText(self, *args, **kwargs):
            self.calls.append(("putText", args))
            return None

        def getTextSize(self, text, font, font_scale, thickness):
            return (len(text) * 8, 12), None

    recorder = RecorderCV2()
    monkeypatch.setattr(worker, "_CV2_AVAILABLE", True)
    monkeypatch.setattr(worker, "cv2", recorder)

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    det = Det(x1=1, y1=2, x2=5, y2=6, conf=0.7)
    _draw_bbox(frame, det)
    _draw_text_lines(frame, ["one", "two"])

    rectangles = [call for call in recorder.calls if call[0] == "rectangle"]
    texts = [call for call in recorder.calls if call[0] == "putText"]
    assert len(rectangles) == 4
    assert len(texts) == 3


def test_write_jpeg_and_mask_images(monkeypatch, tmp_path):
    import hbmon.worker as worker

    class DummyBuf:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def tobytes(self) -> bytes:
            return self._payload

    class DummyCV2:
        IMWRITE_JPEG_QUALITY = 1
        INTER_NEAREST = 2
        INTER_AREA = 3
        FONT_HERSHEY_SIMPLEX = 4

        @staticmethod
        def imencode(ext: str, image, params=None):
            return True, DummyBuf(b"encoded")

        @staticmethod
        def resize(image, size, interpolation=None):
            height, width = size[1], size[0]
            return np.zeros((height, width) + image.shape[2:], dtype=image.dtype)

        @staticmethod
        def addWeighted(src1, alpha, src2, beta, gamma):
            return src1

        @staticmethod
        def rectangle(*args, **kwargs):
            return None

        @staticmethod
        def getTextSize(text, font, font_scale, thickness):
            return (len(text) * 8, 12), None

        @staticmethod
        def putText(*args, **kwargs):
            return None

    monkeypatch.setattr(worker, "_CV2_AVAILABLE", True)
    monkeypatch.setattr(worker, "cv2", DummyCV2)

    out_path = tmp_path / "snapshots" / "frame.jpg"
    _write_jpeg(out_path, np.zeros((4, 4, 3), dtype=np.uint8))
    assert out_path.exists()

    png_path = tmp_path / "snapshots" / "frame.png"
    _write_png(png_path, np.zeros((4, 4, 3), dtype=np.uint8))
    assert png_path.exists()

    mask_path = tmp_path / "masks" / "mask.png"
    overlay_path = tmp_path / "masks" / "overlay.png"
    _save_motion_mask_images(
        motion_mask=np.array([[0, 1], [1, 0]], dtype=np.uint8),
        roi_frame=np.zeros((2, 2, 3), dtype=np.uint8),
        mask_path=mask_path,
        overlay_path=overlay_path,
        downscale_max=1,
    )
    assert mask_path.exists()
    assert overlay_path.exists()

    annotated = _draw_text_lines(
        np.zeros((10, 10, 3), dtype=np.uint8),
        ["line-1", "line-2"],
    )
    assert annotated.shape == (10, 10, 3)

    new_dir = tmp_path / "nested" / "path"
    _safe_mkdir(new_dir)
    assert new_dir.exists()


def test_motion_mask_and_overlap_stats(monkeypatch):
    import hbmon.worker as worker

    class DummyCV2:
        COLOR_BGR2GRAY = 0
        THRESH_BINARY = 0
        MORPH_ELLIPSE = 0
        MORPH_OPEN = 1
        MORPH_CLOSE = 2

        @staticmethod
        def resize(image, size):
            height, width = size[1], size[0]
            if image.ndim == 2:
                return np.zeros((height, width), dtype=image.dtype)
            return np.zeros((height, width, image.shape[2]), dtype=image.dtype)

        @staticmethod
        def cvtColor(image, code):
            if image.ndim == 3:
                return image[:, :, 0]
            return image

        @staticmethod
        def GaussianBlur(image, ksize, sigma):
            return image

        @staticmethod
        def absdiff(a, b):
            return np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)

        @staticmethod
        def threshold(diff, threshold, maxval, type):
            mask = (diff > threshold).astype(np.uint8) * maxval
            return None, mask

        @staticmethod
        def getStructuringElement(shape, ksize):
            return np.ones(ksize, dtype=np.uint8)

        @staticmethod
        def morphologyEx(mask, op, kernel):
            return mask

    monkeypatch.setattr(worker, "_CV2_AVAILABLE", True)
    monkeypatch.setattr(worker, "cv2", DummyCV2)

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[1:3, 1:3] = 255
    background = np.zeros((4, 4, 3), dtype=np.uint8)

    mask = _compute_motion_mask(frame, background, threshold=10, blur_size=3)
    assert mask.shape == (4, 4)
    det = Det(x1=0, y1=0, x2=4, y2=4, conf=0.9)
    stats = _motion_overlap_stats(det, mask)
    assert stats["bbox_total_pixels"] == 16
    roi_stats = _roi_motion_stats(mask)
    assert roi_stats["roi_total_pixels"] == 16
    assert roi_stats["roi_motion_pixels"] >= 0


def test_pick_and_collect_detections_with_motion_mask():
    class DummyScalar:
        def __init__(self, value: float) -> None:
            self._value = value

        def item(self) -> float:
            return float(self._value)

    class DummyTensor:
        def __init__(self, values) -> None:
            self._values = np.array(values, dtype=np.float32)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._values

    class DummyBox:
        def __init__(self, cls_id: int, conf: float, xyxy) -> None:
            self.cls = DummyScalar(cls_id)
            self.conf = DummyScalar(conf)
            self.xyxy = [DummyTensor(xyxy)]

    class DummyResults:
        def __init__(self, boxes) -> None:
            self.boxes = boxes

    results = [DummyResults([
        DummyBox(14, 0.9, [0, 0, 4, 4]),
        DummyBox(0, 0.5, [0, 0, 2, 2]),
        DummyBox(14, 0.2, [0, 0, 2, 2]),
    ])]
    motion_mask = np.zeros((5, 5), dtype=np.uint8)
    motion_mask[0:4, 0:4] = 255

    best = _pick_best_bird_det(results, min_box_area=5, bird_class_id=14, motion_mask=motion_mask)
    assert best is not None
    assert best.area >= 16

    detections = _collect_bird_detections(results, min_box_area=1, bird_class_id=14)
    assert len(detections) == 2

    det = Det(x1=0, y1=0, x2=2, y2=2, conf=0.4)
    selected = _select_best_detection([(det, {"bbox_overlap_ratio": 0.2})])
    assert selected[0].conf == 0.4


def test_load_background_image_reads_file(monkeypatch, tmp_path):
    import hbmon.worker as worker

    class DummyCV2:
        @staticmethod
        def imread(path: str):
            return np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(worker, "_CV2_AVAILABLE", True)
    monkeypatch.setattr(worker, "cv2", DummyCV2)
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))

    bg_path = worker.background_image_path()
    bg_path.parent.mkdir(parents=True, exist_ok=True)
    bg_path.write_bytes(b"fake")

    img = _load_background_image()
    assert img is not None


def test_load_background_image_handles_missing(monkeypatch, tmp_path):
    import hbmon.worker as worker

    monkeypatch.setattr(worker, "_CV2_AVAILABLE", True)
    monkeypatch.setattr(worker, "cv2", type("DummyCV2", (), {"imread": lambda *_: None}))
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))

    img = _load_background_image()
    assert img is None


def test_load_background_image_without_cv2(monkeypatch):
    import hbmon.worker as worker

    monkeypatch.setattr(worker, "_CV2_AVAILABLE", False)
    assert _load_background_image() is None


def test_load_background_image_handles_imread_error(monkeypatch, tmp_path):
    import hbmon.worker as worker

    class DummyCV2:
        @staticmethod
        def imread(path: str):
            raise RuntimeError("boom")

    monkeypatch.setattr(worker, "_CV2_AVAILABLE", True)
    monkeypatch.setattr(worker, "cv2", DummyCV2)
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))

    bg_path = worker.background_image_path()
    bg_path.parent.mkdir(parents=True, exist_ok=True)
    bg_path.write_bytes(b"fake")

    img = _load_background_image()
    assert img is None
