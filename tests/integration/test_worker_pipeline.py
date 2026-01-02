"""
Integration-style tests for the worker pipeline using stubbed ML and recorder components.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import func, select

from hbmon import db as db_module
from hbmon.config import Roi, Settings
from hbmon.models import Observation
import hbmon.worker as worker


class _DummyScalar:
    def __init__(self, value: float):
        self._value = value

    def item(self) -> float:
        return self._value


class _DummyTensor:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def __getitem__(self, idx):
        return _DummyTensor(self._arr[idx])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _DummyBox:
    def __init__(self, x1: int, y1: int, x2: int, y2: int, conf: float):
        self.cls = _DummyScalar(0)
        self.conf = _DummyScalar(conf)
        self.xyxy = _DummyTensor(np.array([[x1, y1, x2, y2]], dtype=np.float32))


class _DummyResult:
    def __init__(self, boxes: list[_DummyBox]):
        self.boxes = boxes


class _DummyYOLO:
    names = {0: "bird"}

    def __init__(self, detect_sequence: list[bool], frame_shapes: list[tuple[int, int, int]]):
        self.detect_sequence = detect_sequence
        self.frame_shapes = frame_shapes
        self.calls = 0

    def predict(self, frame, *, conf, iou, classes, verbose=False, **kwargs):
        self.frame_shapes.append(frame.shape)
        detect = self.detect_sequence[self.calls] if self.calls < len(self.detect_sequence) else False
        self.calls += 1
        if not detect:
            return [_DummyResult([])]
        return [_DummyResult([_DummyBox(2, 2, 8, 8, 0.95)])]


class _DummyClip:
    def __init__(self, device: str | None = None, backend: str | None = None):
        self.backend = backend or device or "cpu"
        self.device = self.backend
        self.model_name = "dummy-clip-model"
        self.pretrained = None

    def set_label_space(self, labels: list[str]) -> None:
        return None

    def predict_species_label_prob(self, crop):
        return "Anna's Hummingbird", 0.92

    def encode_embedding(self, crop):
        return np.ones(4, dtype=np.float32)


class _DummyBackgroundRecorder:
    instances: list["_DummyBackgroundRecorder"] = []

    def __init__(self, out_path: Path, fps: float, width: int, height: int):
        self.out_path = out_path
        self.fps = fps
        self.width = width
        self.height = height
        self.start_calls = 0
        self.stop_calls = 0
        self.feed_calls = 0
        _DummyBackgroundRecorder.instances.append(self)

    def start(self) -> None:
        self.start_calls += 1

    def feed(self, frame) -> None:
        self.feed_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


class _DummyCap:
    def __init__(self, frames: list[np.ndarray]):
        self.frames = frames
        self.index = 0

    def isOpened(self) -> bool:
        return True

    def read(self):
        if self.index < len(self.frames):
            frame = self.frames[self.index]
            self.index += 1
            return True, frame
        raise RuntimeError("stop loop")

    def release(self) -> None:
        return None

    def set(self, *args, **kwargs):
        return True


class _DummyCV2:
    CAP_PROP_BUFFERSIZE = 38
    IMWRITE_JPEG_QUALITY = 1

    def __init__(self, frames: list[np.ndarray]):
        self._frames = frames

    def VideoCapture(self, _url: str):
        return _DummyCap(self._frames)


@pytest.mark.anyio
async def test_worker_pipeline_roi_cooldown_and_recorder(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    media_dir = tmp_path / "media"
    db_path = tmp_path / "db.sqlite"

    monkeypatch.setenv("HBMON_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(media_dir))
    monkeypatch.setenv("HBMON_DB_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("HBMON_DB_ASYNC_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("HBMON_RTSP_URL", "rtsp://example")
    monkeypatch.setenv("HBMON_CAMERA_NAME", "envcam")
    monkeypatch.setenv("HBMON_BG_SUBTRACTION", "0")

    db_module.reset_db_state()

    frames = [np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(7)]
    detect_sequence = [True, True, False, False, True, False, False]
    yolo_shapes: list[tuple[int, int, int]] = []
    dummy_yolo = _DummyYOLO(detect_sequence, yolo_shapes)

    monkeypatch.setattr(worker, "_CV2_AVAILABLE", True)
    monkeypatch.setattr(worker, "_YOLO_AVAILABLE", True)
    monkeypatch.setattr(worker, "cv2", _DummyCV2(frames))
    monkeypatch.setattr(worker, "ClipModel", _DummyClip)
    monkeypatch.setattr(worker, "BackgroundRecorder", _DummyBackgroundRecorder)
    monkeypatch.setattr(worker, "_load_background_image", lambda: None)
    monkeypatch.setattr(worker, "_draw_bbox", lambda frame, det, **kwargs: frame)
    monkeypatch.setattr(worker, "_write_jpeg", lambda *args, **kwargs: None)

    async def _noop_write(*args, **kwargs):
        return None

    async def _noop_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(worker, "_write_jpeg_async", _noop_write)
    monkeypatch.setattr(worker.asyncio, "sleep", _noop_sleep)

    queue_capture: list[asyncio.Queue] = []

    async def _dummy_dispatcher(*args, **kwargs):
        return None

    def _capture_dispatcher(queue: asyncio.Queue, clip):
        queue_capture.append(queue)
        return _dummy_dispatcher()

    monkeypatch.setattr(worker, "processing_dispatcher", _capture_dispatcher)
    monkeypatch.setattr(worker, "_load_yolo_model", lambda: (dummy_yolo, "dummy-device"))

    settings = Settings()
    settings.rtsp_url = "rtsp://example"
    settings.camera_name = "camera"
    settings.fps_limit = 0.0
    settings.detect_conf = 0.2
    settings.detect_iou = 0.5
    settings.min_box_area = 1
    settings.cooldown_seconds = 100.0
    settings.min_species_prob = 0.0
    settings.match_threshold = 0.5
    settings.ema_alpha = 0.2
    settings.crop_padding = 0.0
    settings.bg_subtraction_enabled = False
    settings.temporal_window_frames = 1
    settings.arrival_buffer_seconds = 0.0
    settings.departure_timeout_seconds = 0.0
    settings.post_departure_buffer_seconds = 0.0
    settings.roi = Roi(0.25, 0.25, 0.75, 0.75)
    monkeypatch.setattr(worker, "load_settings", lambda apply_env_overrides=False: settings)

    fake_time = {"value": 1000.0}

    def _fake_time() -> float:
        fake_time["value"] += 0.1
        return fake_time["value"]

    monkeypatch.setattr(worker.time, "time", _fake_time)

    with pytest.raises(RuntimeError, match="stop loop"):
        await worker.run_worker()

    assert queue_capture
    queue = queue_capture[0]
    assert queue.qsize() == 1

    sem = asyncio.Semaphore(1)
    clip = _DummyClip(device="cpu")
    item = queue.get_nowait()
    await worker.process_candidate_task(item, clip, media_dir, sem)

    async with db_module.async_session_scope() as session:
        total = (await session.execute(select(func.count(Observation.id)))).scalar_one()
        assert total == 1

    assert all(shape == (10, 10, 3) for shape in yolo_shapes)

    assert _DummyBackgroundRecorder.instances
    recorder = _DummyBackgroundRecorder.instances[0]
    assert recorder.start_calls == 1
    assert recorder.stop_calls == 1
    assert recorder.feed_calls > 0
