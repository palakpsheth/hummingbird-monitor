from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import func, select
import asyncio

from hbmon import db as db_module
from hbmon.config import Settings
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
    def __init__(self):
        self.cls = _DummyScalar(0)
        self.conf = _DummyScalar(0.95)
        self.xyxy = _DummyTensor(np.array([[2, 2, 12, 12]], dtype=np.float32))


class _DummyResult:
    def __init__(self):
        self.boxes = [_DummyBox()]


class _DummyYOLO:
    # Use dict to allow bird_class_id resolution to find 'bird' = 0
    names = {0: "bird"}

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    def predict(self, frame, *, conf, iou, classes, verbose=False, **kwargs):
        return [_DummyResult()]


class _DummyClip:
    def __init__(self, device: str | None = None, backend: str | None = None):
        # Accept both device and backend for backward compatibility
        self.backend = backend or device or "cpu"
        self.device = self.backend
        self.labels: list[str] | None = None
        # Add model metadata attributes expected by worker
        self.model_name = "dummy-clip-model"
        self.pretrained = None

    def set_label_space(self, labels: list[str]) -> None:
        self.labels = labels

    def predict_species_label_prob(self, crop):
        return "Anna's Hummingbird", 0.95

    def encode_embedding(self, crop):
        return np.ones(4, dtype=np.float32)


class _DummyCap:
    def __init__(self, url: str):
        self.url = url
        self._read_calls = 0

    def isOpened(self) -> bool:
        return True

    def read(self):
        print("DEBUG: _DummyCap.read called")
        self._read_calls += 1
        if self._read_calls == 1:
            frame = np.zeros((20, 20, 3), dtype=np.uint8)
            return True, frame
        raise RuntimeError("stop loop")

    def release(self) -> None:
        return None

    def set(self, *args, **kwargs):
        return True


class _DummyCV2:
    CAP_PROP_BUFFERSIZE = 38
    IMWRITE_JPEG_QUALITY = 1

    VideoCapture = _DummyCap




class _RejectBox:
    def __init__(self):
        self.cls = _DummyScalar(0)
        self.conf = _DummyScalar(0.7)
        self.xyxy = _DummyTensor(np.array([[1, 1, 6, 6]], dtype=np.float32))


class _RejectResult:
    def __init__(self):
        self.boxes = [_RejectBox()]


class _RejectYOLO:
    names = {0: "bird"}

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    def predict(self, frame, *, conf, iou, classes, verbose=False, **kwargs):
        return [_RejectResult()]


@pytest.mark.anyio
async def test_run_worker_logs_rejected_candidate(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    media_dir = tmp_path / "media"
    db_path = tmp_path / "db.sqlite"

    monkeypatch.setenv("HBMON_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(media_dir))
    monkeypatch.setenv("HBMON_DB_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("HBMON_DB_ASYNC_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("HBMON_RTSP_URL", "rtsp://example")
    monkeypatch.setenv("HBMON_BG_LOG_REJECTED", "1")
    monkeypatch.setenv("HBMON_BG_REJECTED_SAVE_CLIP", "0")

    db_module.reset_db_state()

    monkeypatch.setattr(worker, "_CV2_AVAILABLE", True)
    monkeypatch.setattr(worker, "_YOLO_AVAILABLE", True)
    monkeypatch.setattr(worker, "cv2", _DummyCV2)
    monkeypatch.setattr(worker, "YOLO", _RejectYOLO)
    monkeypatch.setattr(worker, "ClipModel", _DummyClip)
    monkeypatch.setattr(worker, "_write_jpeg", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "_draw_bbox", lambda frame, det, **kwargs: frame)
    monkeypatch.setattr(worker, "_draw_text_lines", lambda frame, lines, **kwargs: frame)
    monkeypatch.setattr(worker, "_save_motion_mask_images", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "_compute_motion_mask", lambda *args, **kwargs: np.zeros((20, 20), dtype=np.uint8))
    monkeypatch.setattr(worker, "_load_background_image", lambda: np.zeros((20, 20, 3), dtype=np.uint8))

    async def _noop_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(worker.asyncio, "sleep", _noop_sleep)
    
    queue_capture = []
    def _capture_dispatcher_eager(q, c):
        queue_capture.append(q)
        async def dummy(): pass
        return dummy()

    monkeypatch.setattr(worker, "processing_dispatcher", _capture_dispatcher_eager)
    monkeypatch.setattr("hbmon.worker.processing_dispatcher", _capture_dispatcher_eager)

    settings = Settings()
    settings.rtsp_url = "rtsp://example"
    settings.camera_name = "camera"
    settings.fps_limit = 0.0
    settings.clip_seconds = 0.1
    settings.detect_conf = 0.2
    settings.detect_iou = 0.5
    settings.min_box_area = 1
    settings.cooldown_seconds = 0.0
    settings.min_species_prob = 0.0
    settings.match_threshold = 0.5
    settings.ema_alpha = 0.2
    settings.crop_padding = 0.0
    settings.bg_subtraction_enabled = True
    settings.bg_motion_threshold = 25
    settings.bg_motion_blur = 3
    settings.bg_min_overlap = 0.5
    monkeypatch.setattr(worker, "load_settings", lambda apply_env_overrides=False: settings)

    with pytest.raises(RuntimeError, match="stop loop"):
        await worker.run_worker()
        
    assert len(queue_capture) == 1
    captured_queue = queue_capture[0]
    assert captured_queue.qsize() == 1
    item = captured_queue.get_nowait()
    
    sem = asyncio.Semaphore(1)
    clip = _DummyClip(device="cpu")
    await worker.process_candidate_task(item, clip, media_dir, sem)

    async with db_module.async_session_scope() as session:
        # Rejected candidates go to Candidate table
        total = (await session.execute(select(func.count(worker.Candidate.id)))).scalar_one()
        assert total == 1
