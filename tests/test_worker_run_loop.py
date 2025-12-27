from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import func, select

from hbmon import db as db_module
from hbmon.config import Settings
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
    def __init__(self):
        self.cls = _DummyScalar(0)
        self.conf = _DummyScalar(0.95)
        self.xyxy = _DummyTensor(np.array([[2, 2, 12, 12]], dtype=np.float32))


class _DummyResult:
    def __init__(self):
        self.boxes = [_DummyBox()]


class _DummyYOLO:
    names = ["bird"]

    def __init__(self, model_name: str):
        self.model_name = model_name

    def predict(self, frame, *, conf, iou, classes, imgsz, verbose):
        return [_DummyResult()]


class _DummyClip:
    def __init__(self, device: str):
        self.device = device
        self.labels: list[str] | None = None

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


@pytest.mark.anyio
async def test_run_worker_single_iteration(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    media_dir = tmp_path / "media"
    db_path = tmp_path / "db.sqlite"

    monkeypatch.setenv("HBMON_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(media_dir))
    monkeypatch.setenv("HBMON_DB_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("HBMON_DB_ASYNC_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("HBMON_RTSP_URL", "rtsp://example")
    monkeypatch.setenv("HBMON_CAMERA_NAME", "envcam")
    monkeypatch.setenv("HBMON_BG_SUBTRACTION", "1")
    monkeypatch.setenv("HBMON_BG_MOTION_THRESHOLD", "25")
    monkeypatch.setenv("HBMON_BG_MOTION_BLUR", "3")
    monkeypatch.setenv("HBMON_BG_MIN_OVERLAP", "0.2")
    monkeypatch.setenv("HBMON_SPECIES_LIST", "Anna's, Rufous")

    db_module.reset_db_state()

    monkeypatch.setattr(worker, "_CV2_AVAILABLE", True)
    monkeypatch.setattr(worker, "_YOLO_AVAILABLE", True)
    monkeypatch.setattr(worker, "cv2", _DummyCV2)
    monkeypatch.setattr(worker, "YOLO", _DummyYOLO)
    monkeypatch.setattr(worker, "ClipModel", _DummyClip)
    monkeypatch.setattr(worker, "_load_background_image", lambda: None)
    monkeypatch.setattr(worker, "_write_jpeg", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "_draw_bbox", lambda frame, det, **kwargs: frame)
    monkeypatch.setattr(
        worker,
        "_record_clip_from_rtsp",
        lambda rtsp_url, out_path, seconds, max_fps=20.0: out_path,
    )
    async def _noop_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(worker.asyncio, "sleep", _noop_sleep)

    settings = Settings()
    settings.rtsp_url = ""
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
    settings.bg_subtraction_enabled = False
    monkeypatch.setattr(worker, "load_settings", lambda apply_env_overrides=False: settings)

    with pytest.raises(RuntimeError, match="stop loop"):
        await worker.run_worker()

    async with db_module.async_session_scope() as session:
        total = (await session.execute(select(func.count(Observation.id)))).scalar_one()
        assert total == 1
