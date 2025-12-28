from __future__ import annotations

import sys
import time

import numpy as np

import hbmon.web as web


class _DummyJpeg:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def tobytes(self) -> bytes:
        return self._payload


class _DummyCV2:
    IMWRITE_JPEG_QUALITY = 1
    CAP_PROP_BUFFERSIZE = 2
    _open_calls = 0

    class VideoCapture:
        def __init__(self, url: str):
            _DummyCV2._open_calls += 1
            self._opened = _DummyCV2._open_calls == 1
            self._reads = 0

        def isOpened(self) -> bool:
            return self._opened

        def read(self):
            self._reads += 1
            if self._opened and self._reads == 1:
                return True, np.zeros((4, 4, 3), dtype=np.uint8)
            return False, None

        def release(self) -> None:
            return None

        def set(self, *args, **kwargs) -> None:
            return None

    @staticmethod
    def imencode(ext: str, frame, params):
        return True, _DummyJpeg(b"jpeg-bytes")


def _wait_for(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_load_cv2_prefers_existing_module(monkeypatch) -> None:
    _DummyCV2._open_calls = 0
    monkeypatch.setitem(sys.modules, "cv2", _DummyCV2)
    assert web._load_cv2() is _DummyCV2


def test_frame_broadcaster_caches_frames_and_stops(monkeypatch) -> None:
    _DummyCV2._open_calls = 0
    monkeypatch.setitem(sys.modules, "cv2", _DummyCV2)
    settings = web.MJPEGSettings(
        target_fps=30.0,
        max_width=0,
        max_height=0,
        base_quality=70,
        adaptive_enabled=False,
        min_fps=4.0,
        min_quality=40,
        fps_step=1.0,
        quality_step=5,
    )
    broadcaster = web.FrameBroadcaster("rtsp://example", settings=settings)
    broadcaster.add_client()

    assert _wait_for(lambda: broadcaster.latest_frame()[0] is not None)
    frame, timestamp = broadcaster.latest_frame()
    assert frame == b"jpeg-bytes"
    assert timestamp > 0

    broadcaster.remove_client()
    assert _wait_for(
        lambda: broadcaster._thread is None or not broadcaster._thread.is_alive(),  # noqa: SLF001
        timeout=3.0,
    )


def test_get_frame_broadcaster_reuses_and_switches(monkeypatch) -> None:
    _DummyCV2._open_calls = 0
    monkeypatch.setitem(sys.modules, "cv2", _DummyCV2)
    web._shutdown_frame_broadcaster()

    broadcaster = web._get_frame_broadcaster("rtsp://one")
    same = web._get_frame_broadcaster("rtsp://one")
    assert broadcaster is same

    other = web._get_frame_broadcaster("rtsp://two")
    assert other is not broadcaster
    assert other.rtsp_url == "rtsp://two"

    web._shutdown_frame_broadcaster()
