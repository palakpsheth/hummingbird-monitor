from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import hbmon.web as web
from hbmon.config import ensure_dirs
from hbmon.db import init_db, reset_db_state


pytestmark = pytest.mark.integration


def test_stream_mjpeg_returns_503_when_cv2_unavailable(monkeypatch: pytest.MonkeyPatch):
    reset_db_state()
    ensure_dirs()
    init_db()

    web._shutdown_frame_broadcaster()
    web._FRAME_BROADCASTER = None
    web._CV2_MODULE = None

    monkeypatch.setenv("HBMON_RTSP_URL", "stub://camera")

    def _raise():
        raise RuntimeError("OpenCV (cv2) is not installed")

    monkeypatch.setattr(web, "_load_cv2", _raise)

    app = web.make_app()
    with TestClient(app) as client:
        with pytest.raises(RuntimeError):
            client.get("/api/stream.mjpeg")
