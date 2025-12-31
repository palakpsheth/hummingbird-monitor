from __future__ import annotations

from pathlib import Path
from concurrent.futures import Future
import io
import json
import re
import sys
import tarfile
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pytest
from sqlalchemy import select

from hbmon.config import background_image_path, ensure_dirs, load_settings, media_dir
from hbmon.db import init_db, reset_db_state, session_scope
from hbmon.models import Embedding, Individual, Observation, utcnow
from hbmon.web import make_app

_TEST_CLIENTS: list[TestClient] = []


@pytest.fixture(autouse=True)
def _close_test_clients() -> None:
    yield
    while _TEST_CLIENTS:
        client = _TEST_CLIENTS.pop()
        client.close()


def _has_hidden_column(text: str, tag: str, column_key: str) -> bool:
    pattern = (
        rf"<{tag}"
        rf"(?=[^>]*data-col-key=\"{re.escape(column_key)}\")"
        r"(?=[^>]*col-hidden)"
        r"(?=[^>]*\bhidden\b)"
        r"[^>]*>"
    )
    return re.search(pattern, text) is not None


def _setup_app(tmp_path: Path, monkeypatch) -> TestClient:
    data_dir = tmp_path / "data"
    media = tmp_path / "media"
    db_path = tmp_path / "db.sqlite"
    monkeypatch.setenv("HBMON_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(media))
    monkeypatch.setenv("HBMON_DB_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("HBMON_DB_ASYNC_URL", f"sqlite+aiosqlite:///{db_path}")
    reset_db_state()
    ensure_dirs()
    init_db()
    app = make_app()
    client = TestClient(app)
    _TEST_CLIENTS.append(client)
    return client


def _setup_app_sync(tmp_path: Path, monkeypatch) -> TestClient:
    data_dir = tmp_path / "data"
    media = tmp_path / "media"
    db_path = tmp_path / "db.sqlite"
    monkeypatch.setenv("HBMON_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(media))
    monkeypatch.setenv("HBMON_DB_URL", f"sqlite:///{db_path}")
    monkeypatch.delenv("HBMON_DB_ASYNC_URL", raising=False)
    reset_db_state()
    ensure_dirs()
    init_db()
    app = make_app()
    client = TestClient(app)
    _TEST_CLIENTS.append(client)
    return client


def _install_live_cv2_stub(monkeypatch) -> None:
    class DummyCap:
        def __init__(self, url: str):
            self.url = url

        def isOpened(self) -> bool:
            return True

        def read(self):
            return True, object()

        def release(self) -> None:
            return None

    class DummyJpeg:
        def __init__(self, payload: bytes):
            self._payload = payload

        def tobytes(self) -> bytes:
            return self._payload

    def fake_imencode(ext: str, frame, params):
        return True, DummyJpeg(b"jpeg-bytes")

    stub_cv2 = SimpleNamespace(
        VideoCapture=DummyCap,
        imencode=fake_imencode,
        IMWRITE_JPEG_QUALITY=1,
    )

    monkeypatch.setattr("hbmon.web.importlib.util.find_spec", lambda name: object())
    monkeypatch.setitem(sys.modules, "cv2", stub_cv2)


def test_label_observation_and_clear(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    with session_scope() as db:
        obs = Observation(
            species_label="Hummingbird",
            species_prob=0.5,
            snapshot_path="snap.jpg",
            video_path="vid.mp4",
        )
        db.add(obs)
        db.commit()
        obs_id = obs.id

    r = client.post(f"/observations/{obs_id}/label", data={"label": "true_positive"}, follow_redirects=False)
    assert r.status_code == 303

    with session_scope() as db:
        o = db.get(Observation, obs_id)
        assert o is not None
        assert o.review_label == "true_positive"

    r = client.post(f"/observations/{obs_id}/label", data={"label": " "}, follow_redirects=False)
    assert r.status_code == 303
    with session_scope() as db:
        o = db.get(Observation, obs_id)
        assert o is not None
        extra = o.get_extra() or {}
        assert "review" not in extra or not extra["review"]


def test_candidates_endpoint_empty(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)

    response = client.get("/candidates")
    assert response.status_code == 200


def test_health_endpoint_sync_fallback(tmp_path, monkeypatch):
    import hbmon.db as db_module

    monkeypatch.setattr(db_module, "_ASYNC_SQLALCHEMY_AVAILABLE", False)
    monkeypatch.setattr(db_module, "_SQLALCHEMY_AVAILABLE", True)
    client = _setup_app_sync(tmp_path, monkeypatch)

    response = client.get("/api/health")
    assert response.status_code == 200


def test_sync_fallback_adapter_executes_queries(tmp_path, monkeypatch):
    import hbmon.db as db_module

    monkeypatch.setattr(db_module, "_ASYNC_SQLALCHEMY_AVAILABLE", False)
    monkeypatch.setattr(db_module, "_SQLALCHEMY_AVAILABLE", True)
    client = _setup_app_sync(tmp_path, monkeypatch)

    with session_scope() as db:
        obs = Observation(
            ts=utcnow(),
            species_label="Ruby-throated Hummingbird",
            species_prob=0.9,
            snapshot_path="snap.jpg",
            video_path="vid.mp4",
        )
        db.add(obs)
        db.commit()
        obs_id = obs.id

    response = client.get("/observations")
    assert response.status_code == 200
    assert "Ruby-throated Hummingbird" in response.text

    response = client.get(f"/observations/{obs_id}")
    assert response.status_code == 200
    assert "Ruby-throated Hummingbird" in response.text


def test_sync_fallback_adapter_allows_attribute_access(tmp_path, monkeypatch):
    import asyncio
    import hbmon.db as db_module
    import hbmon.web as web_module

    monkeypatch.setattr(db_module, "_ASYNC_SQLALCHEMY_AVAILABLE", False)
    monkeypatch.setattr(db_module, "_SQLALCHEMY_AVAILABLE", True)
    reset_db_state()
    ensure_dirs()
    init_db()

    adapter = web_module._AsyncSessionAdapter(db_module.get_session_factory())
    assert adapter.bind is not None
    asyncio.run(adapter.close())

@pytest.mark.anyio
async def test_sync_fallback_adapter_cleanup_runs(tmp_path, monkeypatch):
    import hbmon.db as db_module
    import hbmon.web as web_module

    class DummyExecutor:
        def __init__(self, max_workers: int = 1):
            self.shutdown_called = False

        def submit(self, fn, *args, **kwargs):
            fut: Future = Future()
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                fut.set_exception(exc)
            else:
                fut.set_result(result)
            return fut

        def shutdown(self, wait: bool = True) -> None:
            self.shutdown_called = True

    created: list[DummyExecutor] = []

    def _executor_factory(max_workers: int = 1) -> DummyExecutor:
        executor = DummyExecutor(max_workers=max_workers)
        created.append(executor)
        return executor

    monkeypatch.setattr(db_module, "_ASYNC_SQLALCHEMY_AVAILABLE", False)
    monkeypatch.setattr(db_module, "_SQLALCHEMY_AVAILABLE", True)
    monkeypatch.setattr(web_module, "ThreadPoolExecutor", _executor_factory)
    reset_db_state()
    ensure_dirs()
    init_db()

    async for adapter in web_module.get_db_dep():
        await adapter.execute(select(Observation).limit(1))

    assert created
    assert created[-1].shutdown_called is True


def test_shutdown_async_session_executors_handles_cancel_futures(monkeypatch):
    import hbmon.web as web_module

    class DummyExecutor:
        def __init__(self) -> None:
            self.shutdown_calls: list[dict[str, bool]] = []

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            self.shutdown_calls.append({"wait": wait, "cancel_futures": cancel_futures})

    dummy = DummyExecutor()
    monkeypatch.setattr(web_module._AsyncSessionAdapter, "_executors", {dummy})

    web_module._shutdown_async_session_executors()

    assert dummy.shutdown_calls == [{"wait": False, "cancel_futures": True}]


def test_shutdown_async_session_executors_fallbacks_without_cancel_futures(monkeypatch):
    import hbmon.web as web_module

    class DummyExecutor:
        def __init__(self) -> None:
            self.shutdown_calls: list[dict[str, bool]] = []

        def shutdown(self, *, wait: bool) -> None:
            self.shutdown_calls.append({"wait": wait})

    dummy = DummyExecutor()
    monkeypatch.setattr(web_module._AsyncSessionAdapter, "_executors", {dummy})

    web_module._shutdown_async_session_executors()

    assert dummy.shutdown_calls == [{"wait": False}]


def test_delete_observation_cleans_media_and_stats(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    mdir = media_dir()
    snap = mdir / "a.jpg"
    vid = mdir / "b.mp4"
    snap.write_text("x")
    vid.write_text("x")

    with session_scope() as db:
        ind = Individual(name="i1", visit_count=0, last_seen_at=None)
        db.add(ind)
        db.commit()
        ind_id = ind.id
        obs = Observation(
            individual_id=ind.id,
            species_label="Hummingbird",
            species_prob=0.5,
            snapshot_path="a.jpg",
            video_path="b.mp4",
            ts=utcnow(),
        )
        db.add(obs)
        db.commit()
        db.add(Embedding(observation_id=obs.id, individual_id=ind.id, embedding_blob=b"123"))
        db.commit()
        obs_id = obs.id

    r = client.post(f"/observations/{obs_id}/delete", follow_redirects=False)
    assert r.status_code == 303

    assert not snap.exists()
    assert not vid.exists()

    with session_scope() as db:
        assert db.get(Observation, obs_id) is None
        refreshed = db.get(Individual, ind_id)
        assert refreshed is not None
        assert refreshed.visit_count == 0
        assert refreshed.last_seen_at is None


def test_delete_individual_cascades(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    mdir = media_dir()
    snap = mdir / "c.jpg"
    vid = mdir / "d.mp4"
    snap.write_text("x")
    vid.write_text("x")

    with session_scope() as db:
        ind = Individual(name="i1")
        db.add(ind)
        db.commit()
        obs = Observation(
            individual_id=ind.id,
            species_label="Hummingbird",
            species_prob=0.5,
            snapshot_path="c.jpg",
            video_path="d.mp4",
        )
        db.add(obs)
        db.commit()
        emb = Embedding(observation_id=obs.id, individual_id=ind.id, embedding_blob=b"123")
        db.add(emb)
        db.commit()
        ind_id = ind.id
        obs_id = obs.id
        emb_id = emb.id

    r = client.post(f"/individuals/{ind_id}/delete", follow_redirects=False)
    assert r.status_code == 303

    assert not snap.exists()
    assert not vid.exists()

    with session_scope() as db:
        assert db.get(Individual, ind_id) is None
        assert db.get(Observation, obs_id) is None
        assert db.get(Embedding, emb_id) is None


def test_bulk_delete_observations_cleans_media_and_stats(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    mdir = media_dir()
    snap1 = mdir / "bulk_a.jpg"
    snap2 = mdir / "bulk_b.jpg"
    vid1 = mdir / "bulk_a.mp4"
    vid2 = mdir / "bulk_b.mp4"
    snap1.write_text("x")
    snap2.write_text("x")
    vid1.write_text("x")
    vid2.write_text("x")

    with session_scope() as db:
        ind = Individual(name="bulk-ind", visit_count=0, last_seen_at=None)
        db.add(ind)
        db.commit()
        obs1 = Observation(
            individual_id=ind.id,
            species_label="Hummingbird",
            species_prob=0.5,
            snapshot_path="bulk_a.jpg",
            video_path="bulk_a.mp4",
            ts=utcnow(),
        )
        obs2 = Observation(
            individual_id=ind.id,
            species_label="Hummingbird",
            species_prob=0.6,
            snapshot_path="bulk_b.jpg",
            video_path="bulk_b.mp4",
            ts=utcnow(),
        )
        db.add_all([obs1, obs2])
        db.commit()
        db.add_all(
            [
                Embedding(observation_id=obs1.id, individual_id=ind.id, embedding_blob=b"123"),
                Embedding(observation_id=obs2.id, individual_id=ind.id, embedding_blob=b"456"),
            ]
        )
        db.commit()
        obs_ids = [obs1.id, obs2.id]
        ind_id = ind.id

    r = client.post(
        "/observations/bulk_delete",
        data={"obs_ids": obs_ids, "redirect_to": f"/individuals/{ind_id}"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == f"/individuals/{ind_id}"

    assert not snap1.exists()
    assert not snap2.exists()
    assert not vid1.exists()
    assert not vid2.exists()

    with session_scope() as db:
        for obs_id in obs_ids:
            assert db.get(Observation, obs_id) is None
        refreshed = db.get(Individual, ind_id)
        assert refreshed is not None
        assert refreshed.visit_count == 0
        assert refreshed.last_seen_at is None


def test_bulk_delete_redirect_path_sanitized(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    with session_scope() as db:
        obs = Observation(
            species_label="Hummingbird",
            species_prob=0.5,
            snapshot_path="snap.jpg",
            video_path="vid.mp4",
        )
        db.add(obs)
        db.commit()
        obs_id = obs.id

    r = client.post(
        "/observations/bulk_delete",
        data={"obs_ids": [obs_id], "redirect_to": "https://example.com/hijack"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/observations"

def test_video_info_endpoint_with_existing_file(tmp_path, monkeypatch):
    """Test /api/video_info/{obs_id} endpoint with an existing video file."""
    client = _setup_app(tmp_path, monkeypatch)
    mdir = media_dir()

    # Create a fake video file
    clips_dir = mdir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    vid = clips_dir / "test.mp4"
    video_content = b"fake video content 12345"
    vid.write_bytes(video_content)
    expected_size = len(video_content)

    with session_scope() as db:
        obs = Observation(
            species_label="Hummingbird",
            species_prob=0.5,
            snapshot_path="snap.jpg",
            video_path="clips/test.mp4",
        )
        db.add(obs)
        db.commit()
        obs_id = obs.id

    r = client.get(f"/api/video_info/{obs_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["observation_id"] == obs_id
    assert data["file_exists"] is True
    assert data["file_size_bytes"] == expected_size
    assert data["file_suffix"] == ".mp4"
    assert "clips/test.mp4" in data["video_path"]


def test_video_info_endpoint_with_missing_file(tmp_path, monkeypatch):
    """Test /api/video_info/{obs_id} endpoint when video file is missing."""
    client = _setup_app(tmp_path, monkeypatch)

    with session_scope() as db:
        obs = Observation(
            species_label="Hummingbird",
            species_prob=0.5,
            snapshot_path="snap.jpg",
            video_path="clips/nonexistent.mp4",
        )
        db.add(obs)
        db.commit()
        obs_id = obs.id

    r = client.get(f"/api/video_info/{obs_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["observation_id"] == obs_id
    assert data["file_exists"] is False
    assert "error" in data


def test_video_info_endpoint_not_found(tmp_path, monkeypatch):
    """Test /api/video_info/{obs_id} endpoint for non-existent observation."""
    client = _setup_app(tmp_path, monkeypatch)

    r = client.get("/api/video_info/99999")
    assert r.status_code == 404


def test_stream_mjpeg_returns_503_when_rtsp_not_configured(tmp_path, monkeypatch):
    """Test that /api/stream.mjpeg returns 503 when RTSP URL is not configured."""
    client = _setup_app(tmp_path, monkeypatch)
    # Clear RTSP URL to ensure it's not set
    monkeypatch.delenv("HBMON_RTSP_URL", raising=False)
    r = client.get("/api/stream.mjpeg")
    assert r.status_code == 503
    assert "RTSP URL not configured" in r.json()["detail"]


def test_stream_mjpeg_streams_single_frame(tmp_path, monkeypatch):
    """Test that /api/stream.mjpeg streams at least one frame when RTSP is configured."""
    import numpy as np

    monkeypatch.setenv("HBMON_RTSP_URL", "rtsp://example")

    class DummyJpeg:
        def __init__(self, payload: bytes):
            self._payload = payload

        def tobytes(self) -> bytes:
            return self._payload

    class DummyCV2:
        IMWRITE_JPEG_QUALITY = 1
        CAP_PROP_BUFFERSIZE = 2
        INTER_AREA = 3
        _open_calls = 0

        class VideoCapture:
            def __init__(self, url: str):
                DummyCV2._open_calls += 1
                self._opened = DummyCV2._open_calls == 1
                self._reads = 0

            def isOpened(self) -> bool:
                return self._opened

            def read(self):
                self._reads += 1
                if self._opened and self._reads == 1:
                    return True, np.zeros((8, 8, 3), dtype=np.uint8)
                return False, None

            def release(self) -> None:
                return None

            def set(self, *args, **kwargs) -> None:
                return None

        @staticmethod
        def imencode(ext: str, frame, params):
            return True, DummyJpeg(b"jpeg-bytes")

    monkeypatch.setitem(sys.modules, "cv2", DummyCV2)
    monkeypatch.setattr("hbmon.web.time.sleep", lambda *args, **kwargs: None)

    client = _setup_app(tmp_path, monkeypatch)
    with client.stream("GET", "/api/stream.mjpeg") as response:
        assert response.status_code == 200
        chunk = next(response.iter_bytes())
        assert b"--frame" in chunk


def test_stream_mjpeg_resizes_before_encoding(tmp_path, monkeypatch):
    """Test that MJPEG stream downsizes frames that exceed configured max dimensions."""
    import numpy as np
    from hbmon.web import MJPEGSettings, _resize_mjpeg_frame

    settings = MJPEGSettings(
        target_fps=10.0,
        max_width=4,
        max_height=4,
        base_quality=70,
        adaptive_enabled=False,
        min_fps=4.0,
        min_quality=40,
        fps_step=1.0,
        quality_step=5,
    )

    resized_to: list[tuple[int, int]] = []

    class DummyJpeg:
        def __init__(self, payload: bytes):
            self._payload = payload

        def tobytes(self) -> bytes:
            return self._payload

    class DummyCV2:
        IMWRITE_JPEG_QUALITY = 1
        CAP_PROP_BUFFERSIZE = 2
        INTER_AREA = 3
        _open_calls = 0

        class VideoCapture:
            def __init__(self, url: str):
                DummyCV2._open_calls += 1
                self._opened = DummyCV2._open_calls == 1
                self._reads = 0

            def isOpened(self) -> bool:
                return self._opened

            def read(self):
                self._reads += 1
                if self._opened and self._reads == 1:
                    return True, np.zeros((6, 8, 3), dtype=np.uint8)
                return False, None

            def release(self) -> None:
                return None

            def set(self, *args, **kwargs) -> None:
                return None

        @staticmethod
        def resize(frame, size, interpolation=None):
            resized_to.append(size)
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)

        @staticmethod
        def imencode(ext: str, frame, params):
            return True, DummyJpeg(b"jpeg-bytes")

    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    resized = _resize_mjpeg_frame(frame, DummyCV2, settings)

    assert resized.shape[:2] == (3, 4)
    assert resized_to == [(4, 3)]


def test_stream_mjpeg_adaptive_degrades_quality_on_slow_encode(tmp_path, monkeypatch):
    """Test that adaptive MJPEG mode lowers quality when encoding exceeds budget."""
    from hbmon.web import MJPEGSettings, _update_mjpeg_adaptive

    settings = MJPEGSettings(
        target_fps=10.0,
        max_width=0,
        max_height=0,
        base_quality=70,
        adaptive_enabled=True,
        min_fps=4.0,
        min_quality=40,
        fps_step=1.0,
        quality_step=5,
    )

    new_fps, new_quality = _update_mjpeg_adaptive(
        10.0,
        70,
        settings,
        encode_duration=1.0,
    )

    assert new_quality == 65
    assert new_fps == 9.0


def test_swagger_docs_endpoint(tmp_path, monkeypatch):
    """Test that the Swagger UI docs endpoint is accessible."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.get("/docs")
    assert r.status_code == 200
    assert "swagger-ui" in r.text.lower()


def test_redoc_endpoint(tmp_path, monkeypatch):
    """Test that the ReDoc docs endpoint is accessible."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.get("/redoc")
    assert r.status_code == 200
    assert "redoc" in r.text.lower()


def test_openapi_json_endpoint(tmp_path, monkeypatch):
    """Test that the OpenAPI JSON spec is accessible and has correct metadata."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.get("/openapi.json")
    assert r.status_code == 200
    data = r.json()
    assert data["info"]["title"] == "hbmon"
    assert "description" in data["info"]
    assert data["info"]["version"]


def test_index_page(tmp_path, monkeypatch):
    """Test the index/dashboard page."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.get("/")
    assert r.status_code == 200
    assert "Hummingbird" in r.text


def test_index_page_with_observations(tmp_path, monkeypatch):
    """Test the index page with some observations."""
    client = _setup_app(tmp_path, monkeypatch)

    with session_scope() as db:
        ind = Individual(name="Test Bird", visit_count=5)
        db.add(ind)
        db.commit()
        for i in range(3):
            obs = Observation(
                individual_id=ind.id,
                species_label="Anna's Hummingbird",
                species_prob=0.8,
                snapshot_path=f"snap{i}.jpg",
                video_path=f"vid{i}.mp4",
            )
            db.add(obs)
        db.commit()

    r = client.get("/")
    assert r.status_code == 200
    assert "Test Bird" in r.text or "Anna" in r.text


def test_observations_page(tmp_path, monkeypatch):
    """Test the observations gallery page."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.get("/observations")
    assert r.status_code == 200
    assert "Observation" in r.text
    assert "obs-thumb" in r.text
    assert "observations-table" in r.text
    assert "data-column-controls" in r.text
    assert 'data-col-key="snapshot"' in r.text
    assert "column_selector.js" in r.text
    assert 'data-sort-default="desc"' in r.text


def test_observations_page_hides_sensitivity_columns_by_default(tmp_path, monkeypatch):
    """Test that sensitivity columns start hidden when unchecked by default."""
    client = _setup_app(tmp_path, monkeypatch)

    with session_scope() as db:
        obs = Observation(
            species_label="Anna's Hummingbird",
            species_prob=0.8,
            snapshot_path="snap.jpg",
            video_path="vid.mp4",
        )
        obs.set_extra({"sensitivity": {"bg_motion_threshold": 30}})
        db.add(obs)
        db.commit()

    r = client.get("/observations")
    assert r.status_code == 200
    assert _has_hidden_column(r.text, "th", "sensitivity.bg_motion_threshold")
    assert _has_hidden_column(r.text, "td", "sensitivity.bg_motion_threshold")


def test_observations_page_with_filter(tmp_path, monkeypatch):
    """Test the observations page with individual filter."""
    client = _setup_app(tmp_path, monkeypatch)

    with session_scope() as db:
        ind = Individual(name="Test Bird")
        db.add(ind)
        db.commit()
        obs = Observation(
            individual_id=ind.id,
            species_label="Anna's Hummingbird",
            species_prob=0.8,
            snapshot_path="snap.jpg",
            video_path="vid.mp4",
        )
        db.add(obs)
        db.commit()
        ind_id = ind.id

    r = client.get(f"/observations?individual_id={ind_id}")
    assert r.status_code == 200


def test_observation_detail_page(tmp_path, monkeypatch):
    """Test the observation detail page."""
    client = _setup_app(tmp_path, monkeypatch)

    with session_scope() as db:
        obs = Observation(
            species_label="Anna's Hummingbird",
            species_prob=0.85,
            snapshot_path="snap.jpg",
            video_path="vid.mp4",
            bbox_x1=10,
            bbox_y1=20,
            bbox_x2=100,
            bbox_y2=200,
        )
        obs.set_extra({"detection": {"confidence": 0.9}})
        db.add(obs)
        db.commit()
        obs_id = obs.id

    r = client.get(f"/observations/{obs_id}")
    assert r.status_code == 200
    assert "Anna" in r.text


def test_observation_detail_not_found(tmp_path, monkeypatch):
    """Test observation detail page for non-existent observation."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.get("/observations/99999")
    assert r.status_code == 404


def test_individuals_page(tmp_path, monkeypatch):
    """Test the individuals list page."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.get("/individuals")
    assert r.status_code == 200
    assert "Individual" in r.text


def test_individuals_page_with_sort(tmp_path, monkeypatch):
    """Test the individuals page with different sort options."""
    client = _setup_app(tmp_path, monkeypatch)

    with session_scope() as db:
        ind = Individual(name="Test Bird", visit_count=10)
        db.add(ind)
        db.commit()

    # Test the default sort by visits and by id
    for sort in ["visits", "id"]:
        r = client.get(f"/individuals?sort={sort}")
        assert r.status_code == 200


def test_individual_detail_page(tmp_path, monkeypatch):
    """Test the individual detail page."""
    client = _setup_app(tmp_path, monkeypatch)

    with session_scope() as db:
        ind = Individual(name="Ruby", visit_count=5)
        db.add(ind)
        db.commit()
        obs = Observation(
            individual_id=ind.id,
            species_label="Anna's Hummingbird",
            species_prob=0.85,
            snapshot_path="snap.jpg",
            video_path="vid.mp4",
        )
        db.add(obs)
        db.commit()
        ind_id = ind.id

    r = client.get(f"/individuals/{ind_id}")
    assert r.status_code == 200
    assert "Ruby" in r.text
    assert "observations-table" in r.text
    assert "data-column-controls" in r.text
    assert 'data-col-key="snapshot"' in r.text
    assert "column_selector.js" in r.text


def test_individual_detail_hides_sensitivity_columns_by_default(tmp_path, monkeypatch):
    """Test that sensitivity columns start hidden on individual pages."""
    client = _setup_app(tmp_path, monkeypatch)

    with session_scope() as db:
        ind = Individual(name="Ruby", visit_count=5)
        db.add(ind)
        db.commit()
        obs = Observation(
            individual_id=ind.id,
            species_label="Anna's Hummingbird",
            species_prob=0.85,
            snapshot_path="snap.jpg",
            video_path="vid.mp4",
        )
        obs.set_extra({"sensitivity": {"bg_motion_threshold": 30}})
        db.add(obs)
        db.commit()
        ind_id = ind.id

    r = client.get(f"/individuals/{ind_id}")
    assert r.status_code == 200
    assert _has_hidden_column(r.text, "th", "sensitivity.bg_motion_threshold")
    assert _has_hidden_column(r.text, "td", "sensitivity.bg_motion_threshold")


def test_individual_detail_not_found(tmp_path, monkeypatch):
    """Test individual detail page for non-existent individual."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.get("/individuals/99999")
    assert r.status_code == 404


def test_rename_individual(tmp_path, monkeypatch):
    """Test renaming an individual."""
    client = _setup_app(tmp_path, monkeypatch)

    with session_scope() as db:
        ind = Individual(name="(unnamed)")
        db.add(ind)
        db.commit()
        ind_id = ind.id

    r = client.post(f"/individuals/{ind_id}/rename", data={"name": "Ruby"}, follow_redirects=False)
    assert r.status_code == 303

    with session_scope() as db:
        ind = db.get(Individual, ind_id)
        assert ind.name == "Ruby"


def test_rename_individual_empty_name(tmp_path, monkeypatch):
    """Test renaming an individual with empty name."""
    client = _setup_app(tmp_path, monkeypatch)

    with session_scope() as db:
        ind = Individual(name="Ruby")
        db.add(ind)
        db.commit()
        ind_id = ind.id

    # Sending whitespace-only name should reset to (unnamed)
    r = client.post(f"/individuals/{ind_id}/rename", data={"name": "   "}, follow_redirects=False)
    assert r.status_code == 303

    with session_scope() as db:
        ind = db.get(Individual, ind_id)
        assert ind.name == "(unnamed)"


def test_rename_individual_not_found(tmp_path, monkeypatch):
    """Test renaming a non-existent individual."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.post("/individuals/99999/rename", data={"name": "Test"})
    assert r.status_code == 404


def test_config_page(tmp_path, monkeypatch):
    """Test the config page."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.get("/config")
    assert r.status_code == 200
    assert "Config" in r.text


def test_config_save(tmp_path, monkeypatch):
    """Test saving config settings."""
    client = _setup_app(tmp_path, monkeypatch)

    r = client.post("/config", data={
        "detect_conf": "0.40",
        "detect_iou": "0.50",
        "min_box_area": "700",
        "cooldown_seconds": "3.0",
        "min_species_prob": "0.40",
        "match_threshold": "0.30",
        "ema_alpha": "0.15",
        "timezone": "local",
        "bg_subtraction_enabled": "1",
        "bg_motion_threshold": "30",
        "bg_motion_blur": "5",
        "bg_min_overlap": "0.15",
    }, follow_redirects=False)
    assert r.status_code == 303


def test_config_save_invalid(tmp_path, monkeypatch):
    """Test saving config with invalid values."""
    client = _setup_app(tmp_path, monkeypatch)

    r = client.post("/config", data={
        "detect_conf": "not_a_number",
        "detect_iou": "0.50",
        "min_box_area": "700",
        "cooldown_seconds": "3.0",
        "min_species_prob": "0.40",
        "match_threshold": "0.30",
        "ema_alpha": "0.15",
        "bg_subtraction_enabled": "1",
        "bg_motion_threshold": "30",
        "bg_motion_blur": "5",
        "bg_min_overlap": "0.15",
    })
    assert r.status_code == 400


def test_calibrate_page(tmp_path, monkeypatch):
    """Test the calibrate/ROI page."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.get("/calibrate")
    assert r.status_code == 200
    assert "Calibrate" in r.text or "ROI" in r.text
    assert 'data-live-src="/api/live_frame.jpg"' in r.text


def test_health_endpoint(tmp_path, monkeypatch):
    """Test the health API endpoint."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["db_ok"] is True
    assert "version" in data


def test_get_roi_endpoint(tmp_path, monkeypatch):
    """Test the get ROI API endpoint."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.get("/api/roi")
    assert r.status_code == 200
    data = r.json()
    assert "x1" in data
    assert "y1" in data
    assert "x2" in data
    assert "y2" in data


def test_set_roi_endpoint(tmp_path, monkeypatch):
    """Test the set ROI API endpoint."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.post("/api/roi", data={
        "x1": "0.1",
        "y1": "0.2",
        "x2": "0.8",
        "y2": "0.9",
    }, follow_redirects=False)
    assert r.status_code == 303


def test_frame_jpg_endpoint_placeholder(tmp_path, monkeypatch):
    """Test the frame.jpg endpoint returns placeholder when no observations."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.get("/api/frame.jpg")
    # Should return either an image or 404 if PIL not available
    assert r.status_code in [200, 404]


def test_live_frame_jpg_requires_rtsp(tmp_path, monkeypatch):
    """Test the live frame endpoint errors when RTSP is not configured."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.get("/api/live_frame.jpg")
    assert r.status_code == 503


def test_live_frame_jpg_success(tmp_path, monkeypatch):
    """Test the live frame endpoint returns a JPEG when the RTSP stream is mocked."""
    client = _setup_app(tmp_path, monkeypatch)

    monkeypatch.setenv("HBMON_RTSP_URL", "rtsp://example/live")
    _install_live_cv2_stub(monkeypatch)

    r = client.get("/api/live_frame.jpg")
    assert r.status_code == 200
    assert "image/jpeg" in r.headers.get("content-type", "")


def test_background_live_requires_rtsp(tmp_path, monkeypatch):
    """Test background live capture errors when RTSP is not configured."""
    client = _setup_app(tmp_path, monkeypatch)

    r = client.post("/api/background/from_live", follow_redirects=False)
    assert r.status_code == 503


def test_background_live_snapshot_success(tmp_path, monkeypatch):
    """Test capturing a live snapshot saves the background image."""
    client = _setup_app(tmp_path, monkeypatch)

    monkeypatch.setenv("HBMON_RTSP_URL", "rtsp://example/live")
    _install_live_cv2_stub(monkeypatch)

    r = client.post("/api/background/from_live", follow_redirects=False)
    assert r.status_code == 303
    assert background_image_path().exists()
    assert background_image_path().read_bytes() == b"jpeg-bytes"


def test_background_endpoints(tmp_path, monkeypatch):
    """Test background endpoints including snapshot selection and clearing."""
    client = _setup_app(tmp_path, monkeypatch)

    r = client.get("/api/background")
    assert r.status_code == 200
    assert r.json()["configured"] is False

    r = client.get("/api/background.jpg")
    assert r.status_code in [200, 404]

    snap = media_dir() / "snap.jpg"
    snap.write_bytes(b"snapshot")

    with session_scope() as db:
        obs = Observation(
            species_label="Hummingbird",
            species_prob=0.9,
            snapshot_path="snap.jpg",
            video_path="",
        )
        db.add(obs)
        db.commit()
        obs_id = obs.id

    r = client.post(f"/api/background/from_observation/{obs_id}", follow_redirects=False)
    assert r.status_code == 303
    assert background_image_path().exists()

    r = client.post("/api/background/clear", follow_redirects=False)
    assert r.status_code == 303
    assert not background_image_path().exists()


def test_background_upload(tmp_path, monkeypatch):
    """Test uploading a background image."""
    client = _setup_app(tmp_path, monkeypatch)
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (120, 130, 140)).save(buf, format="PNG")
    payload = buf.getvalue()

    r = client.post(
        "/api/background/upload",
        files={"file": ("bg.png", payload, "image/png")},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert background_image_path().exists()


def test_video_info_endpoint(tmp_path, monkeypatch):
    """Test video diagnostics for missing and present files."""
    client = _setup_app(tmp_path, monkeypatch)

    with session_scope() as db:
        obs_missing = Observation(
            species_label="Hummingbird",
            species_prob=0.7,
            snapshot_path="",
            video_path="",
        )
        db.add(obs_missing)
        db.commit()
        missing_id = obs_missing.id

        video_path = "clips/sample.mp4"
        obs_with_video = Observation(
            species_label="Hummingbird",
            species_prob=0.7,
            snapshot_path="",
            video_path=video_path,
        )
        db.add(obs_with_video)
        db.commit()
        with_video_id = obs_with_video.id

    video_file = media_dir() / "clips" / "sample.mp4"
    video_file.parent.mkdir(parents=True, exist_ok=True)
    video_file.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 24)

    r = client.get(f"/api/video_info/{missing_id}")
    assert r.status_code == 200
    assert r.json()["file_exists"] is False

    r = client.get(f"/api/video_info/{with_video_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["file_exists"] is True
    assert data["browser_compatible"] is True
    assert "MP4" in data["codec_hint"]


def test_export_observations_csv(tmp_path, monkeypatch):
    """Test exporting observations as CSV."""
    client = _setup_app(tmp_path, monkeypatch)

    with session_scope() as db:
        obs = Observation(
            species_label="Anna's Hummingbird",
            species_prob=0.85,
            snapshot_path="snap.jpg",
            video_path="vid.mp4",
        )
        db.add(obs)
        db.commit()

    r = client.get("/export/observations.csv")
    assert r.status_code == 200
    assert "text/csv" in r.headers.get("content-type", "")
    assert "observation_id" in r.text


def test_export_individuals_csv(tmp_path, monkeypatch):
    """Test exporting individuals as CSV."""
    client = _setup_app(tmp_path, monkeypatch)

    with session_scope() as db:
        ind = Individual(name="Ruby", visit_count=10)
        db.add(ind)
        db.commit()

    r = client.get("/export/individuals.csv")
    assert r.status_code == 200
    assert "text/csv" in r.headers.get("content-type", "")
    assert "individual_id" in r.text


def test_label_observation_not_found(tmp_path, monkeypatch):
    """Test labeling a non-existent observation."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.post("/observations/99999/label", data={"label": "true_positive"})
    assert r.status_code == 404


def test_label_observation_too_long(tmp_path, monkeypatch):
    """Test labeling with a label that is too long."""
    client = _setup_app(tmp_path, monkeypatch)

    with session_scope() as db:
        obs = Observation(
            species_label="Hummingbird",
            species_prob=0.5,
            snapshot_path="snap.jpg",
            video_path="vid.mp4",
        )
        db.add(obs)
        db.commit()
        obs_id = obs.id

    long_label = "x" * 100
    r = client.post(f"/observations/{obs_id}/label", data={"label": long_label})
    assert r.status_code == 400


def test_delete_observation_not_found(tmp_path, monkeypatch):
    """Test deleting a non-existent observation."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.post("/observations/99999/delete")
    assert r.status_code == 404


def test_delete_individual_not_found(tmp_path, monkeypatch):
    """Test deleting a non-existent individual."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.post("/individuals/99999/delete")
    assert r.status_code == 404


def test_refresh_embedding(tmp_path, monkeypatch):
    """Test refreshing an individual's embedding."""
    import numpy as np
    from hbmon.models import _pack_embedding

    client = _setup_app(tmp_path, monkeypatch)

    with session_scope() as db:
        ind = Individual(name="Test")
        db.add(ind)
        db.commit()

        obs = Observation(
            individual_id=ind.id,
            species_label="Anna's Hummingbird",
            species_prob=0.85,
            snapshot_path="snap.jpg",
            video_path="vid.mp4",
        )
        db.add(obs)
        db.commit()

        emb = Embedding(
            observation_id=obs.id,
            individual_id=ind.id,
            embedding_blob=_pack_embedding(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        )
        db.add(emb)
        db.commit()
        ind_id = ind.id

    r = client.post(f"/individuals/{ind_id}/refresh_embedding", follow_redirects=False)
    assert r.status_code == 303


def test_refresh_embedding_no_embeddings(tmp_path, monkeypatch):
    """Test refreshing an individual with no embeddings."""
    client = _setup_app(tmp_path, monkeypatch)

    with session_scope() as db:
        ind = Individual(name="Test")
        db.add(ind)
        db.commit()
        ind_id = ind.id

    r = client.post(f"/individuals/{ind_id}/refresh_embedding", follow_redirects=False)
    assert r.status_code == 303


def test_refresh_embedding_not_found(tmp_path, monkeypatch):
    """Test refreshing a non-existent individual."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.post("/individuals/99999/refresh_embedding")
    assert r.status_code == 404


def test_split_review_page(tmp_path, monkeypatch):
    """Test the split review page."""
    client = _setup_app(tmp_path, monkeypatch)

    with session_scope() as db:
        ind = Individual(name="Test", visit_count=5)
        db.add(ind)
        db.commit()
        ind_id = ind.id

    r = client.get(f"/individuals/{ind_id}/split_review")
    assert r.status_code == 200
    assert "split" in r.text.lower() or "Split" in r.text


def test_split_review_with_embeddings(tmp_path, monkeypatch):
    """Test split review page with observations that have embeddings."""
    import numpy as np
    from hbmon.models import _pack_embedding

    client = _setup_app(tmp_path, monkeypatch)

    with session_scope() as db:
        ind = Individual(name="Test", visit_count=10)
        db.add(ind)
        db.commit()

        # Create 15 observations with embeddings (enough for split suggestion)
        for i in range(15):
            obs = Observation(
                individual_id=ind.id,
                species_label="Anna's Hummingbird",
                species_prob=0.85,
                snapshot_path=f"snap{i}.jpg",
                video_path=f"vid{i}.mp4",
            )
            db.add(obs)
            db.commit()

            # Create embedding with a vector
            vec = np.random.randn(512).astype(np.float32)
            emb = Embedding(
                observation_id=obs.id,
                individual_id=ind.id,
                embedding_blob=_pack_embedding(vec),
            )
            db.add(emb)
            db.commit()

        ind_id = ind.id

    r = client.get(f"/individuals/{ind_id}/split_review")
    assert r.status_code == 200


def test_split_review_not_found(tmp_path, monkeypatch):
    """Test split review for non-existent individual."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.get("/individuals/99999/split_review")
    assert r.status_code == 404


def test_split_apply(tmp_path, monkeypatch):
    """Test applying a split."""
    client = _setup_app(tmp_path, monkeypatch)

    with session_scope() as db:
        ind = Individual(name="Original")
        db.add(ind)
        db.commit()

        obs1 = Observation(
            individual_id=ind.id,
            species_label="Anna's Hummingbird",
            species_prob=0.85,
            snapshot_path="snap1.jpg",
            video_path="vid1.mp4",
        )
        obs2 = Observation(
            individual_id=ind.id,
            species_label="Anna's Hummingbird",
            species_prob=0.85,
            snapshot_path="snap2.jpg",
            video_path="vid2.mp4",
        )
        db.add(obs1)
        db.add(obs2)
        db.commit()
        ind_id = ind.id
        obs2_id = obs2.id

    r = client.post(f"/individuals/{ind_id}/split_apply", data={
        f"assign_{obs2_id}": "B",
    }, follow_redirects=False)
    assert r.status_code == 303


def test_split_apply_no_changes(tmp_path, monkeypatch):
    """Test applying a split with no B assignments."""
    client = _setup_app(tmp_path, monkeypatch)

    with session_scope() as db:
        ind = Individual(name="Original")
        db.add(ind)
        db.commit()
        ind_id = ind.id

    r = client.post(f"/individuals/{ind_id}/split_apply", data={}, follow_redirects=False)
    assert r.status_code == 303


def test_split_apply_not_found(tmp_path, monkeypatch):
    """Test applying split for non-existent individual."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.post("/individuals/99999/split_apply")
    assert r.status_code == 404


def test_export_media_bundle(tmp_path, monkeypatch):
    """Test exporting media bundle as tar.gz."""
    client = _setup_app(tmp_path, monkeypatch)

    # Create some media files
    mdir = media_dir()
    snap_dir = mdir / "snapshots"
    clips_dir = mdir / "clips"
    snap_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    (snap_dir / "test.jpg").write_text("fake image")
    (clips_dir / "test.mp4").write_text("fake video")

    r = client.get("/export/media_bundle.tar.gz")
    assert r.status_code == 200
    assert "application/gzip" in r.headers.get("content-type", "")


def test_export_integration_test_bundle(tmp_path, monkeypatch):
    """Test exporting a single observation integration test bundle."""
    client = _setup_app(tmp_path, monkeypatch)
    settings = load_settings()

    mdir = media_dir()
    snap_dir = mdir / "snapshots"
    clips_dir = mdir / "clips"
    snap_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    snap_path = snap_dir / "obs-test.jpg"
    clip_path = clips_dir / "obs-test.mp4"
    background_path = snap_dir / "obs-test_background.jpg"
    snap_path.write_text("fake image")
    clip_path.write_text("fake video")
    background_path.write_text("fake background")

    extra = {
        "sensitivity": {
            "detect_conf": 0.35,
            "detect_iou": 0.45,
            "min_box_area": 600,
        },
        "snapshots": {
            "background_path": str(background_path.relative_to(mdir)),
        },
        "identification": {
            "species_label_final": "Anna's Hummingbird",
            "species_accepted": True,
        },
    }

    with session_scope() as db:
        obs = Observation(
            species_label="Anna's Hummingbird",
            species_prob=0.93,
            bbox_x1=10,
            bbox_y1=20,
            bbox_x2=110,
            bbox_y2=220,
            snapshot_path=str(snap_path.relative_to(mdir)),
            video_path=str(clip_path.relative_to(mdir)),
            extra_json=json.dumps(extra),
        )
        db.add(obs)
        db.commit()
        obs_id = obs.id

    r = client.post(
        f"/observations/{obs_id}/export_integration_test",
        data={
            "case_name": "flying_99",
            "description": "Test export",
            "behavior": "flying",
            "location": "Testville",
            "human_verified": "true",
        },
    )
    assert r.status_code == 200
    assert "application/gzip" in r.headers.get("content-type", "")

    bundle = tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz")
    names = set(bundle.getnames())
    assert "flying_99/metadata.json" in names
    assert "flying_99/snapshot.jpg" in names
    assert "flying_99/clip.mp4" in names
    assert "flying_99/background.jpg" in names

    metadata_member = bundle.extractfile("flying_99/metadata.json")
    assert metadata_member is not None
    metadata = json.loads(metadata_member.read().decode("utf-8"))
    assert metadata["expected"]["detection"] is True
    assert metadata["expected"]["species_label_final"] == "Anna's Hummingbird"
    assert metadata["expected"]["species_accepted"] is True
    assert metadata["expected"]["behavior"] == "flying"
    assert metadata["expected"]["human_verified"] is True
    sensitivity_tests = metadata["sensitivity_tests"]
    assert len(sensitivity_tests) == 1
    assert sensitivity_tests[0]["params"]["detect_conf"] == 0.35
    assert sensitivity_tests[0]["params"]["detect_iou"] == settings.detect_iou
    assert sensitivity_tests[0]["params"]["min_box_area"] == settings.min_box_area
    assert sensitivity_tests[0]["params"]["bg_motion_threshold"] == settings.bg_motion_threshold
    assert sensitivity_tests[0]["params"]["bg_motion_blur"] == settings.bg_motion_blur
    assert sensitivity_tests[0]["params"]["bg_min_overlap"] == settings.bg_min_overlap
    assert sensitivity_tests[0]["params"]["bg_subtraction_enabled"] == settings.bg_subtraction_enabled
    assert sensitivity_tests[0]["expected_detection"] == metadata["expected"]["detection"]
    extra = metadata["original_observation"]["extra"]
    sensitivity = extra["sensitivity"]
    identification = extra["identification"]
    snapshots = extra["snapshots"]
    assert sensitivity["detect_conf"] == 0.35
    assert sensitivity["detect_iou"] == settings.detect_iou
    assert sensitivity["min_box_area"] == settings.min_box_area
    assert sensitivity["bg_motion_threshold"] == settings.bg_motion_threshold
    assert sensitivity["bg_motion_blur"] == settings.bg_motion_blur
    assert sensitivity["bg_min_overlap"] == settings.bg_min_overlap
    assert sensitivity["bg_subtraction_enabled"] == settings.bg_subtraction_enabled
    assert identification["species_label"] == "Anna's Hummingbird"
    assert identification["species_prob"] == 0.93
    assert identification["match_score"] == 0.0
    assert identification["individual_id"] is None
    assert snapshots["background_path"] == "background.jpg"


def test_dashboard_contains_live_camera_feed_section(tmp_path, monkeypatch):
    """Test that the dashboard includes the Live Camera Feed section with MJPEG stream."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.get("/")
    assert r.status_code == 200
    # Check for live feed elements in the dashboard HTML
    assert "Live Camera Feed" in r.text
    assert "live-feed-img" in r.text
    assert "live-feed-play" in r.text
    assert "live-feed-pause" in r.text
    assert "data-snapshot-src" in r.text
    assert "/api/stream.mjpeg" in r.text

def test_bulk_delete_observations_empty(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    r = client.post("/observations/bulk_delete", data={"obs_ids": []}, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/observations"

def test_bulk_delete_individuals_empty(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    r = client.post("/individuals/bulk_delete", data={"ind_ids": []}, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/individuals"

def test_candidates_filtering(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    from hbmon.models import Candidate
    with session_scope() as db:
        c1 = Candidate(ts=utcnow(), snapshot_path="c1.jpg")
        c1.set_extra({"reason": "motion_low", "review": {"label": "false_negative"}, "species": "Hummingbird"})
        db.add(c1)
        db.commit()

    # Filter by reason
    r = client.get("/candidates?reason=motion_low")
    assert r.status_code == 200
    assert "c1.jpg" in r.text

    # Filter by label
    r = client.get("/candidates?label=false_negative")
    assert r.status_code == 200
    assert "c1.jpg" in r.text

    # Filter by non-matching reason
    r = client.get("/candidates?reason=other")
    assert r.status_code == 200
    assert "c1.jpg" not in r.text

def test_redirect_sanitization(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    # Test bulk delete with malicious redirect
    r = client.post("/observations/bulk_delete", data={"obs_ids": [], "redirect_to": "http://evil.com"}, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/observations"

def test_individual_rename(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    with session_scope() as db:
        ind = Individual(name="Old Name")
        db.add(ind)
        db.commit()
        ind_id = ind.id
    
    r = client.post(f"/individuals/{ind_id}/rename", data={"name": "New Name"}, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == f"/individuals/{ind_id}"
    
    with session_scope() as db:
        ind = db.get(Individual, ind_id)
        assert ind.name == "New Name"

def test_individual_rename_not_found(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    r = client.post("/individuals/999/rename", data={"name": "New Name"})
    assert r.status_code == 404

def test_individual_delete(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    with session_scope() as db:
        ind = Individual(name="To Delete")
        db.add(ind)
        db.commit()
        ind_id = ind.id
    
    r = client.post(f"/individuals/{ind_id}/delete", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/individuals"
    
    with session_scope() as db:
        assert db.get(Individual, ind_id) is None

def test_individual_refresh_embedding_no_embs(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    with session_scope() as db:
        ind = Individual(name="No Embs")
        db.add(ind)
        db.commit()
        ind_id = ind.id
    
    r = client.post(f"/individuals/{ind_id}/refresh_embedding", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == f"/individuals/{ind_id}"

def test_api_roi_get_set(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    # GET default
    r = client.get("/api/roi")
    assert r.status_code == 200
    assert r.json() == {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}
    
    # POST new ROI
    r = client.post("/api/roi", data={"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9}, follow_redirects=False)
    assert r.status_code == 303
    
    # GET updated
    r = client.get("/api/roi")
    assert r.json() == {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9}

def test_api_background_info(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    r = client.get("/api/background")
    assert r.status_code == 200
    data = r.json()
    assert "configured" in data
    assert "exists" in data

def test_api_background_clear(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    r = client.post("/api/background/clear", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/background"

def test_api_health_extended(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "version" in data
    assert "db_ok" in data

def test_api_video_info_not_found(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    r = client.get("/api/video_info/999")
    assert r.status_code == 404

def test_observation_delete_recomputes_stats(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    with session_scope() as db:
        ind = Individual(name="Test Ind")
        db.add(ind)
        db.commit()
        ind_id = ind.id
        
        obs1 = Observation(
            ts=utcnow(),
            species_label="Anna",
            snapshot_path="s1.jpg",
            video_path="v1.mp4",
            individual_id=ind_id
        )
        obs2 = Observation(
            ts=utcnow(),
            species_label="Anna",
            snapshot_path="s2.jpg",
            video_path="v2.mp4",
            individual_id=ind_id
        )
        db.add_all([obs1, obs2])
        db.commit()
        obs1_id = obs1.id
        
        # Verify initial stats
        db.refresh(ind)
        # _recompute_individual_stats might not have run yet if not triggered, 
        # but worker usually does it. Here we manually trigger via delete.
    
    # Delete one observation
    r = client.post("/observations/bulk_delete", data={"obs_ids": [obs1_id]}, follow_redirects=False)
    assert r.status_code == 303
    
    with session_scope() as db:
        ind = db.get(Individual, ind_id)
        # Total was 2, now should be 1
        assert ind.visit_count == 1

def test_individual_refresh_embedding_with_embs(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    with session_scope() as db:
        ind = Individual(name="With Embs")
        db.add(ind)
        db.commit()
        ind_id = ind.id
        
        obs = Observation(ts=utcnow(), species_label="Anna", snapshot_path="s.jpg", video_path="v.mp4", individual_id=ind_id)
        db.add(obs)
        db.commit()
        
        emb = Embedding(observation_id=obs.id, individual_id=ind_id)
        emb.set_vec(np.zeros(512))
        db.add(emb)
        db.commit()
        
    r = client.post(f"/individuals/{ind_id}/refresh_embedding", follow_redirects=False)
    assert r.status_code == 303
    
    with session_scope() as db:
        ind = db.get(Individual, ind_id)
        assert ind.prototype_blob is not None

def test_config_save_complete(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    data = {
        "detect_conf": "0.55",
        "detect_iou": "0.45",
        "min_box_area": "500",
        "cooldown_seconds": "10.0",
        "min_species_prob": "0.8",
        "match_threshold": "0.7",
        "ema_alpha": "0.5",
        "timezone": "UTC",
        "bg_subtraction_enabled": "on",
        "bg_motion_threshold": "30",
        "bg_motion_blur": "5",
        "bg_min_overlap": "0.15"
    }
    r = client.post("/config", data=data, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/config?saved=1"

def test_config_save_invalid_tz(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    data = {
        "detect_conf": "0.55",
        "detect_iou": "0.45",
        "min_box_area": "500",
        "timezone": "Invalid/Timezone"
    }
    r = client.post("/config", data=data)
    assert r.status_code == 400
    assert "Timezone must be a valid IANA name" in r.text

def test_api_roi_post(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    r = client.post("/api/roi", data={"x1": "0.2", "y1": "0.2", "x2": "0.8", "y2": "0.8"}, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/calibrate"

def test_api_background_upload_invalid(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    # Empty upload
    r = client.post("/api/background/upload", data={})
    assert r.status_code == 400


def test_export_observation_includes_roi(tmp_path, monkeypatch):
    """Test that export includes ROI snapshot if available."""
    client = _setup_app(tmp_path, monkeypatch)
    mdir = media_dir()
    
    # Create media files
    snap = mdir / "snap.jpg"
    roi = mdir / "snap_roi.jpg"
    snap.write_text("orig")
    roi.write_text("roi")
    
    with session_scope() as db:
        obs = Observation(
            species_label="Anna's Hummingbird",
            species_prob=0.8,
            snapshot_path="snap.jpg",
            video_path="vid.mp4",
        )
        obs.set_extra({
            "snapshots": {
                "roi_path": "snap_roi.jpg"
            }
        })
        db.add(obs)
        db.commit()
        obs_id = obs.id

    r = client.post(f"/observations/{obs_id}/export_integration_test", follow_redirects=False)
    assert r.status_code == 200
    
    # Inspect tarball
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tf:
        names = tf.getnames()
        # Should contain snapshot_roi.jpg
        assert any(n.endswith("/snapshot_roi.jpg") for n in names)
        
        # Verify content matches
        roi_member = next(n for n in names if n.endswith("/snapshot_roi.jpg"))
        f = tf.extractfile(roi_member)
        assert f.read() == b"roi"
