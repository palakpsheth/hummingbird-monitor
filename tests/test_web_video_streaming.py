from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from hbmon.config import ensure_dirs, media_dir
from hbmon.db import init_db, reset_db_state, session_scope
from hbmon.models import Observation, utcnow
from hbmon.web import make_app


_TEST_CLIENTS: list[TestClient] = []


@pytest.fixture(autouse=True)
def _close_test_clients() -> None:
    yield
    while _TEST_CLIENTS:
        client = _TEST_CLIENTS.pop()
        client.close()


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


def _create_observation(video_path: str) -> int:
    with session_scope() as db:
        obs = Observation(
            ts=utcnow(),
            species_label="Hummingbird",
            species_prob=0.9,
            snapshot_path="snap.jpg",
            video_path=video_path,
        )
        db.add(obs)
        db.commit()
        return obs.id


def test_video_stream_range_uncompressed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HBMON_VIDEO_STREAM_COMPRESSION", "0")
    client = _setup_app(tmp_path, monkeypatch)

    clips_dir = media_dir() / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    video_path = clips_dir / "sample.mp4"
    payload = b"0123456789ABCDEFGHIJ"
    video_path.write_bytes(payload)

    obs_id = _create_observation("clips/sample.mp4")

    response = client.get(f"/api/video/{obs_id}", headers={"Range": "bytes=0-9"})

    assert response.status_code == 206
    assert response.headers["Content-Range"] == f"bytes 0-9/{len(payload)}"
    assert len(response.content) == 10
    assert response.content == payload[:10]


def test_video_stream_missing_file_returns_404(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HBMON_VIDEO_STREAM_COMPRESSION", "0")
    client = _setup_app(tmp_path, monkeypatch)

    obs_id = _create_observation("clips/missing.mp4")

    response = client.get(f"/api/video/{obs_id}", headers={"Range": "bytes=0-9"})

    assert response.status_code == 404
