from __future__ import annotations

from pathlib import Path
import hashlib
from typing import Any

from fastapi.testclient import TestClient
import pytest

import hbmon.config
import hbmon.web
from hbmon.db import init_db, reset_db_state, session_scope
from hbmon.models import Observation, utcnow


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
    monkeypatch.setattr(hbmon.config, "data_dir", lambda: data_dir)
    monkeypatch.setattr(hbmon.config, "media_dir", lambda: media)
    monkeypatch.setattr(hbmon.web, "data_dir", lambda: data_dir)
    monkeypatch.setattr(hbmon.web, "media_dir", lambda: media)
    reset_db_state()
    hbmon.config.ensure_dirs()
    init_db()
    app = hbmon.web.make_app()
    client = TestClient(app)
    _TEST_CLIENTS.append(client)
    return client


def _create_observation(video_path: str, extra: dict[str, Any] | None = None) -> int:
    with session_scope() as db:
        obs = Observation(
            ts=utcnow(),
            species_label="Hummingbird",
            species_prob=0.9,
            snapshot_path="snap.jpg",
            video_path=video_path,
        )
        if extra:
            obs.set_extra(extra)
        db.add(obs)
        db.commit()
        return obs.id


def test_video_stream_range_uncompressed(tmp_path, monkeypatch) -> None:
    """Test HTTP Range support with a partial byte range when serving an uncompressed video file."""
    monkeypatch.setenv("HBMON_VIDEO_STREAM_COMPRESSION", "0")
    client = _setup_app(tmp_path, monkeypatch)

    clips_dir = hbmon.config.media_dir() / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    video_path = clips_dir / "sample.mp4"
    video_content = b"0123456789ABCDEFGHIJ"
    video_path.write_bytes(video_content)

    obs_id = _create_observation("clips/sample.mp4")

    response = client.get(f"/api/video/{obs_id}", headers={"Range": "bytes=0-9"})

    assert response.status_code == 206
    assert response.headers["Content-Range"] == f"bytes 0-9/{len(video_content)}"
    assert len(response.content) == 10
    assert response.content == video_content[:10]


def test_video_stream_missing_file_returns_404(tmp_path, monkeypatch) -> None:
    """Verify that requesting a video whose file is missing on disk returns a 404 response."""
    monkeypatch.setenv("HBMON_VIDEO_STREAM_COMPRESSION", "0")
    client = _setup_app(tmp_path, monkeypatch)

    obs_id = _create_observation("clips/missing.mp4")

    response = client.get(f"/api/video/{obs_id}", headers={"Range": "bytes=0-9"})

    assert response.status_code == 404


def test_video_stream_nonexistent_observation_returns_404(tmp_path, monkeypatch) -> None:
    """Verify that requesting a video for a non-existent observation ID returns a 404 response."""
    monkeypatch.setenv("HBMON_VIDEO_STREAM_COMPRESSION", "0")
    client = _setup_app(tmp_path, monkeypatch)

    response = client.get("/api/video/99999", headers={"Range": "bytes=0-9"})

    assert response.status_code == 404


def test_video_stream_empty_video_path_returns_404(tmp_path, monkeypatch) -> None:
    """Verify that requesting a video for an observation with empty video_path returns a 404 response."""
    monkeypatch.setenv("HBMON_VIDEO_STREAM_COMPRESSION", "0")
    client = _setup_app(tmp_path, monkeypatch)

    # Create observation with empty video_path
    with session_scope() as db:
        obs = Observation(
            ts=utcnow(),
            species_label="Hummingbird",
            species_prob=0.9,
            snapshot_path="snap.jpg",
            video_path="",
        )
        db.add(obs)
        db.commit()
        obs_id = obs.id

    response = client.get(f"/api/video/{obs_id}", headers={"Range": "bytes=0-9"})

    assert response.status_code == 404


def test_video_stream_full_file_without_range(tmp_path, monkeypatch) -> None:
    """Test requesting a full video file without a Range header returns 200 with complete content."""
    monkeypatch.setenv("HBMON_VIDEO_STREAM_COMPRESSION", "0")
    client = _setup_app(tmp_path, monkeypatch)

    clips_dir = hbmon.config.media_dir() / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    video_path = clips_dir / "sample.mp4"
    video_content = b"0123456789ABCDEFGHIJ"
    video_path.write_bytes(video_content)

    obs_id = _create_observation("clips/sample.mp4")

    response = client.get(f"/api/video/{obs_id}")

    assert response.status_code == 200
    assert len(response.content) == len(video_content)
    assert response.content == video_content


def test_video_stream_range_from_offset(tmp_path, monkeypatch) -> None:
    """Test HTTP Range request with 'bytes=10-' format (from offset to end)."""
    monkeypatch.setenv("HBMON_VIDEO_STREAM_COMPRESSION", "0")
    client = _setup_app(tmp_path, monkeypatch)

    clips_dir = hbmon.config.media_dir() / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    video_path = clips_dir / "sample.mp4"
    video_content = b"0123456789ABCDEFGHIJ"
    video_path.write_bytes(video_content)

    obs_id = _create_observation("clips/sample.mp4")

    response = client.get(f"/api/video/{obs_id}", headers={"Range": "bytes=10-"})

    assert response.status_code == 206
    assert response.headers["Content-Range"] == f"bytes 10-{len(video_content)-1}/{len(video_content)}"
    assert len(response.content) == 10
    assert response.content == video_content[10:]


def test_video_stream_range_suffix(tmp_path, monkeypatch) -> None:
    """Test HTTP Range request with 'bytes=-10' format (last 10 bytes)."""
    monkeypatch.setenv("HBMON_VIDEO_STREAM_COMPRESSION", "0")
    client = _setup_app(tmp_path, monkeypatch)

    clips_dir = hbmon.config.media_dir() / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    video_path = clips_dir / "sample.mp4"
    video_content = b"0123456789ABCDEFGHIJ"
    video_path.write_bytes(video_content)

    obs_id = _create_observation("clips/sample.mp4")

    response = client.get(f"/api/video/{obs_id}", headers={"Range": "bytes=-10"})

    assert response.status_code == 206
    assert response.headers["Content-Range"] == f"bytes 10-{len(video_content)-1}/{len(video_content)}"
    assert len(response.content) == 10
    assert response.content == video_content[-10:]


def test_video_stream_range_middle_section(tmp_path, monkeypatch) -> None:
    """Test HTTP Range request with 'bytes=5-14' format (specific middle section)."""
    monkeypatch.setenv("HBMON_VIDEO_STREAM_COMPRESSION", "0")
    client = _setup_app(tmp_path, monkeypatch)

    clips_dir = hbmon.config.media_dir() / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    video_path = clips_dir / "sample.mp4"
    video_content = b"0123456789ABCDEFGHIJ"
    video_path.write_bytes(video_content)

    obs_id = _create_observation("clips/sample.mp4")

    response = client.get(f"/api/video/{obs_id}", headers={"Range": "bytes=5-14"})

    assert response.status_code == 206
    assert response.headers["Content-Range"] == f"bytes 5-14/{len(video_content)}"
    assert len(response.content) == 10
    assert response.content == video_content[5:15]


def test_streaming_bitrate_returns_null_when_cache_missing(tmp_path, monkeypatch) -> None:
    client = _setup_app(tmp_path, monkeypatch)

    clips_dir = hbmon.config.media_dir() / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    video_path = clips_dir / "sample.mp4"
    video_path.write_bytes(b"1234567890")

    obs_id = _create_observation(
        "clips/sample.mp4",
        extra={"media": {"video": {"duration": 2.0}}},
    )

    response = client.get(f"/api/streaming_bitrate/{obs_id}")

    assert response.status_code == 200
    assert response.json() == {"bitrate_mbps": None}


def test_streaming_bitrate_returns_metrics_when_cached(tmp_path, monkeypatch) -> None:
    client = _setup_app(tmp_path, monkeypatch)

    clips_dir = hbmon.config.media_dir() / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    video_path = clips_dir / "sample.mp4"
    video_path.write_bytes(b"0" * 2000)

    obs_id = _create_observation(
        "clips/sample.mp4",
        extra={"media": {"video": {"duration": 2.0}}},
    )

    # Cache key now includes quality parameter (high=23)
    cache_key = f"{obs_id}_high_23_fast"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    cached_path = hbmon.config.media_dir() / ".cache" / "compressed" / f"{video_path.stem}_{cache_hash}.mp4"
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_bytes(b"1" * 500)

    response = client.get(f"/api/streaming_bitrate/{obs_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["bitrate_mbps"] == pytest.approx((500 * 8) / (2.0 * 1_000_000))
    assert payload["compression_ratio"] == pytest.approx(2000 / 500)
    assert payload["cached_size_kb"] > 0
    assert payload["source_size_kb"] > 0
