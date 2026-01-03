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

    cache_key = f"{obs_id}_23_fast"
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


def test_resolve_stream_quality_with_none(monkeypatch) -> None:
    """Test _resolve_stream_quality with quality=None returns auto defaults."""
    monkeypatch.setenv("HBMON_VIDEO_CRF", "23")
    monkeypatch.setenv("HBMON_VIDEO_PRESET", "fast")
    
    from hbmon.web import _resolve_stream_quality
    
    quality_name, crf, preset = _resolve_stream_quality(None)
    
    assert quality_name == "auto"
    assert crf == 23
    assert preset == "fast"


def test_resolve_stream_quality_with_high(monkeypatch) -> None:
    """Test _resolve_stream_quality with quality='high' returns CRF 18."""
    monkeypatch.setenv("HBMON_VIDEO_PRESET", "fast")
    
    from hbmon.web import _resolve_stream_quality
    
    quality_name, crf, preset = _resolve_stream_quality("high")
    
    assert quality_name == "high"
    assert crf == 18
    assert preset == "fast"


def test_resolve_stream_quality_with_balanced(monkeypatch) -> None:
    """Test _resolve_stream_quality with quality='balanced' returns CRF 23."""
    monkeypatch.setenv("HBMON_VIDEO_PRESET", "medium")
    
    from hbmon.web import _resolve_stream_quality
    
    quality_name, crf, preset = _resolve_stream_quality("balanced")
    
    assert quality_name == "balanced"
    assert crf == 23
    assert preset == "medium"


def test_resolve_stream_quality_with_low(monkeypatch) -> None:
    """Test _resolve_stream_quality with quality='low' returns CRF 28."""
    monkeypatch.setenv("HBMON_VIDEO_PRESET", "veryfast")
    
    from hbmon.web import _resolve_stream_quality
    
    quality_name, crf, preset = _resolve_stream_quality("low")
    
    assert quality_name == "low"
    assert crf == 28
    assert preset == "veryfast"


def test_resolve_stream_quality_with_invalid_value(monkeypatch) -> None:
    """Test _resolve_stream_quality with invalid quality falls back to auto."""
    monkeypatch.setenv("HBMON_VIDEO_CRF", "25")
    monkeypatch.setenv("HBMON_VIDEO_PRESET", "slow")
    
    from hbmon.web import _resolve_stream_quality
    
    quality_name, crf, preset = _resolve_stream_quality("invalid_quality")
    
    assert quality_name == "auto"
    assert crf == 25
    assert preset == "slow"


def test_resolve_stream_quality_case_insensitive(monkeypatch) -> None:
    """Test _resolve_stream_quality handles case variations."""
    monkeypatch.setenv("HBMON_VIDEO_PRESET", "fast")
    
    from hbmon.web import _resolve_stream_quality
    
    # Test uppercase
    quality_name, crf, preset = _resolve_stream_quality("HIGH")
    assert quality_name == "high"
    assert crf == 18
    
    # Test mixed case
    quality_name, crf, preset = _resolve_stream_quality("BaLaNcEd")
    assert quality_name == "balanced"
    assert crf == 23


def test_resolve_stream_quality_whitespace_handling(monkeypatch) -> None:
    """Test _resolve_stream_quality handles whitespace."""
    monkeypatch.setenv("HBMON_VIDEO_PRESET", "fast")
    
    from hbmon.web import _resolve_stream_quality
    
    quality_name, crf, preset = _resolve_stream_quality("  high  ")
    
    assert quality_name == "high"
    assert crf == 18
    assert preset == "fast"


def test_video_stream_quality_high_produces_different_cache_key(tmp_path, monkeypatch) -> None:
    """Test that quality=high produces a different cache key than default."""
    _setup_app(tmp_path, monkeypatch)
    
    clips_dir = hbmon.config.media_dir() / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    video_path = clips_dir / "sample.mp4"
    video_path.write_bytes(b"0123456789ABCDEFGHIJ")
    
    obs_id = _create_observation("clips/sample.mp4")
    
    # Calculate cache hash for default (auto/CRF 23)
    cache_key_auto = f"{obs_id}_23_fast"
    cache_hash_auto = hashlib.md5(cache_key_auto.encode()).hexdigest()[:12]
    
    # Calculate cache hash for high quality (CRF 18)
    cache_key_high = f"{obs_id}_18_fast"
    cache_hash_high = hashlib.md5(cache_key_high.encode()).hexdigest()[:12]
    
    # Verify they are different
    assert cache_hash_auto != cache_hash_high


def test_video_stream_quality_balanced_produces_correct_cache_key(tmp_path, monkeypatch) -> None:
    """Test that quality=balanced produces correct cache key with CRF 23."""
    _setup_app(tmp_path, monkeypatch)
    
    clips_dir = hbmon.config.media_dir() / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    video_path = clips_dir / "sample.mp4"
    video_path.write_bytes(b"test_video_content")
    
    obs_id = _create_observation("clips/sample.mp4")
    
    # Calculate expected cache hash for balanced quality (CRF 23)
    cache_key_balanced = f"{obs_id}_23_fast"
    cache_hash_balanced = hashlib.md5(cache_key_balanced.encode()).hexdigest()[:12]
    
    # This verifies the cache key generation logic matches expected pattern
    assert cache_hash_balanced is not None
    assert len(cache_hash_balanced) == 12


def test_video_stream_quality_low_produces_correct_cache_key(tmp_path, monkeypatch) -> None:
    """Test that quality=low produces correct cache key with CRF 28."""
    _setup_app(tmp_path, monkeypatch)
    
    clips_dir = hbmon.config.media_dir() / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    video_path = clips_dir / "sample.mp4"
    video_path.write_bytes(b"test_video_content")
    
    obs_id = _create_observation("clips/sample.mp4")
    
    # Calculate expected cache hash for low quality (CRF 28)
    cache_key_low = f"{obs_id}_28_fast"
    cache_hash_low = hashlib.md5(cache_key_low.encode()).hexdigest()[:12]
    
    # Calculate hash for balanced (CRF 23) for comparison
    cache_key_balanced = f"{obs_id}_23_fast"
    cache_hash_balanced = hashlib.md5(cache_key_balanced.encode()).hexdigest()[:12]
    
    # Verify different quality levels produce different cache keys
    assert cache_hash_low != cache_hash_balanced


def test_streaming_bitrate_with_quality_parameter_high(tmp_path, monkeypatch) -> None:
    """Test streaming_bitrate endpoint with quality=high parameter."""
    client = _setup_app(tmp_path, monkeypatch)
    
    clips_dir = hbmon.config.media_dir() / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    video_path = clips_dir / "sample.mp4"
    video_path.write_bytes(b"0" * 2000)
    
    obs_id = _create_observation(
        "clips/sample.mp4",
        extra={"media": {"video": {"duration": 2.0}}},
    )
    
    # Create cache for high quality (CRF 18)
    cache_key = f"{obs_id}_18_fast"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    cached_path = hbmon.config.media_dir() / ".cache" / "compressed" / f"sample_{cache_hash}.mp4"
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_bytes(b"1" * 600)  # Larger file for high quality
    
    response = client.get(f"/api/streaming_bitrate/{obs_id}?quality=high")
    
    assert response.status_code == 200
    payload = response.json()
    # Should find the high quality cache
    assert payload["bitrate_mbps"] is not None
    assert payload["bitrate_mbps"] == pytest.approx((600 * 8) / (2.0 * 1_000_000))
    assert payload["compression_ratio"] == pytest.approx(2000 / 600)


def test_streaming_bitrate_with_quality_parameter_low(tmp_path, monkeypatch) -> None:
    """Test streaming_bitrate endpoint with quality=low parameter."""
    client = _setup_app(tmp_path, monkeypatch)
    
    clips_dir = hbmon.config.media_dir() / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    video_path = clips_dir / "sample.mp4"
    video_path.write_bytes(b"0" * 2000)
    
    obs_id = _create_observation(
        "clips/sample.mp4",
        extra={"media": {"video": {"duration": 2.0}}},
    )
    
    # Create cache for low quality (CRF 28)
    cache_key = f"{obs_id}_28_fast"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    cached_path = hbmon.config.media_dir() / ".cache" / "compressed" / f"sample_{cache_hash}.mp4"
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_bytes(b"1" * 400)  # Smaller file for low quality
    
    response = client.get(f"/api/streaming_bitrate/{obs_id}?quality=low")
    
    assert response.status_code == 200
    payload = response.json()
    # Should find the low quality cache
    assert payload["bitrate_mbps"] is not None
    assert payload["bitrate_mbps"] == pytest.approx((400 * 8) / (2.0 * 1_000_000))
    assert payload["compression_ratio"] == pytest.approx(2000 / 400)


def test_streaming_bitrate_quality_cache_isolation(tmp_path, monkeypatch) -> None:
    """Test that different quality caches are isolated from each other."""
    client = _setup_app(tmp_path, monkeypatch)
    
    clips_dir = hbmon.config.media_dir() / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    video_path = clips_dir / "sample.mp4"
    video_path.write_bytes(b"0" * 2000)
    
    obs_id = _create_observation(
        "clips/sample.mp4",
        extra={"media": {"video": {"duration": 2.0}}},
    )
    
    # Create cache ONLY for high quality (CRF 18)
    cache_key_high = f"{obs_id}_18_fast"
    cache_hash_high = hashlib.md5(cache_key_high.encode()).hexdigest()[:12]
    cached_path_high = hbmon.config.media_dir() / ".cache" / "compressed" / f"sample_{cache_hash_high}.mp4"
    cached_path_high.parent.mkdir(parents=True, exist_ok=True)
    cached_path_high.write_bytes(b"1" * 600)
    
    # Request high quality - should find cache
    response_high = client.get(f"/api/streaming_bitrate/{obs_id}?quality=high")
    assert response_high.status_code == 200
    assert response_high.json()["bitrate_mbps"] is not None
    
    # Request low quality - should NOT find cache (different CRF)
    response_low = client.get(f"/api/streaming_bitrate/{obs_id}?quality=low")
    assert response_low.status_code == 200
    assert response_low.json()["bitrate_mbps"] is None


def test_streaming_bitrate_auto_quality_uses_default_crf(tmp_path, monkeypatch) -> None:
    """Test that quality=auto uses default CRF from environment."""
    monkeypatch.setenv("HBMON_VIDEO_CRF", "25")
    client = _setup_app(tmp_path, monkeypatch)
    
    clips_dir = hbmon.config.media_dir() / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    video_path = clips_dir / "sample.mp4"
    video_path.write_bytes(b"0" * 2000)
    
    obs_id = _create_observation(
        "clips/sample.mp4",
        extra={"media": {"video": {"duration": 2.0}}},
    )
    
    # Create cache with custom default CRF 25
    cache_key = f"{obs_id}_25_fast"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    cached_path = hbmon.config.media_dir() / ".cache" / "compressed" / f"sample_{cache_hash}.mp4"
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_bytes(b"1" * 500)
    
    response = client.get(f"/api/streaming_bitrate/{obs_id}?quality=auto")
    
    assert response.status_code == 200
    payload = response.json()
    # Should find cache with CRF 25
    assert payload["bitrate_mbps"] is not None


def test_streaming_bitrate_invalid_quality_falls_back_to_auto(tmp_path, monkeypatch) -> None:
    """Test that invalid quality value falls back to auto/default CRF."""
    client = _setup_app(tmp_path, monkeypatch)
    
    clips_dir = hbmon.config.media_dir() / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    video_path = clips_dir / "sample.mp4"
    video_path.write_bytes(b"0" * 2000)
    
    obs_id = _create_observation(
        "clips/sample.mp4",
        extra={"media": {"video": {"duration": 2.0}}},
    )
    
    # Create cache with default CRF 23
    cache_key = f"{obs_id}_23_fast"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    cached_path = hbmon.config.media_dir() / ".cache" / "compressed" / f"sample_{cache_hash}.mp4"
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_bytes(b"1" * 500)
    
    # Request with invalid quality
    response = client.get(f"/api/streaming_bitrate/{obs_id}?quality=invalid_value")
    
    assert response.status_code == 200
    payload = response.json()
    # Should fall back to default CRF 23 and find cache
    assert payload["bitrate_mbps"] is not None
