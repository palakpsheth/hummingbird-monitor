from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from hbmon.config import ensure_dirs, media_dir
from hbmon.db import init_db, session_scope
from hbmon.models import Embedding, Individual, Observation, utcnow
from hbmon.web import make_app


def _setup_app(tmp_path: Path, monkeypatch) -> TestClient:
    data_dir = tmp_path / "data"
    media = tmp_path / "media"
    monkeypatch.setenv("HBMON_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(media))
    monkeypatch.setenv("HBMON_DB_URL", f"sqlite:///{tmp_path/'db.sqlite'}")
    ensure_dirs()
    init_db()
    app = make_app()
    return TestClient(app)


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


def test_dashboard_contains_live_camera_feed_section(tmp_path, monkeypatch):
    """Test that the dashboard includes the Live Camera Feed section with MJPEG stream."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.get("/")
    assert r.status_code == 200
    # Check for live feed elements in the dashboard HTML
    assert "Live Camera Feed" in r.text
    assert "live-feed-img" in r.text
    assert "live-feed-toggle" in r.text
    assert "/api/stream.mjpeg" in r.text
