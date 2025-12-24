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
    })
    assert r.status_code == 400


def test_calibrate_page(tmp_path, monkeypatch):
    """Test the calibrate/ROI page."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.get("/calibrate")
    assert r.status_code == 200
    assert "Calibrate" in r.text or "ROI" in r.text


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


def test_dashboard_contains_live_camera_feed_section(tmp_path, monkeypatch):
    """Test that the dashboard includes the Live Camera Feed section."""
    client = _setup_app(tmp_path, monkeypatch)
    r = client.get("/")
    assert r.status_code == 200
    # Check for live feed elements in the dashboard HTML
    assert "Live Camera Feed" in r.text
    assert "live-feed-img" in r.text
    assert "live-feed-refresh" in r.text
    assert "live-feed-toggle" in r.text
    assert "/api/frame.jpg" in r.text
