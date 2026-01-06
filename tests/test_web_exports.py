from __future__ import annotations

from fastapi.testclient import TestClient
import io
from pathlib import Path
import tarfile

import pytest

import hbmon.config
import hbmon.web
from hbmon.db import init_db, reset_db_state, get_async_session_factory
from hbmon.models import Individual, Observation

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
    data_dir.mkdir(parents=True, exist_ok=True)
    media.mkdir(parents=True, exist_ok=True)
    (media / "snapshots").mkdir(parents=True, exist_ok=True)
    (media / "clips").mkdir(parents=True, exist_ok=True)
    
    init_db()
    
    app = hbmon.web.make_app()
    return TestClient(app)

@pytest.mark.asyncio
async def test_export_observations_csv(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    async with get_async_session_factory()() as db:
        db.add(Observation(species_label="Bird A", species_prob=0.8, snapshot_path="snap.jpg", video_path="vid.mp4"))
        await db.commit()
        
    response = client.get("/export/observations.csv")
    assert response.status_code == 200
    assert "Bird A" in response.text
    assert "observation_id" in response.text

@pytest.mark.asyncio
async def test_export_individuals_csv(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    async with get_async_session_factory()() as db:
        db.add(Individual(name="Indy 1", visit_count=5))
        await db.commit()
        
    response = client.get("/export/individuals.csv")
    assert response.status_code == 200
    assert "Indy 1" in response.text
    assert "individual_id" in response.text

def test_get_roi_snapshot_path_presence_and_absence() -> None:
    obs = Observation(species_label="Bird", species_prob=0.7, snapshot_path="snap.jpg", video_path="clip.mp4")

    assert hbmon.web.get_roi_snapshot_path(obs) is None

    obs.set_extra({"snapshots": {"roi_path": "snapshots/roi.jpg"}})
    assert hbmon.web.get_roi_snapshot_path(obs) == "snapshots/roi.jpg"

@pytest.mark.asyncio
async def test_export_media_bundle(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    
    # Create dummy media files
    sdir = hbmon.config.snapshots_dir()
    cdir = hbmon.config.clips_dir()
    sdir.mkdir(parents=True, exist_ok=True)
    cdir.mkdir(parents=True, exist_ok=True)
    (sdir / "test.jpg").write_text("fake image")
    (cdir / "test.mp4").write_text("fake clip")
    
    response = client.get("/export/media_bundle.tar.gz")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/gzip"
    assert len(response.content) > 0


@pytest.mark.asyncio
async def test_export_observation_integration_bundle_includes_media(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    media_root = hbmon.config.media_dir()

    snapshot_path = media_root / "snapshots" / "snapshot.jpg"
    video_path = media_root / "clips" / "clip.mp4"
    annotated_path = media_root / "snapshots" / "annotated.jpg"
    clip_snapshot_path = media_root / "snapshots" / "clip_snapshot.jpg"
    background_path = media_root / "snapshots" / "background.jpg"
    roi_path = media_root / "snapshots" / "roi.jpg"
    mask_path = media_root / "snapshots" / "mask.png"
    mask_overlay_path = media_root / "snapshots" / "mask_overlay.png"

    for path in (
        snapshot_path,
        video_path,
        annotated_path,
        clip_snapshot_path,
        background_path,
        roi_path,
        mask_path,
        mask_overlay_path,
    ):
        path.write_text("data")

    async with get_async_session_factory()() as db:
        obs = Observation(
            species_label="Bird A",
            species_prob=0.8,
            snapshot_path="snapshots/snapshot.jpg",
            video_path="clips/clip.mp4",
        )
        obs.set_extra(
            {
                "snapshots": {
                    "annotated_path": "snapshots/annotated.jpg",
                    "clip_path": "snapshots/clip_snapshot.jpg",
                    "background_path": "snapshots/background.jpg",
                    "roi_path": "snapshots/roi.jpg",
                },
                "media": {
                    "mask_path": "snapshots/mask.png",
                    "mask_overlay_path": "snapshots/mask_overlay.png",
                },
            }
        )
        db.add(obs)
        await db.commit()
        await db.refresh(obs)
        obs_id = obs.id

    response = client.post(f"/observations/{obs_id}/export_integration_test", data={"case_name": ""})

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/gzip"

    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tf:
        names = set(tf.getnames())

    safe_case = f"observation_{obs_id}"
    expected = {
        f"{safe_case}/metadata.json",
        f"{safe_case}/snapshot.jpg",
        f"{safe_case}/clip.mp4",
        f"{safe_case}/snapshot_annotated.jpg",
        f"{safe_case}/snapshot_clip.jpg",
        f"{safe_case}/background.jpg",
        f"{safe_case}/roi.jpg",
        f"{safe_case}/mask.png",
        f"{safe_case}/mask_overlay.png",
    }
    assert expected.issubset(names)
