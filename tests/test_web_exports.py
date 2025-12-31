from __future__ import annotations

from pathlib import Path
import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from hbmon.config import load_settings, save_settings, media_dir, snapshots_dir, clips_dir
from hbmon.db import init_db, reset_db_state, get_async_session_factory
from hbmon.models import Individual, Observation
from hbmon.web import make_app

def _setup_app(tmp_path: Path, monkeypatch) -> TestClient:
    data_dir = tmp_path / "data"
    media = tmp_path / "media"
    db_path = tmp_path / "db.sqlite"
    
    monkeypatch.setenv("HBMON_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(media))
    monkeypatch.setenv("HBMON_DB_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("HBMON_DB_ASYNC_URL", f"sqlite+aiosqlite:///{db_path}")
    
    reset_db_state()
    data_dir.mkdir(parents=True, exist_ok=True)
    media.mkdir(parents=True, exist_ok=True)
    (media / "snapshots").mkdir(parents=True, exist_ok=True)
    (media / "clips").mkdir(parents=True, exist_ok=True)
    
    init_db()
    
    app = make_app()
    return TestClient(app)

@pytest.mark.anyio
async def test_export_observations_csv(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    async with get_async_session_factory()() as db:
        db.add(Observation(species_label="Bird A", species_prob=0.8, snapshot_path="snap.jpg", video_path="vid.mp4"))
        await db.commit()
        
    response = client.get("/export/observations.csv")
    assert response.status_code == 200
    assert "Bird A" in response.text
    assert "observation_id" in response.text

@pytest.mark.anyio
async def test_export_individuals_csv(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    async with get_async_session_factory()() as db:
        db.add(Individual(name="Indy 1", visit_count=5))
        await db.commit()
        
    response = client.get("/export/individuals.csv")
    assert response.status_code == 200
    assert "Indy 1" in response.text
    assert "individual_id" in response.text

@pytest.mark.anyio
async def test_export_media_bundle(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    
    # Create dummy media files
    sdir = snapshots_dir()
    cdir = clips_dir()
    sdir.mkdir(parents=True, exist_ok=True)
    cdir.mkdir(parents=True, exist_ok=True)
    (sdir / "test.jpg").write_text("fake image")
    (cdir / "test.mp4").write_text("fake clip")
    
    response = client.get("/export/media_bundle.tar.gz")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/gzip"
    assert len(response.content) > 0
