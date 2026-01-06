from __future__ import annotations

from pathlib import Path
import json
import pytest
from fastapi.testclient import TestClient

from hbmon.config import media_dir
from hbmon.db import init_db, reset_db_state, get_async_session_factory
from hbmon.models import Observation
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
    
    init_db()
    
    app = make_app()
    return TestClient(app)

@pytest.mark.asyncio
async def test_observation_detail_rendering(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    async with get_async_session_factory()() as db:
        obs = Observation(
            species_label="Test Bird",
            species_prob=0.99,
            snapshot_path="snap.jpg",
            video_path="vid.mp4",
            extra_json=json.dumps({
                "detector": {"name": "yolo"},
                "review": {"label": "true_positive"}
            })
        )
        db.add(obs)
        await db.commit()
        await db.refresh(obs)
        obs_id = obs.id
        
    response = client.get(f"/observations/{obs_id}")
    assert response.status_code == 200
    assert "Test Bird" in response.text
    assert "detector" in response.text

@pytest.mark.asyncio
async def test_api_video_info_missing(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    async with get_async_session_factory()() as db:
        obs = Observation(snapshot_path="s.jpg", video_path="missing.mp4")
        db.add(obs)
        await db.commit()
        await db.refresh(obs)
        obs_id = obs.id
        
    response = client.get(f"/api/video_info/{obs_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["file_exists"] is False

@pytest.mark.asyncio
async def test_observation_label(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    async with get_async_session_factory()() as db:
        obs = Observation(snapshot_path="s.jpg", video_path="v.mp4")
        db.add(obs)
        await db.commit()
        await db.refresh(obs)
        obs_id = obs.id
        
    response = client.post(f"/observations/{obs_id}/label", data={"label": "true_positive"}, follow_redirects=True)
    assert response.status_code == 200
    
    async with get_async_session_factory()() as db:
        updated = await db.get(Observation, obs_id)
        extra = updated.get_extra()
        assert extra["review"]["label"] == "true_positive"

@pytest.mark.asyncio
async def test_export_integration_test(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    m = media_dir()
    snap = m / "snap.jpg"
    vid = m / "vid.mp4"
    snap.write_text("fake image")
    vid.write_text("fake video")
    
    async with get_async_session_factory()() as db:
        obs = Observation(
            snapshot_path="snap.jpg",
            video_path="vid.mp4",
            bbox_x1=10, bbox_y1=10, bbox_x2=100, bbox_y2=100
        )
        db.add(obs)
        await db.commit()
        await db.refresh(obs)
        obs_id = obs.id
        
    response = client.post(f"/observations/{obs_id}/export_integration_test", data={"case_name": "test_case"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/gzip"
    assert len(response.content) > 0
