from __future__ import annotations

from pathlib import Path
import os
import pytest
from fastapi.testclient import TestClient

from hbmon.config import load_settings, save_settings, Roi
from hbmon.db import init_db, reset_db_state
from hbmon.web import make_app

def _setup_app(tmp_path: Path, monkeypatch) -> TestClient:
    data_dir = tmp_path / "data"
    media = tmp_path / "media"
    db_path = tmp_path / "db.sqlite"
    
    # Force everything to use tmp_path
    monkeypatch.setenv("HBMON_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(media))
    monkeypatch.setenv("HBMON_DB_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("HBMON_DB_ASYNC_URL", f"sqlite+aiosqlite:///{db_path}")
    
    # Clear settings-related env vars that might interfere
    monkeypatch.delenv("HBMON_RTSP_URL", raising=False)
    monkeypatch.delenv("HBMON_BG_SUBTRACTION", raising=False)
    monkeypatch.delenv("HBMON_ROI", raising=False)
    
    reset_db_state()
    data_dir.mkdir(parents=True, exist_ok=True)
    media.mkdir(parents=True, exist_ok=True)
    init_db()
    
    # Ensure config.json is fresh or matches our expectations
    conf_p = data_dir / "config.json"
    if conf_p.exists():
        conf_p.unlink()
        
    app = make_app()
    return TestClient(app)

def test_calibrate_page_rendering(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    response = client.get("/calibrate")
    assert response.status_code == 200
    assert "Calibrate ROI" in response.text

def test_get_roi_api(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    s = load_settings()
    s.roi = Roi(x1=0.1, y1=0.2, x2=0.5, y2=0.6)
    save_settings(s)
    
    response = client.get("/api/roi")
    assert response.status_code == 200
    data = response.json()
    assert float(data["x1"]) == 0.1
    assert float(data["y1"]) == 0.2

def test_post_roi_api_valid(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    roi_data = {"x1": 0.05, "y1": 0.05, "x2": 0.5, "y2": 0.5}
    response = client.post("/api/roi", data=roi_data, follow_redirects=False)
    assert response.status_code == 303
    
    s = load_settings()
    assert s.roi is not None
    assert s.roi.x1 == 0.05

def test_post_roi_api_invalid(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    response = client.post("/api/roi", data={"x1": 0.5})
    assert response.status_code == 422

def test_api_health(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["db_ok"] is True

def test_api_background_get_metadata_empty(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    s = load_settings()
    s.background_image = ""
    save_settings(s)
    
    monkeypatch.setattr("hbmon.web.background_image_path", lambda: tmp_path / "data/background/not-there.jpg")

    response = client.get("/api/background")
    assert response.status_code == 200
    data = response.json()
    assert data["configured"] is False

@pytest.mark.anyio
async def test_async_session_adapter_cleanup_logic(tmp_path, monkeypatch):
    from hbmon.web import _AsyncSessionAdapter, get_session_factory
    from hbmon.db import init_db, reset_db_state
    from sqlalchemy import text
    
    data_dir = tmp_path / "data"
    monkeypatch.setenv("HBMON_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HBMON_DB_URL", f"sqlite:///{tmp_path}/db_async.sqlite")
    reset_db_state()
    data_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    
    adapter = _AsyncSessionAdapter(get_session_factory())
    await adapter.execute(text("SELECT 1"))
    await adapter.close()
    assert adapter._closed is True
