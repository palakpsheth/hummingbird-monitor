from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from hbmon.config import load_settings
from hbmon.db import init_db, reset_db_state
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
    
    conf_p = data_dir / "config.json"
    if conf_p.exists():
        conf_p.unlink()
        
    app = make_app()
    return TestClient(app)

def _valid_form_data():
    return {
        "detect_conf": "0.25",
        "detect_iou": "0.45",
        "min_box_area": "600",
        "cooldown_seconds": "2.0",
        "min_species_prob": "0.35",
        "match_threshold": "0.25",
        "ema_alpha": "0.10",
        "bg_motion_threshold": "30",
        "bg_motion_blur": "5",
        "bg_min_overlap": "0.15",
        "timezone": "local",
        # New fields
        "fps_limit": "8",
        "crop_padding": "0.05",
        "bg_rejected_cooldown_seconds": "3.0",
        "arrival_buffer_seconds": "5.0",
        "departure_timeout_seconds": "2.0",
        "post_departure_buffer_seconds": "3.0",
    }

def test_config_page_rendering(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    response = client.get("/config")
    assert response.status_code == 200
    assert "Config" in response.text

def test_config_save_valid(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    form_data = _valid_form_data()
    form_data.update({
        "detect_conf": "0.50",
        "min_box_area": "1000",
        "timezone": "UTC",
        "bg_subtraction_enabled": "on"
    })
    # config_save redirects to /config?saved=1
    response = client.post("/config", data=form_data, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/config?saved=1"
    
    s = load_settings()
    assert s.detect_conf == 0.5
    assert s.min_box_area == 1000
    assert s.timezone == "UTC"
    assert s.bg_subtraction_enabled is True

def test_config_save_invalid_nan(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    form_data = _valid_form_data()
    form_data["detect_conf"] = "abc"
    response = client.post("/config", data=form_data)
    # web.py returns HTMLResponse with status_code=400 when errors exist
    assert response.status_code == 400
    assert "must be a number" in response.text

def test_config_save_invalid_range(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    form_data = _valid_form_data()
    form_data["detect_conf"] = "1.5"
    response = client.post("/config", data=form_data)
    assert response.status_code == 400
    assert "must be between 0.05 and 0.95" in response.text

def test_config_save_invalid_odd_int(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    form_data = _valid_form_data()
    form_data["bg_motion_blur"] = "4"
    response = client.post("/config", data=form_data)
    assert response.status_code == 400
    assert "must be an odd number" in response.text

def test_config_save_invalid_timezone(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    form_data = _valid_form_data()
    form_data["timezone"] = "Invalid/Zone"
    response = client.post("/config", data=form_data)
    assert response.status_code == 400
    assert "Timezone must be a valid IANA name" in response.text
