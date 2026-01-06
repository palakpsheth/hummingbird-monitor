"""
Tests for cached latest-observation lookups in the web module.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

import hbmon.db as db
from hbmon.models import Observation
from hbmon.web import _get_latest_observation_data


def _setup_db(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.setenv("HBMON_DB_URL", f"sqlite:///{tmp_path/'db.sqlite'}")
    monkeypatch.delenv("HBMON_DB_ASYNC_URL", raising=False)
    monkeypatch.setattr(db, "_ENGINE", None)
    monkeypatch.setattr(db, "_SessionLocal", None)
    monkeypatch.setattr(db, "_ASYNC_ENGINE", None)
    monkeypatch.setattr(db, "_AsyncSessionLocal", None)
    db.init_db()


@pytest.mark.asyncio
async def test_latest_observation_cache_validation(monkeypatch, tmp_path) -> None:
    _setup_db(monkeypatch, tmp_path)

    ts = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    with db.session_scope() as session:
        obs = Observation(
            species_label="Hummingbird",
            species_prob=0.9,
            snapshot_path="snap.jpg",
            video_path="clip.mp4",
            ts=ts,
        )
        session.add(obs)
        session.commit()
        obs_id = obs.id
        expected_ts_utc = obs.ts_utc

    async def _fake_cache_get_json(key: str):
        return ["bad"]

    async def _fake_cache_set_json(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr("hbmon.web.cache_get_json", _fake_cache_get_json)
    monkeypatch.setattr("hbmon.web.cache_set_json", _fake_cache_set_json)

    async with db.async_session_scope() as session:
        data = await _get_latest_observation_data(session)

    assert data is not None
    assert data["id"] == obs_id
    assert data["ts_utc"] == expected_ts_utc

@pytest.mark.asyncio
async def test_index_caching(monkeypatch, tmp_path) -> None:
    _setup_db(monkeypatch, tmp_path)
    from hbmon.web import make_app
    from fastapi.testclient import TestClient
    app = make_app()
    client = TestClient(app)

    # 1. First request (Miss)
    r1 = client.get("/")
    assert r1.status_code == 200
    
    # 2. Mock cache hit
    async def _fake_cache_get_json(key: str):
        if "index" in key:
            return {
                "top_inds_out": [
                    {
                        "id": 1,
                        "name": "Cached Bird",
                        "visit_count": 5,
                        "last_seen_at": "2024-01-01T12:00:00Z",
                        "created_at": "2024-01-01T10:00:00Z",
                        "last_species": "Anna's Hummingbird",
                        "species_css": "species-annas-hummingbird",
                    }
                ],
                "recent": [],
                "current_page": 1,
                "clamped_page_size": 10,
                "total_pages": 1,
                "total_recent": 1,
                "ind_current_page": 1,
                "ind_clamped_page_size": 10,
                "ind_total_pages": 1,
                "total_individuals": 1,
                "last_capture_utc": "2024-01-01T12:00:00Z",
            }
        return None
    
    monkeypatch.setattr("hbmon.web.cache_get_json", _fake_cache_get_json)
    
    r2 = client.get("/")
    assert r2.status_code == 200
    assert "Cached Bird" in r2.text
