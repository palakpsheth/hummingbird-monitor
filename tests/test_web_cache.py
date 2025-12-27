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


@pytest.mark.anyio
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
