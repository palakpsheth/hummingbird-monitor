from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from hbmon.config import media_dir
from hbmon.db import get_async_session_factory, init_db, reset_db_state
from hbmon.models import Candidate
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


@pytest.mark.anyio
async def test_candidate_list_and_filtering(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    async with get_async_session_factory()() as db:
        c1 = Candidate(
            snapshot_path="c1.jpg",
            extra_json=json.dumps({"reason": "motion", "review": {"label": "bird"}})
        )
        c2 = Candidate(
            snapshot_path="c2.jpg",
            extra_json=json.dumps({"reason": "manual", "review": {"label": "not_bird"}})
        )
        db.add_all([c1, c2])
        await db.commit()

    # List all
    response = client.get("/candidates")
    assert response.status_code == 200
    assert "c1.jpg" in response.text
    assert "c2.jpg" in response.text

    # Filter by label
    response = client.get("/candidates", params={"label": "bird"})
    assert response.status_code == 200
    assert "c1.jpg" in response.text
    assert "c2.jpg" not in response.text

    # Filter by reason
    response = client.get("/candidates", params={"reason": "manual"})
    assert response.status_code == 200
    assert "c1.jpg" not in response.text
    assert "c2.jpg" in response.text


@pytest.mark.anyio
async def test_candidate_detail_rendering(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    async with get_async_session_factory()() as db:
        c = Candidate(
            snapshot_path="cand.jpg",
            extra_json=json.dumps({"reason": "test", "media": {"mask_path": "mask.png"}})
        )
        db.add(c)
        await db.commit()
        await db.refresh(c)
        cid = c.id

    response = client.get(f"/candidates/{cid}")
    assert response.status_code == 200
    assert "cand.jpg" in response.text
    assert "mask.png" in response.text


@pytest.mark.anyio
async def test_candidate_label(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    async with get_async_session_factory()() as db:
        c = Candidate(snapshot_path="c.jpg")
        db.add(c)
        await db.commit()
        await db.refresh(c)
        cid = c.id

    # Post label
    response = client.post(f"/candidates/{cid}/label", data={"label": "true_negative"}, follow_redirects=True)
    assert response.status_code == 200
    
    async with get_async_session_factory()() as db:
        updated = await db.get(Candidate, cid)
        extra = updated.get_extra()
        assert extra["review"]["label"] == "true_negative"


@pytest.mark.anyio
async def test_candidate_export_integration_test(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    m = media_dir()
    snap = m / "snap.jpg"
    snap.write_text("fake image")

    async with get_async_session_factory()() as db:
        c = Candidate(snapshot_path="snap.jpg")
        db.add(c)
        await db.commit()
        await db.refresh(c)
        cid = c.id

    response = client.post(f"/candidates/{cid}/export_integration_test", data={"case_name": "test_cand"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/gzip"
    assert len(response.content) > 0


@pytest.mark.anyio
async def test_bulk_delete_candidates(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    async with get_async_session_factory()() as db:
        c1 = Candidate(snapshot_path="c1.jpg")
        c2 = Candidate(snapshot_path="c2.jpg")
        db.add_all([c1, c2])
        await db.commit()
        await db.refresh(c1)
        await db.refresh(c2)
        ids = [c1.id, c2.id]

    # Bulk delete
    response = client.post("/candidates/bulk_delete", data={"candidate_ids": ids}, follow_redirects=True)
    assert response.status_code == 200
    
    async with get_async_session_factory()() as db:
        res = await db.execute(select(Candidate).where(Candidate.id.in_(ids)))
        assert len(res.scalars().all()) == 0
