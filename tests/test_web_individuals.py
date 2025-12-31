from __future__ import annotations

from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, func

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
    
    init_db()
    
    app = make_app()
    return TestClient(app)

@pytest.mark.anyio
async def test_individuals_list_rendering(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    async with get_async_session_factory()() as db:
        db.add(Individual(name="Indy 1", visit_count=5))
        db.add(Individual(name="Indy 2", visit_count=10))
        await db.commit()
        
    response = client.get("/individuals")
    assert response.status_code == 200
    assert "Indy 1" in response.text
    assert "Indy 2" in response.text

@pytest.mark.anyio
async def test_individual_detail_rendering(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    async with get_async_session_factory()() as db:
        ind = Individual(name="Indy 1", visit_count=5)
        db.add(ind)
        await db.commit()
        await db.refresh(ind)
        ind_id = ind.id
        
    response = client.get(f"/individuals/{ind_id}")
    assert response.status_code == 200
    assert "Indy 1" in response.text

@pytest.mark.anyio
async def test_rename_individual(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    async with get_async_session_factory()() as db:
        ind = Individual(name="Old Name")
        db.add(ind)
        await db.commit()
        await db.refresh(ind)
        ind_id = ind.id
        
    response = client.post(f"/individuals/{ind_id}/rename", data={"name": "New Name"}, follow_redirects=True)
    assert response.status_code == 200
    assert "New Name" in response.text
    
    async with get_async_session_factory()() as db:
        updated = await db.get(Individual, ind_id)
        assert updated.name == "New Name"

@pytest.mark.anyio
async def test_delete_individual(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    async with get_async_session_factory()() as db:
        ind = Individual(name="To Delete")
        db.add(ind)
        await db.commit()
        await db.refresh(ind)
        ind_id = ind.id
        
        # Add an observation
        db.add(Observation(individual_id=ind_id, species_label="label", snapshot_path="s.jpg", video_path="v.mp4"))
        await db.commit()

    response = client.post(f"/individuals/{ind_id}/delete", follow_redirects=True)
    assert response.status_code == 200
    
    async with get_async_session_factory()() as db:
        ind_deleted = await db.get(Individual, ind_id)
        assert ind_deleted is None
        # Observations for this individual should be deleted too (cascade in web.py delete_individual)
        res = await db.execute(select(func.count(Observation.id)).where(Observation.individual_id == ind_id))
        assert res.scalar_one() == 0

@pytest.mark.anyio
async def test_bulk_delete_individuals(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    async with get_async_session_factory()() as db:
        i1 = Individual(name="A")
        i2 = Individual(name="B")
        db.add_all([i1, i2])
        await db.commit()
        await db.refresh(i1)
        await db.refresh(i2)
        ids = [i1.id, i2.id]
        
    # bulk_delete in web.py uses 'ind_ids' as form field name
    # FastAPI list[int] = Form([]) expect multiple fields with same name ind_ids
    # In httpx, data={'key': ['val1', 'val2']} sends multiple fields.
    response = client.post("/individuals/bulk_delete", data={"ind_ids": [str(ids[0]), str(ids[1])]}, follow_redirects=True)
    assert response.status_code == 200
    
    async with get_async_session_factory()() as db:
        res = await db.execute(select(func.count(Individual.id)).where(Individual.id.in_(ids)))
        assert res.scalar_one() == 0

@pytest.mark.anyio
async def test_split_apply(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    async with get_async_session_factory()() as db:
        ind = Individual(name="Indy A")
        db.add(ind)
        await db.commit()
        await db.refresh(ind)
        ind_id = ind.id
        
        o1 = Observation(individual_id=ind_id, species_label="label", snapshot_path="s1.jpg", video_path="v1.mp4")
        o2 = Observation(individual_id=ind_id, species_label="label", snapshot_path="s2.jpg", video_path="v2.mp4")
        db.add_all([o1, o2])
        await db.commit()
        oid1, oid2 = o1.id, o2.id

    # Split: move o2 to new individual B
    form_data = {
        f"assign_{oid1}": "A",
        f"assign_{oid2}": "B"
    }
    response = client.post(f"/individuals/{ind_id}/split_apply", data=form_data, follow_redirects=True)
    assert response.status_code == 200
    
    async with get_async_session_factory()() as db:
        # Check that a new individual was created
        res = await db.execute(select(Individual).where(Individual.name.like("%split%")))
        ind_b = res.scalar_one()
        assert ind_b is not None
        
        # Check that o2 was reassigned to B
        updated_o2 = await db.get(Observation, oid2)
        assert updated_o2.individual_id == ind_b.id
        # o1 should still be A
        updated_o1 = await db.get(Observation, oid1)
        assert updated_o1.individual_id == ind_id
