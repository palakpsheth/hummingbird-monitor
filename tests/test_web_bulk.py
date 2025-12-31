from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from hbmon.config import media_dir
from hbmon.db import get_async_session_factory, init_db, reset_db_state
from hbmon.models import Individual, Observation, Embedding
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
async def test_bulk_delete_observations(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    m = media_dir()
    
    # Create fake files
    snap1 = m / "o1.jpg"
    snap2 = m / "o2.jpg"
    snap1.write_text("fake 1")
    snap2.write_text("fake 2")

    async with get_async_session_factory()() as db:
        # Create an individual to test stat recomputation
        ind = Individual(name="Indy", visit_count=10)
        db.add(ind)
        await db.commit()
        await db.refresh(ind)
        ind_id = ind.id

        o1 = Observation(
            snapshot_path="o1.jpg", 
            video_path="o1.mp4",
            individual_id=ind_id,
            species_label="Bird",
            species_prob=0.9
        )
        o2 = Observation(
            snapshot_path="o2.jpg", 
            video_path="o2.mp4",
            individual_id=ind_id,
            species_label="Bird",
            species_prob=0.8
        )
        db.add_all([o1, o2])
        await db.commit()
        await db.refresh(o1)
        await db.refresh(o2)
        
        # Add an embedding to test deletion
        emb1 = Embedding(observation_id=o1.id, individual_id=ind_id, embedding_blob=b"fake")
        db.add(emb1)
        await db.commit()
        
        obs_ids = [o1.id, o2.id]

    print(f"[TEST DEBUG] obs_ids to delete: {obs_ids}")

    # Bulk delete - ensure multiple values for the same key are sent correctly
    response = client.post(
        "/observations/bulk_delete", 
        data={"obs_ids": [str(obs_ids[0]), str(obs_ids[1])]}, 
        follow_redirects=True
    )
    assert response.status_code == 200
    
    # Verify DB state
    async with get_async_session_factory()() as db:
        obs_res = await db.execute(select(Observation).where(Observation.id.in_(obs_ids)))
        obs_list = obs_res.scalars().all()
        assert len(obs_list) == 0
        
        emb_res = await db.execute(select(Embedding).where(Embedding.observation_id.in_(obs_ids)))
        assert len(emb_res.scalars().all()) == 0
        
        # Verify individual stats updated (should be 0 visits now)
        updated_ind = await db.get(Individual, ind_id)
        assert updated_ind.visit_count == 0

    # Verify filenames unlinked
    assert not snap1.exists()
    assert not snap2.exists()


@pytest.mark.anyio
async def test_bulk_delete_observations_empty(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    response = client.post("/observations/bulk_delete", data={"obs_ids": []}, follow_redirects=True)
    assert response.status_code == 200
    assert "/observations" in response.url.path
