from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hbmon.db import get_async_session_factory, init_db, reset_db_state
from hbmon.models import Individual, Observation, Embedding, _pack_embedding
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
async def test_refresh_embedding(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    async with get_async_session_factory()() as db:
        ind = Individual(name="Indy")
        db.add(ind)
        await db.commit()
        await db.refresh(ind)
        ind_id = ind.id

        # Use separate observations for each embedding to avoid UNIQUE constraint
        o1 = Observation(snapshot_path="o1.jpg", video_path="o1.mp4", individual_id=ind_id)
        o2 = Observation(snapshot_path="o2.jpg", video_path="o2.mp4", individual_id=ind_id)
        db.add_all([o1, o2])
        await db.commit()
        await db.refresh(o1)
        await db.refresh(o2)

        # Add some embeddings
        vec1 = np.ones(128, dtype=np.float32) * 0.5
        vec2 = np.ones(128, dtype=np.float32) * 0.7
        emb1 = Embedding(observation_id=o1.id, individual_id=ind_id, embedding_blob=_pack_embedding(vec1))
        emb2 = Embedding(observation_id=o2.id, individual_id=ind_id, embedding_blob=_pack_embedding(vec2))
        db.add_all([emb1, emb2])
        await db.commit()

    response = client.post(f"/individuals/{ind_id}/refresh_embedding", follow_redirects=True)
    assert response.status_code == 200

    async with get_async_session_factory()() as db:
        updated = await db.get(Individual, ind_id)
        proto = updated.get_prototype()
        assert proto is not None
        # Mean should be 0.6, then L2 normalized
        expected = (vec1 + vec2) / 2
        # L2 norm of [0.6]*128 is sqrt(128 * 0.6^2)
        # after normalization, it should be [1/sqrt(128)]*128
        norm = np.linalg.norm(expected)
        expected_norm = expected / norm
        np.testing.assert_allclose(proto, expected_norm, atol=1e-5)


@pytest.mark.asyncio
async def test_split_review_rendering(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    async with get_async_session_factory()() as db:
        ind = Individual(name="Indy")
        db.add(ind)
        await db.commit()
        await db.refresh(ind)
        ind_id = ind.id

        # Need embeddings for suggest_split_two_groups to work
        o1 = Observation(snapshot_path="o1.jpg", video_path="o1.mp4", individual_id=ind_id)
        o2 = Observation(snapshot_path="o2.jpg", video_path="o2.mp4", individual_id=ind_id)
        db.add_all([o1, o2])
        await db.commit()
        await db.refresh(o1)
        await db.refresh(o2)

        v1 = np.array([1.0, 0.0] + [0.0]*126, dtype=np.float32)
        v2 = np.array([0.0, 1.0] + [0.0]*126, dtype=np.float32)
        db.add(Embedding(observation_id=o1.id, individual_id=ind_id, embedding_blob=_pack_embedding(v1)))
        db.add(Embedding(observation_id=o2.id, individual_id=ind_id, embedding_blob=_pack_embedding(v2)))
        await db.commit()

    response = client.get(f"/individuals/{ind_id}/split_review")
    assert response.status_code == 200
    assert "Split review" in response.text
    assert "Indy" in response.text

@pytest.mark.asyncio
async def test_refresh_embedding_no_embs(tmp_path, monkeypatch):
    client = _setup_app(tmp_path, monkeypatch)
    async with get_async_session_factory()() as db:
        ind = Individual(name="Indy")
        db.add(ind)
        await db.commit()
        await db.refresh(ind)
        ind_id = ind.id

    response = client.post(f"/individuals/{ind_id}/refresh_embedding", follow_redirects=True)
    assert response.status_code == 200
    assert f"/individuals/{ind_id}" in response.url.path
