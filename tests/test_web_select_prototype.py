from __future__ import annotations

import importlib

import numpy as np
import pytest

from hbmon import db as db_module
from hbmon.models import Embedding, Individual, Observation, utcnow


def _import_web(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    media_dir = tmp_path / "media"
    db_path = tmp_path / "db.sqlite"
    monkeypatch.setenv("HBMON_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(media_dir))
    monkeypatch.setenv("HBMON_DB_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("HBMON_DB_ASYNC_URL", f"sqlite+aiosqlite:///{db_path}")
    if "hbmon.web" in importlib.sys.modules:
        importlib.sys.modules.pop("hbmon.web")
    return importlib.import_module("hbmon.web")


@pytest.mark.anyio
async def test_select_prototype_observations_prefers_embeddings(tmp_path, monkeypatch):
    web = _import_web(monkeypatch, tmp_path)
    db_module.reset_db_state()
    await db_module.init_async_db()

    async with db_module.async_session_scope() as session:
        ind_proto = Individual(name="proto", visit_count=0)
        ind_proto.set_prototype(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        ind_fallback = Individual(name="fallback", visit_count=0)
        session.add_all([ind_proto, ind_fallback])
        await session.flush()

        obs1 = Observation(
            ts=utcnow(),
            camera_name="cam",
            species_label="Anna's Hummingbird",
            species_prob=0.9,
            individual_id=ind_proto.id,
            match_score=0.2,
            snapshot_path="s1.jpg",
            video_path="v1.mp4",
        )
        obs2 = Observation(
            ts=utcnow(),
            camera_name="cam",
            species_label="Anna's Hummingbird",
            species_prob=0.8,
            individual_id=ind_proto.id,
            match_score=0.1,
            snapshot_path="s2.jpg",
            video_path="v2.mp4",
        )
        obs3 = Observation(
            ts=utcnow(),
            camera_name="cam",
            species_label="Unknown",
            species_prob=0.1,
            individual_id=ind_fallback.id,
            match_score=0.1,
            snapshot_path="s3.jpg",
            video_path="v3.mp4",
        )
        obs4 = Observation(
            ts=utcnow(),
            camera_name="cam",
            species_label="Unknown",
            species_prob=0.2,
            individual_id=ind_fallback.id,
            match_score=0.9,
            snapshot_path="s4.jpg",
            video_path="v4.mp4",
        )
        session.add_all([obs1, obs2, obs3, obs4])
        await session.flush()

        emb1 = Embedding(observation_id=obs1.id, individual_id=ind_proto.id)
        emb1.set_vec(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        emb2 = Embedding(observation_id=obs2.id, individual_id=ind_proto.id)
        emb2.set_vec(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        session.add_all([emb1, emb2])

    async with db_module.async_session_scope() as session:
        selected = await web.select_prototype_observations(
            session,
            [int(ind_proto.id), int(ind_fallback.id)],
        )

    assert selected[int(ind_proto.id)].id == obs1.id
    assert selected[int(ind_fallback.id)].id == obs4.id
