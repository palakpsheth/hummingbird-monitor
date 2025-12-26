"""
Tests for per-individual prototype observation selection in the web UI.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import hbmon.db as db
import numpy as np

from hbmon.models import Embedding, Individual, Observation
from hbmon.web import select_prototype_observations


def _setup_db(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.setenv("HBMON_DB_URL", f"sqlite:///{tmp_path/'db.sqlite'}")
    monkeypatch.setattr(db, "_ENGINE", None)
    monkeypatch.setattr(db, "_SessionLocal", None)
    db.init_db()


def _make_observation(
    *,
    individual_id: int,
    match_score: float,
    ts: datetime,
) -> Observation:
    return Observation(
        individual_id=individual_id,
        species_label="Hummingbird",
        species_prob=0.5,
        snapshot_path=f"snap-{individual_id}-{match_score}.jpg",
        video_path=f"vid-{individual_id}-{match_score}.mp4",
        match_score=match_score,
        ts=ts,
    )


def test_select_prototype_observations_prefers_embedding_similarity(monkeypatch, tmp_path):
    _setup_db(monkeypatch, tmp_path)

    with db.session_scope() as session:
        ind = Individual(name="A")
        session.add(ind)
        session.commit()
        ind_id = ind.id
        ind.set_prototype(np.array([1.0, 0.0], dtype=np.float32))
        base_ts = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        obs_preferred = _make_observation(individual_id=ind_id, match_score=0.1, ts=base_ts)
        obs_other = _make_observation(individual_id=ind_id, match_score=0.9, ts=base_ts + timedelta(minutes=5))
        session.add_all([obs_preferred, obs_other])
        session.commit()
        emb_preferred = Embedding(observation_id=obs_preferred.id, individual_id=ind_id, embedding_blob=b"")
        emb_preferred.set_vec(np.array([1.0, 0.0], dtype=np.float32))
        emb_other = Embedding(observation_id=obs_other.id, individual_id=ind_id, embedding_blob=b"")
        emb_other.set_vec(np.array([0.0, 1.0], dtype=np.float32))
        session.add_all([emb_preferred, emb_other])

    with db.session_scope() as session:
        result = select_prototype_observations(session, [ind_id])
        assert ind_id in result
        assert result[ind_id].match_score == 0.1
        expected_ts = base_ts.replace(tzinfo=None)
        assert result[ind_id].ts == expected_ts


def test_select_prototype_observations_handles_multiple_individuals(monkeypatch, tmp_path):
    _setup_db(monkeypatch, tmp_path)

    with db.session_scope() as session:
        ind_a = Individual(name="A")
        ind_b = Individual(name="B")
        session.add_all([ind_a, ind_b])
        session.commit()
        ind_a_id = ind_a.id
        ind_b_id = ind_b.id
        ind_a.set_prototype(np.array([1.0, 0.0], dtype=np.float32))
        base_ts = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        obs_a = _make_observation(individual_id=ind_a_id, match_score=0.7, ts=base_ts)
        obs_b = _make_observation(individual_id=ind_b_id, match_score=0.6, ts=base_ts)
        session.add_all([obs_a, obs_b])
        session.commit()
        emb_a = Embedding(observation_id=obs_a.id, individual_id=ind_a_id, embedding_blob=b"")
        emb_a.set_vec(np.array([1.0, 0.0], dtype=np.float32))
        session.add(emb_a)

    with db.session_scope() as session:
        result = select_prototype_observations(session, [ind_a_id, ind_b_id])

    assert set(result.keys()) == {ind_a_id, ind_b_id}


def test_select_prototype_observations_empty_list(monkeypatch, tmp_path):
    _setup_db(monkeypatch, tmp_path)

    with db.session_scope() as session:
        result = select_prototype_observations(session, [])

    assert result == {}


def test_select_prototype_observations_falls_back_without_embeddings(monkeypatch, tmp_path):
    _setup_db(monkeypatch, tmp_path)

    with db.session_scope() as session:
        ind = Individual(name="A")
        session.add(ind)
        session.commit()
        ind_id = ind.id
        base_ts = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        obs_early = _make_observation(individual_id=ind_id, match_score=0.9, ts=base_ts)
        obs_late = _make_observation(individual_id=ind_id, match_score=0.8, ts=base_ts + timedelta(minutes=5))
        session.add_all([obs_early, obs_late])

    with db.session_scope() as session:
        result = select_prototype_observations(session, [ind_id])
        assert ind_id in result
        assert result[ind_id].match_score == 0.9
