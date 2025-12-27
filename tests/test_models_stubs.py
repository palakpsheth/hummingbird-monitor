"""
Tests for lightweight stubs in ``hbmon.models`` when SQLAlchemy is not available.

These tests exercise the behavior of the dataclass-based fallback classes
provided when SQLAlchemy is missing.  They ensure that embedding
serialization/deserialization works, that properties compute as expected,
and that extra JSON metadata round-trips correctly.
"""

import builtins
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

import hbmon.models as models


def _load_models_without_sqlalchemy(monkeypatch) -> object:
    module_path = Path(models.__file__).resolve()
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("sqlalchemy"):
            raise ImportError("Forced missing SQLAlchemy for stub tests")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    spec = importlib.util.spec_from_file_location("hbmon.models_stub", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load hbmon.models for stub tests")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _get_stub_models(monkeypatch):
    sa_available = getattr(models, "_SQLALCHEMY_AVAILABLE", True)
    if not sa_available:
        return models
    return _load_models_without_sqlalchemy(monkeypatch)


def test_embedding_pack_unpack_roundtrip():
    # Test the private pack/unpack helpers work for arbitrary vectors
    vec = np.random.randn(8).astype(np.float32)
    blob = models._pack_embedding(vec)
    out = models._unpack_embedding(blob)
    assert out.dtype == np.float32
    assert out.shape == (8,)
    assert np.allclose(vec.astype(np.float32).reshape(-1), out)


def test_individual_stub_prototype_and_last_seen(monkeypatch):
    stub_models = _get_stub_models(monkeypatch)

    ind = stub_models.Individual(id=42)
    assert ind.visit_count == 0
    # Set and retrieve prototype vector
    vec = np.random.randn(5).astype(np.float32)
    ind.set_prototype(vec)
    proto = ind.get_prototype()
    assert proto is not None
    assert proto.dtype == np.float32
    assert np.allclose(proto, vec.astype(np.float32))
    # last_seen_utc should be None initially
    assert ind.last_seen_utc is None
    # After assigning a datetime, last_seen_utc returns ISO string
    now = datetime(2020, 1, 1, 12, 30, 45, tzinfo=timezone.utc)
    ind.last_seen_at = now
    assert ind.last_seen_utc == "2020-01-01T12:30:45Z"


def test_observation_stub_bbox_and_extra(monkeypatch):
    stub_models = _get_stub_models(monkeypatch)

    obs = stub_models.Observation(
        id=1,
        bbox_x1=1,
        bbox_y1=2,
        bbox_x2=3,
        bbox_y2=4,
        snapshot_path="snap.jpg",
        video_path="clip.mp4",
    )
    # bbox_xyxy should return a tuple of ints
    assert obs.bbox_xyxy == (1, 2, 3, 4)
    # bbox_str should format correctly
    assert obs.bbox_str == "1,2,3,4"
    # JSON metadata round-trip
    data = {"score": 0.9, "foo": [1, 2]}
    obs.set_extra(data)
    back = obs.get_extra()
    assert back == json.loads(json.dumps(data, sort_keys=True))
    # get_extra returns None for empty string
    obs.extra_json = ""
    assert obs.get_extra() is None
    merged = obs.merge_extra({"review": {"label": "true_positive"}})
    assert merged["review"]["label"] == "true_positive"
    assert obs.review_label == "true_positive"


def test_embedding_stub_set_get_vector(monkeypatch):
    stub_models = _get_stub_models(monkeypatch)

    emb = stub_models.Embedding(id=1, observation_id=1)
    vec = np.random.randn(6).astype(np.float32)
    emb.set_vec(vec)
    out = emb.get_vec()
    assert out.dtype == np.float32
    assert out.shape == (6,)
    assert np.allclose(out, vec.astype(np.float32))
