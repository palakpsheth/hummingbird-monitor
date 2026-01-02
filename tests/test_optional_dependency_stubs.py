from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def _load_module_with_missing_deps(module_name: str, path: Path, missing: list[str]):
    sentinel = object()
    originals = {dep: sys.modules.get(dep, sentinel) for dep in missing}
    try:
        for dep in missing:
            sys.modules[dep] = None
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for dep, original in originals.items():
            if original is sentinel:
                sys.modules.pop(dep, None)
            else:
                sys.modules[dep] = original
        sys.modules.pop(module_name, None)


def test_schema_dataclass_stubs():
    import hbmon.schema as schema_ref

    mod = _load_module_with_missing_deps(
        "hbmon.schema_stub",
        Path(schema_ref.__file__),
        ["pydantic"],
    )

    assert mod._PYDANTIC_AVAILABLE is False
    roi = mod.RoiIn(x1=0.1, y1=0.2, x2=0.8, y2=0.9)
    assert roi.x1 == 0.1
    assert roi.y2 == 0.9

    health = mod.HealthOut(ok=True, version="0.0.0", db_ok=False)
    assert health.last_observation_utc is None
    assert health.rtsp_url is None

    obs = mod.ObservationOut(
        id=1,
        ts_utc="2025-01-01T00:00:00Z",
        camera_name=None,
        species_label="Test",
        species_prob=0.5,
        individual_id=None,
        match_score=0.0,
        snapshot_path="snap.jpg",
        video_path="vid.mp4",
    )
    assert obs.bbox_xyxy is None

    ind = mod.IndividualOut(
        id=10,
        name="Birdy",
        visit_count=3,
        created_utc="2025-01-01T00:00:00Z",
        last_seen_utc=None,
    )
    assert ind.last_species_label is None


def test_models_dataclass_stubs():
    import hbmon.models as models_ref

    mod = _load_module_with_missing_deps(
        "hbmon.models_stub",
        Path(models_ref.__file__),
        ["sqlalchemy", "sqlalchemy.orm"],
    )

    assert mod._SQLALCHEMY_AVAILABLE is False

    ind = mod.Individual(id=1, name="Test")
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ind.set_prototype(vec)
    restored = ind.get_prototype()
    assert restored is not None
    assert np.allclose(restored, vec)

    obs = mod.Observation(id=2, snapshot_path="snap.jpg", video_path="vid.mp4")
    assert obs.bbox_xyxy is None
    obs.bbox_x1 = 1
    obs.bbox_y1 = 2
    obs.bbox_x2 = 3
    obs.bbox_y2 = 4
    assert obs.bbox_xyxy == (1, 2, 3, 4)
    assert obs.bbox_str == "1,2,3,4"

    obs.set_extra({"review": {"label": "true_positive"}, "meta": {"x": 1}})
    assert obs.get_extra()["review"]["label"] == "true_positive"
    merged = obs.merge_extra({"meta": {"y": 2}})
    assert merged["meta"]["x"] == 1
    assert merged["meta"]["y"] == 2
    assert obs.review_label == "true_positive"

    candidate = mod.Candidate(id=5, snapshot_path="snap.jpg")
    candidate.set_extra({"reason": "motion_rejected"})
    assert candidate.get_extra()["reason"] == "motion_rejected"

    emb = mod.Embedding(id=3, observation_id=obs.id)
    emb.set_vec(vec)
    unpacked = emb.get_vec()
    assert np.allclose(unpacked, vec)

    frame = mod.AnnotationFrame(
        id=10,
        observation_id=obs.id,
        frame_index=0,
        frame_path="frames/obs/frame_000001.jpg",
    )
    assert frame.bird_present is False
    assert frame.status == "queued"

    box = mod.AnnotationBox(
        id=11,
        frame_id=frame.id,
        class_id=0,
        x=0.5,
        y=0.5,
        w=0.2,
        h=0.2,
    )
    assert box.is_false_positive is False
