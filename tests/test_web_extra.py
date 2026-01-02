import pytest
from unittest.mock import MagicMock, patch
from hbmon.web import (
    _load_cv2,
    _get_db_dialect_name,
    _candidate_json_value,
)
import sys
from fastapi import HTTPException

def test_load_cv2_not_found(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    # Mock find_spec to return None to simulate missing package
    with patch("importlib.util.find_spec", return_value=None):
        with pytest.raises(HTTPException) as excinfo:
            _load_cv2()
        assert excinfo.value.status_code == 503

def test_get_db_dialect_name_various():
    db = MagicMock()
    # CASE 1: No bind
    db.bind = None
    if hasattr(db, "get_bind"):
        del db.get_bind
    assert _get_db_dialect_name(db) == ""
    
    # CASE 2: bind has dialect
    db.bind = MagicMock()
    db.bind.dialect.name = "sqlite"
    assert _get_db_dialect_name(db) == "sqlite"
    
    # CASE 3: sync_engine dialect
    del db.bind.dialect
    db.bind.sync_engine.dialect.name = "postgresql"
    assert _get_db_dialect_name(db) == "postgresql"

def test_candidate_json_value_dialects():
    expr = "extra_json"
    path = ["review", "label"]
    
    # SQLite
    assert "json_extract" in str(_candidate_json_value(expr, path, "sqlite"))
    
    # Postgres
    assert "jsonb_extract_path_text" in str(_candidate_json_value(expr, path, "postgres"))
    
    # Unsupported
    assert _candidate_json_value(expr, path, "mysql") is None
    
    # Empty path
    assert _candidate_json_value(expr, [], "sqlite") is None

from hbmon.web import _flatten_extra_metadata, _validate_detection_inputs, _prepare_observation_extras, paginate

def test_flatten_extra_metadata():
    extra = {
        "detection": {"conf": 0.5, "box": [1, 2, 3, 4]},
        "identification": {"species": "Anna"},
        "nested": {"a": {"b": 1}}
    }
    flattened = _flatten_extra_metadata(extra)
    assert flattened["detection.conf"] == 0.5
    assert flattened["nested.a.b"] == 1
    assert _flatten_extra_metadata(None) == {}
    assert _flatten_extra_metadata([]) == {}

def test_validate_detection_inputs():
    valid_raw = {
        "detect_conf": "0.5",
        "detect_iou": "0.45",
        "min_box_area": "500",
        "cooldown_seconds": "3.0",
        "min_species_prob": "0.8",
        "match_threshold": "0.5",
        "ema_alpha": "0.1",
        "bg_motion_threshold": "30",
        "bg_motion_blur": "5",
        "bg_min_overlap": "0.15",
        "bg_subtraction_enabled": "true",
        "timezone": "America/Los_Angeles",
        # New fields
        "fps_limit": "10",
        "temporal_window_frames": "5",
        "temporal_min_detections": "1",
        "crop_padding": "0.10",
        "bg_rejected_cooldown_seconds": "3.0",
        "arrival_buffer_seconds": "5.0",
        "departure_timeout_seconds": "2.0",
        "post_departure_buffer_seconds": "3.0",
    }
    parsed, errors = _validate_detection_inputs(valid_raw)
    assert not errors
    assert parsed["detect_conf"] == 0.5
    assert parsed["bg_motion_blur"] == 5
    assert parsed["timezone"] == "America/Los_Angeles"

    # Test invalid values
    invalid_raw = {
        "detect_conf": "1.5", # > 0.95
        "bg_motion_blur": "4", # not odd
        "timezone": "Invalid/Place"
    }
    parsed, errors = _validate_detection_inputs(invalid_raw)
    assert len(errors) >= 3
    assert "Detection confidence" in errors[0]
    assert "Background motion blur" in " ".join(errors)
    assert "Timezone" in " ".join(errors)

def test_paginate():
    assert paginate(100, 1, 10) == (1, 10, 10, 0)
    assert paginate(100, 2, 10) == (2, 10, 10, 10)
    assert paginate(0, 1, 10) == (1, 10, 1, 0)
    assert paginate(100, 11, 10) == (10, 10, 10, 90) # clamps to max

def test_prepare_observation_extras():
    class MockObs:
        def __init__(self, extra):
            self.extra = extra
        def get_extra(self):
            return self.extra
    
    obs_list = [
        MockObs({"detection": {"box_confidence": 0.9}, "foo": "bar"}),
        MockObs({"detection": {"box_confidence": 0.8}, "snapshots": {"path": "a.jpg"}})
    ]
    columns, sort_types, labels = _prepare_observation_extras(obs_list)
    assert "detection.box_confidence" in columns
    assert "foo" in columns
    assert "snapshots.path" not in columns # Filtered out by _is_snapshot_key
    assert hasattr(obs_list[0], "extra_display")
