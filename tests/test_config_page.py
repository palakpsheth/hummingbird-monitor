from test_web_helpers import _import_web


def test_validate_detection_inputs_ok(monkeypatch):
    web = _import_web(monkeypatch)
    parsed, errors = web._validate_detection_inputs(
        {
            "detect_conf": "0.35",
            "detect_iou": "0.60",
            "min_box_area": "1200",
            "cooldown_seconds": "3.5",
            "min_species_prob": "0.40",
            "match_threshold": "0.20",
            "ema_alpha": "0.15",
        }
    )
    assert errors == []
    assert parsed["detect_conf"] == 0.35
    assert parsed["detect_iou"] == 0.60
    assert parsed["min_box_area"] == 1200
    assert parsed["cooldown_seconds"] == 3.5
    assert parsed["min_species_prob"] == 0.40
    assert parsed["match_threshold"] == 0.20
    assert parsed["ema_alpha"] == 0.15


def test_validate_detection_inputs_errors(monkeypatch):
    web = _import_web(monkeypatch)
    _, errors = web._validate_detection_inputs(
        {
            "detect_conf": "1.5",
            "detect_iou": "not-a-number",
            "min_box_area": "0.5",
            "cooldown_seconds": "-2",
            "min_species_prob": "2",
            "match_threshold": "-0.1",
            "ema_alpha": "abc",
        }
    )
    assert any("between 0.05 and 0.95" in e for e in errors)
    assert any("must be a number" in e for e in errors)
    assert any("whole number" in e for e in errors)
    assert any("Cooldown seconds" in e for e in errors)
    assert any("Minimum species probability" in e for e in errors)
    assert any("Match threshold" in e for e in errors)
    assert any("EMA alpha" in e for e in errors)
