from test_web_helpers import _import_web


def test_validate_detection_inputs_ok(monkeypatch):
    web = _import_web(monkeypatch)
    parsed, errors = web._validate_detection_inputs(
        {
            "detect_conf": "0.35",
            "detect_iou": "0.60",
            "min_box_area": "1200",
            "cooldown_seconds": "3.5",
        }
    )
    assert errors == []
    assert parsed["detect_conf"] == 0.35
    assert parsed["detect_iou"] == 0.60
    assert parsed["min_box_area"] == 1200
    assert parsed["cooldown_seconds"] == 3.5


def test_validate_detection_inputs_errors(monkeypatch):
    web = _import_web(monkeypatch)
    _, errors = web._validate_detection_inputs(
        {
            "detect_conf": "1.5",
            "detect_iou": "not-a-number",
            "min_box_area": "0.5",
            "cooldown_seconds": "-2",
        }
    )
    assert any("between 0.05 and 0.95" in e for e in errors)
    assert any("must be a number" in e for e in errors)
    assert any("whole number" in e for e in errors)
    assert any("Cooldown seconds" in e for e in errors)
