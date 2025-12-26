"""
Basic checks for snapshot zoom UI wiring in templates.
"""

from pathlib import Path


TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "src" / "hbmon" / "templates"


def test_base_includes_zoom_script():
    base = (TEMPLATE_DIR / "base.html").read_text()
    assert "zoom.js" in base


def test_observation_detail_has_zoom_controls():
    tpl = (TEMPLATE_DIR / "observation_detail.html").read_text()
    assert "data-zoom-container" in tpl
    assert "data-zoom-target" in tpl
    assert "data-zoom-in" in tpl
    assert "data-zoom-out" in tpl
    assert "data-zoom-reset" in tpl
    assert "data-snapshot-view" in tpl
    assert "data-default-view" in tpl
