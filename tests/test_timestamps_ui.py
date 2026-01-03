"""
Lightweight checks that UI templates mark UTC timestamps for local rendering.
"""

from pathlib import Path


TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "src" / "hbmon" / "templates"


def test_base_includes_timestamp_script():
    base = (TEMPLATE_DIR / "base.html").read_text()
    assert "timestamps.js" in base
    assert "data-hbmon-tz" in base
    assert "footer-current-time" in base


def test_ts_utc_lines_have_data_attribute():
    # Files that contain timestamps (either directly or via macro)
    files_with_ts = [
        "observations.html",  # Uses observation_card macro
        "observation_detail.html",
        "individual_detail.html",
        "split_review.html",
        "_macros.html",  # observation_card macro contains timestamp rendering
    ]
    for name in files_with_ts:
        text = (TEMPLATE_DIR / name).read_text()
        # Filter out lines in Jinja comments {# ... #}
        in_comment = False
        lines = []
        for ln in text.splitlines():
            # Track comment blocks
            if "{#" in ln:
                in_comment = True
            if "#}" in ln:
                in_comment = False
                continue
            # Skip lines inside comment blocks
            if in_comment:
                continue
            # Collect lines with ts_utc outside of comments
            if "ts_utc" in ln:
                lines.append(ln)
        
        assert lines, f"Expected ts_utc usage in {name}"
        for ln in lines:
            assert "data-utc-ts" in ln, f"Missing data-utc-ts in {name}: {ln}"


def test_last_seen_and_capture_marked_for_local():
    idx = (TEMPLATE_DIR / "index.html").read_text()
    assert "data-utc-ts" in idx and "last_capture_utc" in idx

    individuals = (TEMPLATE_DIR / "individuals.html").read_text()
    assert "data-utc-ts" in individuals and "last_seen" in individuals
