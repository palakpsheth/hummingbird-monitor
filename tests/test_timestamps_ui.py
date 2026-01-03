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
            cleaned = ""
            i = 0
            # Strip Jinja comment blocks {# ... #} while preserving code outside comments.
            while i < len(ln):
                if in_comment:
                    end = ln.find("#}", i)
                    if end == -1:
                        # Rest of the line is inside a comment.
                        i = len(ln)
                        break
                    # Exit comment and continue scanning after "#}".
                    in_comment = False
                    i = end + 2
                    continue

                start = ln.find("{#", i)
                if start == -1:
                    # No more comment starts; take the rest of the line as code.
                    cleaned += ln[i:]
                    break

                # Add code before the comment start.
                cleaned += ln[i:start]
                end = ln.find("#}", start + 2)
                if end == -1:
                    # Comment continues on the next line.
                    in_comment = True
                    i = len(ln)
                    break

                # Inline comment; skip it and continue scanning.
                i = end + 2

            # Skip lines that are entirely comments or whitespace after stripping.
            if not cleaned.strip():
                continue
            # Collect lines with ts_utc outside of comments.
            if "ts_utc" in cleaned:
                lines.append(cleaned)
        
        assert lines, f"Expected ts_utc usage in {name}"
        for ln in lines:
            assert "data-utc-ts" in ln, f"Missing data-utc-ts in {name}: {ln}"


def test_last_seen_and_capture_marked_for_local():
    idx = (TEMPLATE_DIR / "index.html").read_text()
    assert "data-utc-ts" in idx and "last_capture_utc" in idx

    individuals = (TEMPLATE_DIR / "individuals.html").read_text()
    assert "data-utc-ts" in individuals and "last_seen" in individuals
