"""
Tests for web module validation functions.

These tests cover form validation, input parsing, and helper functions
in the web module without requiring FastAPI/SQLAlchemy.
"""

from __future__ import annotations

import importlib
from pathlib import Path


def _import_web(monkeypatch):
    """Import hbmon.web after setting safe directories."""
    cwd = Path.cwd().resolve()
    monkeypatch.setenv("HBMON_DATA_DIR", str(cwd / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(cwd / "media"))

    if 'hbmon.web' in importlib.sys.modules:
        importlib.sys.modules.pop('hbmon.web')

    return importlib.import_module('hbmon.web')


def _base_raw() -> dict[str, str]:
    return {
        "detect_conf": "0.35",
        "detect_iou": "0.45",
        "min_box_area": "600",
        "cooldown_seconds": "2.5",
        "min_species_prob": "0.35",
        "match_threshold": "0.25",
        "ema_alpha": "0.10",
        "bg_subtraction_enabled": "1",
        "bg_motion_threshold": "30",
        "bg_motion_blur": "5",
        "bg_min_overlap": "0.15",
        # New fields
        "fps_limit": "8",
        "crop_padding": "0.05",
        "bg_rejected_cooldown_seconds": "3.0",
        "arrival_buffer_seconds": "5.0",
        "departure_timeout_seconds": "2.0",
        "post_departure_buffer_seconds": "3.0",
    }


class TestValidateDetectionInputs:
    """Tests for the _validate_detection_inputs function."""

    def test_valid_inputs(self, monkeypatch):
        """Test validation with all valid inputs."""
        web = _import_web(monkeypatch)

        raw = _base_raw()
        raw["timezone"] = "America/Los_Angeles"

        parsed, errors = web._validate_detection_inputs(raw)

        assert errors == []
        assert parsed["detect_conf"] == 0.35
        assert parsed["detect_iou"] == 0.45
        assert parsed["min_box_area"] == 600
        assert parsed["cooldown_seconds"] == 2.5
        assert parsed["min_species_prob"] == 0.35
        assert parsed["match_threshold"] == 0.25
        assert parsed["ema_alpha"] == 0.10
        assert parsed["timezone"] == "America/Los_Angeles"

    def test_invalid_detect_conf_not_number(self, monkeypatch):
        """Test validation with invalid detect_conf."""
        web = _import_web(monkeypatch)

        raw = _base_raw()
        raw["detect_conf"] = "not_a_number"

        parsed, errors = web._validate_detection_inputs(raw)

        assert "Detection confidence must be a number." in errors

    def test_invalid_detect_conf_out_of_range(self, monkeypatch):
        """Test validation with detect_conf out of range."""
        web = _import_web(monkeypatch)

        raw = _base_raw()
        raw["detect_conf"] = "0.01"  # Below 0.05

        parsed, errors = web._validate_detection_inputs(raw)

        assert any("0.05" in e and "0.95" in e for e in errors)

    def test_invalid_min_box_area_not_integer(self, monkeypatch):
        """Test validation with non-integer min_box_area."""
        web = _import_web(monkeypatch)

        raw = _base_raw()
        raw["min_box_area"] = "600.5"  # Not a whole number

        parsed, errors = web._validate_detection_inputs(raw)

        assert "Minimum box area must be a whole number." in errors

    def test_invalid_min_box_area_out_of_range(self, monkeypatch):
        """Test validation with min_box_area out of range."""
        web = _import_web(monkeypatch)

        raw = _base_raw()
        raw["min_box_area"] = "999999"  # Above 200000

        parsed, errors = web._validate_detection_inputs(raw)

        assert any("1" in e and "200000" in e for e in errors)

    def test_invalid_timezone(self, monkeypatch):
        """Test validation with invalid timezone."""
        web = _import_web(monkeypatch)

        raw = _base_raw()
        raw["timezone"] = "Invalid/Timezone"

        parsed, errors = web._validate_detection_inputs(raw)

        assert any("Timezone" in e for e in errors)

    def test_timezone_local(self, monkeypatch):
        """Test validation with 'local' timezone."""
        web = _import_web(monkeypatch)

        raw = _base_raw()
        raw["timezone"] = "local"

        parsed, errors = web._validate_detection_inputs(raw)

        assert errors == []
        assert parsed["timezone"] == "local"

    def test_timezone_empty(self, monkeypatch):
        """Test validation with empty timezone defaults to local."""
        web = _import_web(monkeypatch)

        raw = _base_raw()
        raw["timezone"] = ""

        parsed, errors = web._validate_detection_inputs(raw)

        assert errors == []
        assert parsed["timezone"] == "local"

    def test_cooldown_seconds_zero(self, monkeypatch):
        """Test validation with zero cooldown seconds."""
        web = _import_web(monkeypatch)

        raw = _base_raw()
        raw["cooldown_seconds"] = "0.0"

        parsed, errors = web._validate_detection_inputs(raw)

        assert "cooldown_seconds" in parsed
        assert parsed["cooldown_seconds"] == 0.0


class TestPaginate:
    """Tests for the paginate function."""

    def test_paginate_normal(self, monkeypatch):
        """Test pagination with normal inputs."""
        web = _import_web(monkeypatch)

        page, size, total_pages, offset = web.paginate(100, 2, 10)

        assert page == 2
        assert size == 10
        assert total_pages == 10
        assert offset == 10

    def test_paginate_zero_total(self, monkeypatch):
        """Test pagination with zero total."""
        web = _import_web(monkeypatch)

        page, size, total_pages, offset = web.paginate(0, 1, 10)

        assert total_pages == 1  # Always at least 1
        assert page == 1
        assert offset == 0

    def test_paginate_page_too_high(self, monkeypatch):
        """Test that page is clamped to max."""
        web = _import_web(monkeypatch)

        page, size, total_pages, offset = web.paginate(50, 100, 10)

        assert total_pages == 5
        assert page == 5  # Clamped from 100
        assert offset == 40

    def test_paginate_page_too_low(self, monkeypatch):
        """Test that page is clamped to 1."""
        web = _import_web(monkeypatch)

        page, size, total_pages, offset = web.paginate(50, 0, 10)

        assert page == 1

    def test_paginate_size_clamped_to_max(self, monkeypatch):
        """Test that page_size is clamped to max."""
        web = _import_web(monkeypatch)

        page, size, total_pages, offset = web.paginate(100, 1, 500, max_page_size=50)

        assert size == 50


class TestSpeciesToCss:
    """Additional tests for species_to_css function."""

    def test_species_with_apostrophe_variants(self, monkeypatch):
        """Test that different apostrophe variants work."""
        web = _import_web(monkeypatch)

        # Standard straight apostrophe
        assert web.species_to_css("Anna's Hummingbird") == "species-anna"
        # Curly apostrophe
        assert web.species_to_css("Anna's Hummingbird") == "species-anna"

    def test_species_case_insensitive(self, monkeypatch):
        """Test that species matching is case insensitive."""
        web = _import_web(monkeypatch)

        assert web.species_to_css("ANNA'S HUMMINGBIRD") == "species-anna"
        assert web.species_to_css("anna's hummingbird") == "species-anna"
        assert web.species_to_css("AnNa'S hUmMiNgBiRd") == "species-anna"

    def test_species_empty_string(self, monkeypatch):
        """Test with empty string."""
        web = _import_web(monkeypatch)

        assert web.species_to_css("") == "species-unknown"

    def test_species_none_like(self, monkeypatch):
        """Test with None-like values."""
        web = _import_web(monkeypatch)

        assert web.species_to_css(None) == "species-unknown"


class TestBuildHourHeatmap:
    """Additional tests for build_hour_heatmap function."""

    def test_heatmap_empty_input(self, monkeypatch):
        """Test heatmap with empty input."""
        web = _import_web(monkeypatch)

        result = web.build_hour_heatmap([])

        assert len(result) == 24
        assert all(h["level"] == 0 for h in result)
        assert all(h["count"] == 0 for h in result)

    def test_heatmap_all_same_count(self, monkeypatch):
        """Test heatmap when all hours have same count."""
        web = _import_web(monkeypatch)

        hours = [(h, 10) for h in range(24)]
        result = web.build_hour_heatmap(hours)

        # All should have level 5 (max)
        assert all(h["level"] == 5 for h in result)

    def test_heatmap_single_hour(self, monkeypatch):
        """Test heatmap with single hour."""
        web = _import_web(monkeypatch)

        hours = [(12, 100)]
        result = web.build_hour_heatmap(hours)

        level_map = {h["hour"]: h["level"] for h in result}
        assert level_map[12] == 5  # Only entry is max
        assert level_map[0] == 0  # Others are zero


class TestPrettyJson:
    """Additional tests for pretty_json function."""

    def test_pretty_json_none(self, monkeypatch):
        """Test with None input."""
        web = _import_web(monkeypatch)

        assert web.pretty_json(None) is None

    def test_pretty_json_empty(self, monkeypatch):
        """Test with empty string."""
        web = _import_web(monkeypatch)

        assert web.pretty_json("") is None

    def test_pretty_json_complex(self, monkeypatch):
        """Test with complex nested JSON."""
        web = _import_web(monkeypatch)

        input_json = '{"nested": {"a": 1, "b": [1, 2, 3]}, "top": "value"}'
        result = web.pretty_json(input_json)

        assert result is not None
        assert "\n" in result
        assert "nested" in result

    def test_pretty_json_invalid_fallback(self, monkeypatch):
        """Test that invalid JSON returns original string."""
        web = _import_web(monkeypatch)

        result = web.pretty_json("{not valid json}")
        assert result == "{not valid json}"

    def test_pretty_json_type_error(self, monkeypatch):
        """Test pretty_json with non-serializable content after parse."""
        web = _import_web(monkeypatch)

        # Valid JSON that might trigger TypeError on re-serialize (rare case)
        result = web.pretty_json('{"key": "value"}')
        assert result is not None


class TestAsUtcStr:
    """Tests for the _as_utc_str function."""

    def test_as_utc_str_none(self, monkeypatch):
        """Test _as_utc_str with None returns None."""
        web = _import_web(monkeypatch)

        result = web._as_utc_str(None)
        assert result is None

    def test_as_utc_str_aware_datetime(self, monkeypatch):
        """Test _as_utc_str with aware datetime."""
        from datetime import datetime, timezone

        web = _import_web(monkeypatch)

        dt = datetime(2024, 1, 15, 12, 30, 45, tzinfo=timezone.utc)
        result = web._as_utc_str(dt)

        assert result == "2024-01-15T12:30:45Z"


class TestAllowedReviewLabels:
    """Tests for the ALLOWED_REVIEW_LABELS constant."""

    def test_allowed_labels_exist(self, monkeypatch):
        """Test that ALLOWED_REVIEW_LABELS is defined."""
        web = _import_web(monkeypatch)

        assert hasattr(web, "ALLOWED_REVIEW_LABELS")
        assert "true_positive" in web.ALLOWED_REVIEW_LABELS
        assert "false_positive" in web.ALLOWED_REVIEW_LABELS
        assert "unknown" in web.ALLOWED_REVIEW_LABELS


class TestToUtc:
    """Tests for the _to_utc function."""

    def test_to_utc_naive(self, monkeypatch):
        """Test _to_utc with naive datetime."""
        from datetime import datetime, timezone

        web = _import_web(monkeypatch)

        naive = datetime(2024, 1, 15, 12, 0, 0)
        result = web._to_utc(naive)

        assert result.tzinfo == timezone.utc

    def test_to_utc_already_utc(self, monkeypatch):
        """Test _to_utc with already UTC datetime."""
        from datetime import datetime, timezone

        web = _import_web(monkeypatch)

        utc_dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = web._to_utc(utc_dt)

        assert result == utc_dt
