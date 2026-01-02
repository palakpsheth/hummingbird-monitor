"""
Tests for helper functions in ``hbmon.web``.

The tests here cover CSS class selection for species names and the
construction of hour heatmaps used in the web UI.  To avoid importing
the FastAPI app at module import time (which would attempt to create
directories in protected locations like ``/data``), we import
``hbmon.web`` within each test function after setting up safe
environment variables.  This ensures that the module sees the
appropriate settings and that any fallback directory creation happens
in a writable location.
"""

import os
import time
from datetime import datetime

import pytest



def test_species_to_css_variants(import_web, monkeypatch):
    """Verify that various species names map to the correct CSS classes."""
    web = import_web(monkeypatch)
    assert web.species_to_css("Anna's hummingbird") == "species-anna"
    assert web.species_to_css("Anna’s Hummingbird") == "species-anna"
    assert web.species_to_css("Allen's hummingbird") == "species-allens"
    assert web.species_to_css("rufous") == "species-rufous"
    assert web.species_to_css("Costa's") == "species-costas"
    assert web.species_to_css("black chinned") == "species-black-chinned"
    assert web.species_to_css("calliope hummingbird") == "species-calliope"
    assert web.species_to_css("broad billed hummingbird") == "species-broad-billed"
    # Unknown species fallback
    assert web.species_to_css("mystery bird") == "species-unknown"


def test_paginate_bounds(import_web, monkeypatch):
    """Pagination should clamp page and page_size and compute offsets safely."""
    web = import_web(monkeypatch)

    page, size, total_pages, offset = web.paginate(total_count=0, page=3, page_size=500, max_page_size=50)
    assert (page, size, total_pages, offset) == (1, 50, 1, 0)

    page, size, total_pages, offset = web.paginate(total_count=95, page=3, page_size=10)
    assert (page, size, total_pages, offset) == (3, 10, 10, 20)

    # page should clamp to last page when requesting beyond range
    page, size, total_pages, offset = web.paginate(total_count=15, page=5, page_size=4)
    assert (page, size, total_pages, offset) == (4, 4, 4, 12)


def test_build_hour_heatmap_levels(import_web, monkeypatch):
    """
    Construct a simple heatmap and verify level assignments based on counts.
    The heatmap function divides counts into 5 buckets based on the maximum.
    Level 0: count <= 0 or no data
    Level 1: 0 < frac <= 0.20
    Level 2: 0.20 < frac <= 0.40
    Level 3: 0.40 < frac <= 0.60
    Level 4: 0.60 < frac <= 0.80
    Level 5: 0.80 < frac <= 1.0
    """
    web = import_web(monkeypatch)
    # Provide counts for a few hours.  Max count is 10.
    hours = [(0, 0), (1, 2), (2, 3), (3, 5), (4, 8), (5, 10)]
    heat = web.build_hour_heatmap(hours)
    level_map = {d['hour']: d['level'] for d in heat}
    # hour 0 has count 0 -> level 0
    assert level_map[0] == 0
    # hour 1: 2/10 = 0.2 => level 1 (upper bound inclusive)
    assert level_map[1] == 1
    # hour 2: 3/10 = 0.3 => level 2
    assert level_map[2] == 2
    # hour 3: 5/10 = 0.5 => level 3
    assert level_map[3] == 3
    # hour 4: 8/10 = 0.8 => level 4 (upper bound inclusive)
    assert level_map[4] == 4
    # hour 5: 10/10 = 1.0 => level 5
    assert level_map[5] == 5
    # Hours not provided default to level 0
    for h in range(6, 24):
        assert level_map[h] == 0


def test_pretty_json(import_web, monkeypatch):
    web = import_web(monkeypatch)
    pretty = web.pretty_json('{"b":2,"a":1}')
    assert pretty is not None
    assert "\n" in pretty
    # Indent should add 4 spaces
    assert "    \"a\": 1" in pretty

    # Fallback to original on parse errors
    bad = web.pretty_json("{not-json}")
    assert bad == "{not-json}"


def test_timezone_helpers(import_web, monkeypatch):
    web = import_web(monkeypatch)
    assert web._normalize_timezone(None) == "local"
    assert web._normalize_timezone("") == "local"
    assert web._normalize_timezone(" local ") == "local"
    assert web._normalize_timezone("America/Denver") == "America/Denver"

    assert web._timezone_label(None) == "Browser local"
    assert web._timezone_label("LOCAL") == "Browser local"
    assert web._timezone_label("Europe/Paris") == "Europe/Paris"


def test_as_utc_str_treats_naive_as_utc(import_web, monkeypatch):
    if not hasattr(time, "tzset"):  # pragma: no cover - platform guard
        pytest.skip("tzset not available on this platform")

    web = import_web(monkeypatch)

    old_tz = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "US/Pacific")
    time.tzset()
    try:
        naive = datetime(2024, 1, 1, 12, 0, 0)
        assert web._as_utc_str(naive) == "2024-01-01T12:00:00Z"
    finally:
        if old_tz is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", old_tz)
        time.tzset()


def test_get_git_commit_without_git(import_web, monkeypatch, tmp_path):
    web = import_web(monkeypatch)
    monkeypatch.setattr(web, "_GIT_PATH", None)
    repo_root = tmp_path / "nogit"
    repo_root.mkdir()
    monkeypatch.setattr(web, "_REPO_ROOT", repo_root)
    assert web._get_git_commit() == "unknown"


def test_get_git_commit_from_head(import_web, monkeypatch, tmp_path):
    web = import_web(monkeypatch)
    repo_root = tmp_path / "repo"
    git_dir = repo_root / ".git" / "refs" / "heads"
    git_dir.mkdir(parents=True, exist_ok=True)
    (repo_root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "main").write_text("abc1234def5678\n")
    monkeypatch.setattr(web, "_GIT_PATH", None)
    monkeypatch.setattr(web, "_REPO_ROOT", repo_root)
    assert web._get_git_commit() == "abc1234"


def test_get_git_commit_from_gitdir_file(import_web, monkeypatch, tmp_path):
    web = import_web(monkeypatch)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    actual_git = tmp_path / "actual_git"
    ref_dir = actual_git / "refs" / "heads"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (actual_git / "HEAD").write_text("ref: refs/heads/main\n")
    (ref_dir / "main").write_text("deadbeefcafebabe\n")
    (repo_root / ".git").write_text(f"gitdir: {actual_git}\n")
    monkeypatch.setattr(web, "_GIT_PATH", None)
    monkeypatch.setattr(web, "_REPO_ROOT", repo_root)
    assert web._get_git_commit() == "deadbee"


def test_get_git_commit_detached_head(import_web, monkeypatch, tmp_path):
    web = import_web(monkeypatch)
    repo_root = tmp_path / "repo"
    git_dir = repo_root / ".git"
    git_dir.mkdir(parents=True, exist_ok=True)
    (git_dir / "HEAD").write_text("cafebabe1234567890\n")
    monkeypatch.setattr(web, "_GIT_PATH", None)
    monkeypatch.setattr(web, "_REPO_ROOT", repo_root)
    assert web._get_git_commit() == "cafebab"


def test_get_git_commit_from_packed_refs(import_web, monkeypatch, tmp_path):
    web = import_web(monkeypatch)
    repo_root = tmp_path / "repo"
    git_dir = repo_root / ".git"
    refs_dir = git_dir / "refs" / "heads"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "packed-refs").write_text("deadbeefcafebabe refs/heads/main\n")
    monkeypatch.setattr(web, "_GIT_PATH", None)
    monkeypatch.setattr(web, "_REPO_ROOT", repo_root)
    assert web._get_git_commit() == "deadbee"


def test_get_git_commit_from_env(import_web, monkeypatch, tmp_path):
    web = import_web(monkeypatch)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(web, "_GIT_PATH", None)
    monkeypatch.setattr(web, "_REPO_ROOT", repo_root)
    monkeypatch.setenv("HBMON_GIT_COMMIT", "envhash123")
    assert web._get_git_commit() == "envhash123"


def test_get_git_commit_env_unknown_falls_back_to_git(import_web, monkeypatch, tmp_path):
    """When HBMON_GIT_COMMIT=unknown, fallback to git metadata instead."""
    web = import_web(monkeypatch)
    repo_root = tmp_path / "repo"
    git_dir = repo_root / ".git" / "refs" / "heads"
    git_dir.mkdir(parents=True, exist_ok=True)
    (repo_root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "main").write_text("abc1234def5678\n")
    monkeypatch.setattr(web, "_GIT_PATH", None)
    monkeypatch.setattr(web, "_REPO_ROOT", repo_root)
    monkeypatch.setenv("HBMON_GIT_COMMIT", "unknown")
    assert web._get_git_commit() == "abc1234"


class MockObservation:
    """Mock Observation for testing get_annotated_snapshot_path."""

    def __init__(self, extra: dict | None = None):
        self._extra = extra

    def get_extra(self) -> dict | None:
        return self._extra


class MockObservationWithExtras:
    def __init__(self, extra: dict | None = None):
        self._extra = extra

    def get_extra(self) -> dict | None:
        return self._extra


def test_get_annotated_snapshot_path_with_valid_path(import_web, monkeypatch):
    """Test that get_annotated_snapshot_path returns path when present."""
    web = import_web(monkeypatch)
    obs = MockObservation(extra={"snapshots": {"annotated_path": "snapshots/2024-01-01/abc123_annotated.jpg"}})
    result = web.get_annotated_snapshot_path(obs)
    assert result == "snapshots/2024-01-01/abc123_annotated.jpg"


def test_get_annotated_snapshot_path_with_none_extra(import_web, monkeypatch):
    """Test that get_annotated_snapshot_path returns None when extra_json is None."""
    web = import_web(monkeypatch)
    obs = MockObservation(extra=None)
    result = web.get_annotated_snapshot_path(obs)
    assert result is None


def test_get_annotated_snapshot_path_with_empty_extra(import_web, monkeypatch):
    """Test that get_annotated_snapshot_path returns None when extra_json is empty dict."""
    web = import_web(monkeypatch)
    obs = MockObservation(extra={})
    result = web.get_annotated_snapshot_path(obs)
    assert result is None


def test_get_annotated_snapshot_path_with_missing_snapshots_key(import_web, monkeypatch):
    """Test that get_annotated_snapshot_path returns None when snapshots key is missing."""
    web = import_web(monkeypatch)
    obs = MockObservation(extra={"detection": {"box_confidence": 0.85}})
    result = web.get_annotated_snapshot_path(obs)
    assert result is None


def test_get_annotated_snapshot_path_with_snapshots_not_dict(import_web, monkeypatch):
    """Test that get_annotated_snapshot_path returns None when snapshots is not a dict."""
    web = import_web(monkeypatch)
    obs = MockObservation(extra={"snapshots": "invalid_string"})
    result = web.get_annotated_snapshot_path(obs)
    assert result is None


def test_get_annotated_snapshot_path_with_missing_annotated_path(import_web, monkeypatch):
    """Test that get_annotated_snapshot_path returns None when annotated_path key is missing."""
    web = import_web(monkeypatch)
    obs = MockObservation(extra={"snapshots": {"some_other_key": "value"}})
    result = web.get_annotated_snapshot_path(obs)
    assert result is None


def test_get_clip_snapshot_path_with_valid_path(import_web, monkeypatch):
    """Test that get_clip_snapshot_path returns path when present."""
    web = import_web(monkeypatch)
    obs = MockObservation(extra={"snapshots": {"clip_path": "snapshots/2024-01-01/abc123_clip.jpg"}})
    result = web.get_clip_snapshot_path(obs)
    assert result == "snapshots/2024-01-01/abc123_clip.jpg"


def test_get_clip_snapshot_path_with_none_extra(import_web, monkeypatch):
    """Test that get_clip_snapshot_path returns None when extra_json is None."""
    web = import_web(monkeypatch)
    obs = MockObservation(extra=None)
    result = web.get_clip_snapshot_path(obs)
    assert result is None


def test_get_clip_snapshot_path_with_snapshots_not_dict(import_web, monkeypatch):
    """Test that get_clip_snapshot_path returns None when snapshots is not a dict."""
    web = import_web(monkeypatch)
    obs = MockObservation(extra={"snapshots": ["nope"]})
    result = web.get_clip_snapshot_path(obs)
    assert result is None


def test_get_clip_snapshot_path_with_missing_snapshots_key(import_web, monkeypatch):
    """Test that get_clip_snapshot_path returns None when snapshots key is missing."""
    web = import_web(monkeypatch)
    obs = MockObservation(extra={"detection": {"box_confidence": 0.85}})
    result = web.get_clip_snapshot_path(obs)
    assert result is None


def test_get_clip_snapshot_path_with_missing_clip_path(import_web, monkeypatch):
    """Test that get_clip_snapshot_path returns None when clip_path key is missing."""
    web = import_web(monkeypatch)
    obs = MockObservation(extra={"snapshots": {"annotated_path": "snapshots/2024-01-01/abc123_annotated.jpg"}})
    result = web.get_clip_snapshot_path(obs)
    assert result is None


def test_flatten_extra_metadata(import_web, monkeypatch):
    web = import_web(monkeypatch)
    extra = {
        "detection": {"box_confidence": 0.8756, "extra": {"foo": "bar"}},
        "identification": {
            "individual_id": 7,
            "match_score": 0.9123,
            "species_label": "Anna's Hummingbird",
            "species_prob": 0.8231,
        },
        "review": {"label": "ok"},
        "flags": ["a", "b"],
        "score": 2,
    }
    flattened = web._flatten_extra_metadata(extra)
    assert flattened["detection.box_confidence"] == 0.8756
    assert flattened["detection.extra.foo"] == "bar"
    assert flattened["identification.individual_id"] == 7
    assert flattened["identification.match_score"] == 0.9123
    assert flattened["identification.species_label"] == "Anna's Hummingbird"
    assert flattened["identification.species_prob"] == 0.8231
    assert flattened["review.label"] == "ok"
    assert flattened["flags"] == ["a", "b"]
    assert flattened["score"] == 2


def test_prepare_observation_extras_formats_values(import_web, monkeypatch):
    web = import_web(monkeypatch)
    obs = [
        MockObservationWithExtras(
            extra={
                "detection": {"box_confidence": 0.8756},
                "snapshots": {"annotated_path": "snapshots/2024-01-01/abc123_annotated.jpg"},
                "review": {"label": "ok"},
                "flags": ["a", "b"],
            }
        ),
        MockObservationWithExtras(extra={"review": {"label": "bad"}, "score": 2}),
        MockObservationWithExtras(extra=None),
    ]

    columns, sort_types, labels = web._prepare_observation_extras(obs)

    assert columns[0] == "detection.box_confidence"
    assert "snapshots.annotated_path" not in columns
    assert sort_types["detection.box_confidence"] == "number"
    assert labels["review.label"] == "Review · Label"
    assert obs[0].extra_display["detection.box_confidence"] == "0.876"
    assert obs[0].extra_display["flags"] == '["a", "b"]'
    assert obs[1].extra_sort_values["score"] == "2"
    assert obs[2].extra_display["review.label"] == ""


def test_default_extra_column_visibility_hides_sensitivity(import_web, monkeypatch):
    web = import_web(monkeypatch)
    columns = [
        "detection.box_confidence",
        "sensitivity.bg_motion_threshold",
        "identification.match_score",
    ]

    defaults = web._default_extra_column_visibility(columns)

    assert defaults["detection.box_confidence"] is True
    assert defaults["sensitivity.bg_motion_threshold"] is False
    assert defaults["identification.match_score"] is True


def test_sanitize_redirect_path(import_web, monkeypatch):
    web = import_web(monkeypatch)
    # Basic cases
    assert web._sanitize_redirect_path(None) == "/observations"
    assert web._sanitize_redirect_path("") == "/observations"
    assert web._sanitize_redirect_path("nope") == "/observations"
    assert web._sanitize_redirect_path("/ok") == "/ok"
    assert web._sanitize_redirect_path("/ok", default="/default") == "/ok"

    # External URLs with schemes should be rejected
    assert web._sanitize_redirect_path("https://evil.com") == "/observations"
    assert web._sanitize_redirect_path("http://evil.com") == "/observations"
    assert web._sanitize_redirect_path("javascript:alert(1)") == "/observations"
    assert web._sanitize_redirect_path("data:text/html,<script>") == "/observations"

    # Malformed scheme URLs (colon before slash) should be rejected
    assert web._sanitize_redirect_path("https:/evil.com") == "/observations"
    assert web._sanitize_redirect_path("custom-scheme:foo") == "/observations"

    # Protocol-relative URLs should be rejected
    assert web._sanitize_redirect_path("//evil.com") == "/observations"
    assert web._sanitize_redirect_path("//evil.com/path") == "/observations"

    # Backslash normalization - backslashes converted to forward slashes
    assert web._sanitize_redirect_path("\\\\evil.com") == "/observations"
    assert web._sanitize_redirect_path("\\/evil.com") == "/observations"

    # Valid internal paths should be allowed
    assert web._sanitize_redirect_path("/observations") == "/observations"
    assert web._sanitize_redirect_path("/candidates") == "/candidates"
    assert web._sanitize_redirect_path("/individuals/123") == "/individuals/123"

    # Query strings and fragments should be preserved
    assert web._sanitize_redirect_path("/page?foo=bar") == "/page?foo=bar"
    assert web._sanitize_redirect_path("/page#section") == "/page#section"
    assert web._sanitize_redirect_path("/page?a=1#sec") == "/page?a=1#sec"

    # Custom default should be used when input is invalid
    assert web._sanitize_redirect_path("https://evil.com", default="/custom") == "/custom"
    assert web._sanitize_redirect_path("//evil.com", default="/custom") == "/custom"

    # Control characters (CR/LF) should be rejected to prevent response splitting
    assert web._sanitize_redirect_path("/path\r\nInjected: header") == "/observations"
    assert web._sanitize_redirect_path("/path\nInjected: header") == "/observations"


def test_safe_internal_url(import_web, monkeypatch):
    """Test _safe_internal_url helper for constructing safe redirect URLs."""
    web = import_web(monkeypatch)
    
    # Valid path without resource_id
    assert web._safe_internal_url("/observations") == "/observations"
    assert web._safe_internal_url("/candidates") == "/candidates"
    assert web._safe_internal_url("/individuals") == "/individuals"
    
    # Valid path with integer resource_id
    assert web._safe_internal_url("/observations", 123) == "/observations/123"
    assert web._safe_internal_url("/individuals", 456) == "/individuals/456"
    assert web._safe_internal_url("/candidates", 789) == "/candidates/789"
    
    # URL construction correctness with various ID values
    assert web._safe_internal_url("/observations", 1) == "/observations/1"
    assert web._safe_internal_url("/observations", 0) == "/observations/0"
    assert web._safe_internal_url("/observations", 999999) == "/observations/999999"
    
    # Path validation - path must start with "/"
    with pytest.raises(ValueError, match="Path must start with"):
        web._safe_internal_url("observations", 123)
    
    with pytest.raises(ValueError, match="Path must start with"):
        web._safe_internal_url("relative/path")
    
    with pytest.raises(ValueError, match="Path must start with"):
        web._safe_internal_url("")
    
    # resource_id validation - non-integer values should raise ValueError
    with pytest.raises(ValueError):
        web._safe_internal_url("/observations", "not_an_int")
    
    with pytest.raises(ValueError):
        web._safe_internal_url("/observations", "123; DROP TABLE")
    
    # Lists and dicts should raise TypeError (not ValueError) from int()
    with pytest.raises(TypeError):
        web._safe_internal_url("/observations", [123])
    
    with pytest.raises(TypeError):
        web._safe_internal_url("/observations", {"id": 123})
    
    # String numbers should be converted to integers (defensive)
    assert web._safe_internal_url("/observations", "456") == "/observations/456"
    
    # Float values should be truncated to integers
    assert web._safe_internal_url("/observations", 123.9) == "/observations/123"


def test_sanitize_case_name(import_web, monkeypatch):
    web = import_web(monkeypatch)
    assert web._sanitize_case_name(None, "fallback") == "fallback"
    assert web._sanitize_case_name("", "fallback") == "fallback"
    assert web._sanitize_case_name("  My Case!  ", "fallback") == "My-Case"
    assert web._sanitize_case_name("___", "fallback") == "fallback"


def test_format_extra_label_and_value(import_web, monkeypatch):
    web = import_web(monkeypatch)
    assert web._format_extra_label("sensitivity.detect_conf") == "Sensitivity · Detect Conf"
    assert web._format_extra_value(None) == ""
    assert web._format_extra_value(1.23456) == "1.235"
    assert web._format_extra_value(True) == "true"
    assert web._format_extra_value(False) == "false"
    assert web._format_extra_value(12) == "12"
    assert web._format_extra_value([1, "a"]) == "[1, \"a\"]"
    assert web._format_extra_value({"a": 1}) == "{\"a\": 1}"
    assert web._format_extra_value("ok") == "ok"


def test_extra_sort_helpers(import_web, monkeypatch):
    web = import_web(monkeypatch)
    assert web._extra_sort_type([]) == "text"
    assert web._extra_sort_type([None, 1, 2.5]) == "number"
    assert web._extra_sort_type([True, False]) == "text"
    assert web._extra_sort_type([1, "a"]) == "text"

    assert web._format_sort_value(None, "text") == ""
    assert web._format_sort_value(True, "number") == "1"
    assert web._format_sort_value(False, "number") == "0"
    assert web._format_sort_value(1.25, "number") == "1.25"
    assert web._format_sort_value("ABC", "text") == "abc"
    assert web._format_sort_value([1, "A"], "text") == "[1, \"a\"]"


def test_order_extra_columns(import_web, monkeypatch):
    web = import_web(monkeypatch)
    ordered = web._order_extra_columns(
        iter(["foo", "detection.box_confidence", "bar", "alpha"])
    )
    assert ordered[0] == "detection.box_confidence"
    assert ordered[1:] == ["alpha", "bar", "foo"]


def test_candidate_json_value_helpers(import_web, monkeypatch):
    web = import_web(monkeypatch)

    assert web._candidate_json_value("extra_json", [], "sqlite") is None

    sqlite_expr = web._candidate_json_value("extra_json", ["review", "label"], "sqlite")
    assert sqlite_expr is not None
    assert "json_extract" in str(sqlite_expr)

    postgres_expr = web._candidate_json_value("extra_json", ["reason"], "postgresql")
    assert postgres_expr is not None
    assert "extract_path_text" in str(postgres_expr)
