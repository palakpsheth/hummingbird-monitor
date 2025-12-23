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

import importlib
import os
import time
from datetime import datetime

import pytest


def _import_web(monkeypatch):
    """Import ``hbmon.web`` after setting safe directories via monkeypatch.

    FastAPI and SQLAlchemy are not required for these helpers; however,
    ``hbmon.web`` attempts to create directories at import time.  We
    therefore set ``HBMON_DATA_DIR`` and ``HBMON_MEDIA_DIR`` to
    writable locations (the current working directory) before importing.
    """
    from pathlib import Path
    # Use current working directory names to avoid permission issues
    cwd = Path.cwd().resolve()
    monkeypatch.setenv("HBMON_DATA_DIR", str(cwd / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(cwd / "media"))
    # Remove module from sys.modules if already loaded to force re-import
    if 'hbmon.web' in importlib.sys.modules:
        importlib.sys.modules.pop('hbmon.web')
    web = importlib.import_module('hbmon.web')
    return web


def test_species_to_css_variants(monkeypatch):
    """Verify that various species names map to the correct CSS classes."""
    web = _import_web(monkeypatch)
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


def test_paginate_bounds(monkeypatch):
    """Pagination should clamp page and page_size and compute offsets safely."""
    web = _import_web(monkeypatch)

    page, size, total_pages, offset = web.paginate(total_count=0, page=3, page_size=500, max_page_size=50)
    assert (page, size, total_pages, offset) == (1, 50, 1, 0)

    page, size, total_pages, offset = web.paginate(total_count=95, page=3, page_size=10)
    assert (page, size, total_pages, offset) == (3, 10, 10, 20)

    # page should clamp to last page when requesting beyond range
    page, size, total_pages, offset = web.paginate(total_count=15, page=5, page_size=4)
    assert (page, size, total_pages, offset) == (4, 4, 4, 12)


def test_build_hour_heatmap_levels(monkeypatch):
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
    web = _import_web(monkeypatch)
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


def test_pretty_json(monkeypatch):
    web = _import_web(monkeypatch)
    pretty = web.pretty_json('{"b":2,"a":1}')
    assert pretty is not None
    assert "\n" in pretty
    # Indent should add 4 spaces
    assert "    \"a\": 1" in pretty

    # Fallback to original on parse errors
    bad = web.pretty_json("{not-json}")
    assert bad == "{not-json}"


def test_timezone_helpers(monkeypatch):
    web = _import_web(monkeypatch)
    assert web._normalize_timezone(None) == "local"
    assert web._normalize_timezone("") == "local"
    assert web._normalize_timezone(" local ") == "local"
    assert web._normalize_timezone("America/Denver") == "America/Denver"

    assert web._timezone_label(None) == "Browser local"
    assert web._timezone_label("LOCAL") == "Browser local"
    assert web._timezone_label("Europe/Paris") == "Europe/Paris"


def test_as_utc_str_treats_naive_as_utc(monkeypatch):
    if not hasattr(time, "tzset"):  # pragma: no cover - platform guard
        pytest.skip("tzset not available on this platform")

    web = _import_web(monkeypatch)

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


def test_get_git_commit_without_git(monkeypatch, tmp_path):
    web = _import_web(monkeypatch)
    monkeypatch.setattr(web, "_GIT_PATH", None)
    repo_root = tmp_path / "nogit"
    repo_root.mkdir()
    monkeypatch.setattr(web, "_REPO_ROOT", repo_root)
    assert web._get_git_commit() == "unknown"


def test_get_git_commit_from_head(monkeypatch, tmp_path):
    web = _import_web(monkeypatch)
    repo_root = tmp_path / "repo"
    git_dir = repo_root / ".git" / "refs" / "heads"
    git_dir.mkdir(parents=True, exist_ok=True)
    (repo_root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "main").write_text("abc1234def5678\n")
    monkeypatch.setattr(web, "_GIT_PATH", None)
    monkeypatch.setattr(web, "_REPO_ROOT", repo_root)
    assert web._get_git_commit() == "abc1234"


def test_get_git_commit_from_gitdir_file(monkeypatch, tmp_path):
    web = _import_web(monkeypatch)
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


def test_get_git_commit_detached_head(monkeypatch, tmp_path):
    web = _import_web(monkeypatch)
    repo_root = tmp_path / "repo"
    git_dir = repo_root / ".git"
    git_dir.mkdir(parents=True, exist_ok=True)
    (git_dir / "HEAD").write_text("cafebabe1234567890\n")
    monkeypatch.setattr(web, "_GIT_PATH", None)
    monkeypatch.setattr(web, "_REPO_ROOT", repo_root)
    assert web._get_git_commit() == "cafebab"


def test_get_git_commit_from_packed_refs(monkeypatch, tmp_path):
    web = _import_web(monkeypatch)
    repo_root = tmp_path / "repo"
    git_dir = repo_root / ".git"
    refs_dir = git_dir / "refs" / "heads"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "packed-refs").write_text("deadbeefcafebabe refs/heads/main\n")
    monkeypatch.setattr(web, "_GIT_PATH", None)
    monkeypatch.setattr(web, "_REPO_ROOT", repo_root)
    assert web._get_git_commit() == "deadbee"


def test_get_git_commit_from_env(monkeypatch, tmp_path):
    web = _import_web(monkeypatch)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(web, "_GIT_PATH", None)
    monkeypatch.setattr(web, "_REPO_ROOT", repo_root)
    monkeypatch.setenv("HBMON_GIT_COMMIT", "envhash123")
    assert web._get_git_commit() == "envhash123"
