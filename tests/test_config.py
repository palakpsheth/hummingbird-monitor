# tests/test_config.py
from __future__ import annotations

from pathlib import Path

import pytest


def test_env_parsing_smoke(monkeypatch, tmp_path: Path):
    """
    Expected behavior:
    - App reads env vars like HBMON_DB_PATH, HBMON_MEDIA_DIR, etc.
    - Falls back to reasonable defaults if missing.
    """
    monkeypatch.setenv("HBMON_DB_PATH", str(tmp_path / "hbmon.sqlite3"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.setenv("HBMON_PRETRIGGER_SECONDS", "2.5")

    # Adjust these imports to match your code.
    # Example expected API:
    #   from hbmon.config import Settings
    #   s = Settings.from_env()
    #
    # If you don’t have this yet, this test is the spec.
    from hbmon.config import Settings  # type: ignore

    s = Settings.from_env()
    assert str(s.db_path).endswith("hbmon.sqlite3")
    assert Path(s.media_dir).name == "media"
    assert abs(float(s.pretrigger_seconds) - 2.5) < 1e-6


def test_env_invalid_number(monkeypatch):
    monkeypatch.setenv("HBMON_PRETRIGGER_SECONDS", "not-a-number")

    from hbmon.config import Settings  # type: ignore

    with pytest.raises(ValueError):
        Settings.from_env()
