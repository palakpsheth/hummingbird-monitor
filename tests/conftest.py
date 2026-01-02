"""
Pytest configuration for hbmon tests.

This conftest defines:
- Fixtures for setting up safe test directories
- Markers for unit and integration tests
- Test data paths for integration tests
- Shared fixture for importing hbmon.web safely
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from hbmon import db


# Path to the integration test data directory
INTEGRATION_TEST_DATA_DIR = Path(__file__).parent / "integration" / "test_data"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


@pytest.fixture
def test_data_dir() -> Path:
    """Return the path to the integration test data directory."""
    return INTEGRATION_TEST_DATA_DIR


@pytest.fixture
def safe_dirs(tmp_path, monkeypatch):
    """
    Set up safe data and media directories for tests.

    This fixture ensures tests don't write to /data or /media by
    redirecting to temporary directories.
    """
    data_dir = tmp_path / "data"
    media_dir = tmp_path / "media"
    data_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HBMON_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(media_dir))

    return {"data_dir": data_dir, "media_dir": media_dir}


@pytest.fixture(autouse=True)
def isolate_test_dirs(tmp_path, monkeypatch):
    """
    Ensure tests do not create data/media directories in the repository root.
    """
    data_dir = tmp_path / "data"
    media_dir = tmp_path / "media"
    data_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HBMON_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(media_dir))


@pytest.fixture(autouse=True)
def cleanup_db_engines():
    """
    Reset DB engines after each test to prevent leaking async loops/connections.
    """
    yield
    db.reset_db_state()


@pytest.fixture
def import_web():
    """
    Factory fixture for importing hbmon.web safely with test-appropriate environment.

    Returns a callable that accepts monkeypatch and optional tmp_path to set up
    safe directories before importing hbmon.web. This avoids permission issues
    and ensures tests don't write to /data or /media.

    Usage:
        def test_something(import_web, monkeypatch):
            web = import_web(monkeypatch)
            # use web module

        def test_with_db(import_web, monkeypatch, tmp_path):
            web = import_web(monkeypatch, tmp_path=tmp_path, with_db=True)
            # use web module with DB configured
    """
    def _import_web(monkeypatch, tmp_path=None, with_db=False):
        """Import hbmon.web after setting safe directories via monkeypatch.

        Args:
            monkeypatch: pytest monkeypatch fixture
            tmp_path: optional pytest tmp_path fixture for isolated directories
            with_db: if True, also configure DB URLs (requires tmp_path)

        Returns:
            The imported hbmon.web module
        """
        if tmp_path is not None:
            data_dir = tmp_path / "data"
            media_dir = tmp_path / "media"
            if with_db:
                db_path = tmp_path / "db.sqlite"
                monkeypatch.setenv("HBMON_DB_URL", f"sqlite:///{db_path}")
                monkeypatch.setenv("HBMON_DB_ASYNC_URL", f"sqlite+aiosqlite:///{db_path}")
        else:
            # Use current working directory names to avoid permission issues
            cwd = Path.cwd().resolve()
            data_dir = cwd / "data"
            media_dir = cwd / "media"

        monkeypatch.setenv("HBMON_DATA_DIR", str(data_dir))
        monkeypatch.setenv("HBMON_MEDIA_DIR", str(media_dir))

        # Remove module from sys.modules if already loaded to force re-import
        if "hbmon.web" in importlib.sys.modules:
            importlib.sys.modules.pop("hbmon.web")

        return importlib.import_module("hbmon.web")

    return _import_web
