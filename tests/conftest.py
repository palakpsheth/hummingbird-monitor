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

    Note: When tmp_path is not provided, this fixture reuses the directories
    already set by the isolate_test_dirs autouse fixture to avoid conflicts.

    Usage:
        def test_something(import_web, monkeypatch):
            web = import_web(monkeypatch)
            # use web module (reuses isolate_test_dirs directories)

        def test_with_db(import_web, monkeypatch, tmp_path):
            web = import_web(monkeypatch, tmp_path=tmp_path, with_db=True)
            # use web module with DB configured
    """
    def _import_web(monkeypatch, tmp_path=None, with_db=False):
        """Import hbmon.web after setting safe directories via monkeypatch.

        Args:
            monkeypatch: pytest monkeypatch fixture
            tmp_path: optional pytest tmp_path fixture for isolated directories.
                      When None, reuses directories from isolate_test_dirs.
            with_db: if True, also configure DB URLs. Requires tmp_path to be provided.

        Returns:
            The imported hbmon.web module

        Raises:
            ValueError: if with_db=True but tmp_path is None
        """
        if with_db and tmp_path is None:
            raise ValueError("tmp_path must be provided when with_db=True")

        if tmp_path is not None:
            # Set up isolated directories with optional DB configuration
            data_dir = tmp_path / "data"
            media_dir = tmp_path / "media"
            monkeypatch.setenv("HBMON_DATA_DIR", str(data_dir))
            monkeypatch.setenv("HBMON_MEDIA_DIR", str(media_dir))

            if with_db:
                db_path = tmp_path / "db.sqlite"
                monkeypatch.setenv("HBMON_DB_URL", f"sqlite:///{db_path}")
                monkeypatch.setenv("HBMON_DB_ASYNC_URL", f"sqlite+aiosqlite:///{db_path}")
        # else: When tmp_path is None, rely on isolate_test_dirs autouse fixture
        # which already sets HBMON_DATA_DIR and HBMON_MEDIA_DIR from its own tmp_path

        # Remove module from sys.modules if already loaded to force re-import
        if "hbmon.web" in importlib.sys.modules:
            importlib.sys.modules.pop("hbmon.web")

        return importlib.import_module("hbmon.web")

    return _import_web
