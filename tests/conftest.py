"""
Pytest configuration for hbmon tests.

This conftest defines:
- Fixtures for setting up safe test directories
- Markers for unit and integration tests
- Test data paths for integration tests
- UI testing fixtures for Playwright and a live FastAPI server
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from pathlib import Path
import socket
import threading
import time
from typing import Any

import pytest
from hbmon import db
from hbmon.web import get_app
import uvicorn


# Path to the integration test data directory
INTEGRATION_TEST_DATA_DIR = Path(__file__).parent / "integration" / "test_data"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "ui: marks tests as UI tests (requires Playwright)")


def _playwright_browsers_installed() -> bool:
    if importlib.util.find_spec("playwright") is None:
        return False
    sync_playwright = importlib.import_module("playwright.sync_api").sync_playwright
    with sync_playwright() as p:
        return Path(p.chromium.executable_path).exists()


def pytest_collection_modifyitems(config, items):
    if _playwright_browsers_installed():
        return
    skip_marker = pytest.mark.skip(reason="Playwright browsers are not installed.")
    for item in items:
        if "tests/ui/" in item.nodeid:
            item.add_marker(skip_marker)


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
def isolate_test_dirs(request, tmp_path, monkeypatch):
    """
    Ensure tests do not create data/media directories in the repository root.
    
    Skipped for UI tests which use session-scoped ui_test_dirs instead.
    """
    # Skip for UI tests - they use session-scoped ui_test_dirs
    if "ui_page" in request.fixturenames or "live_server_url" in request.fixturenames:
        return
    
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


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float = 5.0, thread: threading.Thread | None = None) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError as e:
            last_error = e
            time.sleep(0.05)
    
    error_msg = f"Timed out waiting for server on {host}:{port}"
    if thread and not thread.is_alive():
        error_msg += " (server thread died)"
    if last_error:
        error_msg += f" (last error: {last_error})"
    raise RuntimeError(error_msg)


@pytest.fixture(scope="session")
def browser_type_launch_args() -> dict[str, bool]:
    """Force Playwright browsers to launch in headless mode for CI-friendly UI tests."""
    return {"headless": True}


@pytest.fixture
def ui_page(page: Any) -> Any:
    """Provide a Playwright page for UI tests."""
    return page


@pytest.fixture(scope="session")
def ui_test_dirs(tmp_path_factory):
    """
    Session-scoped temporary directories for UI tests.
    
    This ensures the live server uses consistent directories across all UI tests,
    avoiding database initialization issues with session-scoped fixtures.
    """
    import os
    tmp_base = tmp_path_factory.mktemp("ui_tests")
    data_dir = tmp_base / "data"
    media_dir = tmp_base / "media"
    data_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables for the session
    old_data_dir = os.environ.get("HBMON_DATA_DIR")
    old_media_dir = os.environ.get("HBMON_MEDIA_DIR")
    
    os.environ["HBMON_DATA_DIR"] = str(data_dir)
    os.environ["HBMON_MEDIA_DIR"] = str(media_dir)
    
    yield {"data_dir": data_dir, "media_dir": media_dir}
    
    # Restore original values
    if old_data_dir:
        os.environ["HBMON_DATA_DIR"] = old_data_dir
    else:
        os.environ.pop("HBMON_DATA_DIR", None)
    if old_media_dir:
        os.environ["HBMON_MEDIA_DIR"] = old_media_dir
    else:
        os.environ.pop("HBMON_MEDIA_DIR", None)


@pytest.fixture(scope="session")
def live_server_url(ui_test_dirs) -> str:
    """Start a live FastAPI server for UI tests and return its base URL."""
    host = "127.0.0.1"
    port = _get_free_port()
    config = uvicorn.Config(get_app(), host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_for_port(host, port, thread=thread)
    yield f"http://{host}:{port}"
    server.should_exit = True
    if thread.is_alive():
        thread.join(timeout=5)
        if thread.is_alive():
            logging.warning("Server thread did not exit cleanly within timeout")
