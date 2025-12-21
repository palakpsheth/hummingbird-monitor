# tests/test_api.py
from __future__ import annotations

import sqlite3

import pytest


@pytest.mark.parametrize("path", ["/health", "/api/health"])
def test_health_endpoint(path):
    """
    Spec: web app exposes a health endpoint for uptime checks.
    """
    from fastapi.testclient import TestClient
    from hbmon.web.app import app  # type: ignore

    client = TestClient(app)
    r = client.get(path)
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") in {"ok", "healthy"}


def test_list_observations(sqlite_conn: sqlite3.Connection, monkeypatch):
    """
    Spec: GET /api/observations returns most recent observations.
    Uses injected DB conn/session in app dependency for testability.
    """
    from fastapi.testclient import TestClient

    from hbmon.db.migrate import migrate  # type: ignore
    from hbmon.db.repo import ObservationRepo  # type: ignore
    from hbmon.web.app import app  # type: ignore

    migrate(sqlite_conn)
    repo = ObservationRepo(sqlite_conn)
    repo.insert_observation(
        timestamp="2025-12-20T12:00:00Z",
        species_label="unknown",
        individual_id=1,
        clip_path="media/clip1.mp4",
        confidence=0.5,
    )

    # Example: app has dependency `get_db()` we can override.
    # Adjust to match your app’s structure.
    if hasattr(app, "dependency_overrides"):
        from hbmon.web.deps import get_db  # type: ignore

        app.dependency_overrides[get_db] = lambda: sqlite_conn

    client = TestClient(app)
    r = client.get("/api/observations?limit=10")
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload, list)
    assert len(payload) >= 1
    assert payload[0]["species_label"] == "unknown"
