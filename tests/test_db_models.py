# tests/test_db_models.py
from __future__ import annotations

import sqlite3
from pathlib import Path


def test_db_migration_creates_tables(sqlite_conn: sqlite3.Connection):
    """
    Spec: on init, DB should contain tables for:
    - observations
    - individuals
    - media (or clips)
    """
    from hbmon.db.migrate import migrate  # type: ignore

    migrate(sqlite_conn)

    cur = sqlite_conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = {row[0] for row in cur.fetchall()}

    # Adjust names to match your schema.
    assert "observations" in tables
    assert "individuals" in tables


def test_insert_and_fetch_observation(sqlite_conn: sqlite3.Connection, tmp_path: Path):
    from hbmon.db.migrate import migrate  # type: ignore
    from hbmon.db.repo import ObservationRepo  # type: ignore

    migrate(sqlite_conn)
    repo = ObservationRepo(sqlite_conn)

    obs_id = repo.insert_observation(
        timestamp="2025-12-20T12:34:56Z",
        species_label="anna_hummingbird",
        individual_id=1,
        clip_path=str(tmp_path / "clip.mp4"),
        confidence=0.91,
    )
    row = repo.get_observation(obs_id)

    assert row["species_label"] == "anna_hummingbird"
    assert row["individual_id"] == 1
    assert abs(float(row["confidence"]) - 0.91) < 1e-6
