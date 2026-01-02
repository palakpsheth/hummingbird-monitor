"""
Additional tests for helper functions in ``hbmon.web``.

These tests focus on internal helpers that do not require running the
FastAPI application but do depend on safe import-time configuration.
"""

import importlib

import pytest


def _import_web(monkeypatch):
    """Import ``hbmon.web`` after setting safe directories via monkeypatch."""
    from pathlib import Path

    cwd = Path.cwd().resolve()
    monkeypatch.setenv("HBMON_DATA_DIR", str(cwd / "data"))
    monkeypatch.setenv("HBMON_MEDIA_DIR", str(cwd / "media"))
    if "hbmon.web" in importlib.sys.modules:
        importlib.sys.modules.pop("hbmon.web")
    return importlib.import_module("hbmon.web")


def test_read_git_head_from_packed_refs(monkeypatch, tmp_path):
    web = _import_web(monkeypatch)
    repo_root = tmp_path / "repo"
    git_dir = repo_root / ".git"
    git_dir.mkdir(parents=True, exist_ok=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled\n"
        "deadbeefcafebabe1234567890abcdef12345678 refs/heads/main\n"
        "^baddcafe1234567890abcdef1234567890abcdef\n"
    )

    assert web._read_git_head(repo_root) == "deadbee"


def test_timezone_helpers_extra(monkeypatch):
    web = _import_web(monkeypatch)
    assert web._normalize_timezone("  ") == "local"
    assert web._normalize_timezone("America/Los_Angeles") == "America/Los_Angeles"
    assert web._timezone_label("local") == "Browser local"
    assert web._timezone_label("Europe/Berlin") == "Europe/Berlin"


def test_candidate_json_value_paths(monkeypatch):
    web = _import_web(monkeypatch)
    if not getattr(web, "_SQLA_AVAILABLE", False):
        pytest.skip("SQLAlchemy not available")

    from sqlalchemy import column
    from sqlalchemy.dialects import postgresql, sqlite

    expr = column("extra_json")

    sqlite_expr = web._candidate_json_value(expr, ["outer", "inner"], "sqlite")
    assert sqlite_expr is not None
    sqlite_compiled = sqlite_expr.compile(dialect=sqlite.dialect())
    assert list(sqlite_compiled.params.values()) == ["$.outer.inner"]

    pg_expr = web._candidate_json_value(expr, ["outer", "inner"], "postgresql")
    assert pg_expr is not None
    pg_compiled = pg_expr.compile(dialect=postgresql.dialect())
    assert "extract_path_text" in pg_compiled.string
    assert list(pg_compiled.params.values()) == ["outer", "inner"]

    assert web._candidate_json_value(expr, [], "sqlite") is None
    assert web._candidate_json_value(expr, ["x"], "mysql") is None


def test_get_db_dialect_name_variants(monkeypatch):
    web = _import_web(monkeypatch)

    class Dialect:
        def __init__(self, name):
            self.name = name

    class Bind:
        def __init__(self, name):
            self.dialect = Dialect(name)

    class SessionWithGetBind:
        def get_bind(self):
            return Bind("sqlite")

    class SessionGetBindRaises:
        def __init__(self):
            self.bind = Bind("sqlite")

        def get_bind(self):
            raise RuntimeError("boom")

    class SyncEngine:
        def __init__(self, name):
            self.dialect = Dialect(name)

    class BindWithSync:
        def __init__(self, name):
            self.sync_engine = SyncEngine(name)

    class SessionWithSyncEngine:
        def __init__(self):
            self.bind = BindWithSync("postgresql")

    class SessionWithoutBind:
        pass

    assert web._get_db_dialect_name(SessionWithGetBind()) == "sqlite"
    assert web._get_db_dialect_name(SessionGetBindRaises()) == "sqlite"
    assert web._get_db_dialect_name(SessionWithSyncEngine()) == "postgresql"
    assert web._get_db_dialect_name(SessionWithoutBind()) == ""
