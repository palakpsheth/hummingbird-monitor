
"""
Tests for database module helper functions.

These tests cover the db module utility functions.
"""

from __future__ import annotations


import pytest
from sqlalchemy import select

import hbmon.db as db


class TestGetDbUrl:
    """Tests for the get_db_url function."""

    def test_get_db_url_uses_env_override(self, monkeypatch, tmp_path):
        """Test that HBMON_DB_URL environment variable overrides default."""
        monkeypatch.setenv("HBMON_DB_URL", "sqlite:///test.db")
        monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path))

        url = db.get_db_url()
        assert url == "sqlite:///test.db"

    def test_get_db_url_default_sqlite(self, monkeypatch, tmp_path):
        """Test that default URL uses sqlite file in data dir."""
        monkeypatch.delenv("HBMON_DB_URL", raising=False)
        monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path))

        url = db.get_db_url()
        assert url.startswith("sqlite:////")
        assert "hbmon.sqlite" in url


class TestGetAsyncDbUrl:
    """Tests for the get_async_db_url function."""

    def test_get_async_db_url_prefers_env_override(self, monkeypatch):
        monkeypatch.setenv("HBMON_DB_ASYNC_URL", "postgresql+asyncpg://user:pass@host/db")
        monkeypatch.setenv("HBMON_DB_URL", "sqlite:///ignored.db")

        url = db.get_async_db_url()
        assert url == "postgresql+asyncpg://user:pass@host/db"

    def test_get_async_db_url_derives_from_sqlite(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HBMON_DB_ASYNC_URL", raising=False)
        monkeypatch.setenv("HBMON_DB_URL", f"sqlite:///{tmp_path/'db.sqlite'}")

        url = db.get_async_db_url()
        assert url.startswith("sqlite+aiosqlite:///")

    def test_get_async_db_url_derives_from_psycopg(self, monkeypatch):
        monkeypatch.delenv("HBMON_DB_ASYNC_URL", raising=False)
        monkeypatch.setenv("HBMON_DB_URL", "postgresql+psycopg://user:pass@host/db")

        url = db.get_async_db_url()
        assert url == "postgresql+asyncpg://user:pass@host/db"

    def test_get_async_db_url_derives_from_postgresql(self, monkeypatch):
        monkeypatch.delenv("HBMON_DB_ASYNC_URL", raising=False)
        monkeypatch.setenv("HBMON_DB_URL", "postgresql://user:pass@host/db")

        url = db.get_async_db_url()
        assert url == "postgresql+asyncpg://user:pass@host/db"

    def test_get_async_db_url_passthrough(self, monkeypatch):
        monkeypatch.delenv("HBMON_DB_ASYNC_URL", raising=False)
        monkeypatch.setenv("HBMON_DB_URL", "postgresql+asyncpg://user:pass@host/db")

        url = db.get_async_db_url()
        assert url == "postgresql+asyncpg://user:pass@host/db"

    def test_get_async_db_url_returns_empty_when_unsupported(self, monkeypatch):
        monkeypatch.delenv("HBMON_DB_ASYNC_URL", raising=False)
        monkeypatch.setenv("HBMON_DB_URL", "mysql://user:pass@host/db")

        url = db.get_async_db_url()
        assert url == ""


def test_pool_settings_for_non_sqlite(monkeypatch):
    monkeypatch.setenv("HBMON_DB_POOL_SIZE", "7")
    monkeypatch.setenv("HBMON_DB_MAX_OVERFLOW", "3")
    monkeypatch.setenv("HBMON_DB_POOL_TIMEOUT", "11")
    monkeypatch.setenv("HBMON_DB_POOL_RECYCLE", "123")
    settings = db._pool_settings("postgresql://user:pass@host/db")
    assert settings["pool_size"] == 7
    assert settings["max_overflow"] == 3
    assert settings["pool_timeout"] == 11
    assert settings["pool_recycle"] == 123


def test_session_scope_rolls_back_on_error(monkeypatch, tmp_path):
    monkeypatch.setenv("HBMON_DB_URL", f"sqlite:///{tmp_path/'db.sqlite'}")
    db.reset_db_state()
    db.init_db()

    from hbmon.models import Individual

    with pytest.raises(ValueError, match="boom"):
        with db.session_scope() as session:
            session.add(Individual(name="rollback"))
            raise ValueError("boom")

    with db.session_scope() as session:
        assert session.query(Individual).count() == 0


@pytest.mark.asyncio
async def test_async_session_scope_rolls_back_on_error(monkeypatch, tmp_path):
    db_path = tmp_path / "db.sqlite"
    monkeypatch.setenv("HBMON_DB_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("HBMON_DB_ASYNC_URL", f"sqlite+aiosqlite:///{db_path}")
    db.reset_db_state()
    await db.init_async_db()

    from hbmon.models import Individual

    with pytest.raises(ValueError, match="boom"):
        async with db.async_session_scope() as session:
            session.add(Individual(name="rollback-async"))
            raise ValueError("boom")

    async with db.async_session_scope() as session:
        result = await session.execute(select(Individual))
        assert result.scalars().all() == []


def test_get_async_engine_requires_url(monkeypatch):
    monkeypatch.delenv("HBMON_DB_ASYNC_URL", raising=False)
    monkeypatch.setenv("HBMON_DB_URL", "mysql://user:pass@host/db")
    db.reset_db_state()

    with pytest.raises(RuntimeError, match="Async DB URL is not configured"):
        db.get_async_engine()

@pytest.mark.asyncio
async def test_get_async_engine_reuses_instance(monkeypatch, tmp_path):
    db_path = tmp_path / "db.sqlite"
    monkeypatch.setenv("HBMON_DB_ASYNC_URL", f"sqlite+aiosqlite:///{db_path}")
    db.reset_db_state()
    first = db.get_async_engine()
    second = db.get_async_engine()
    assert first is second

    # Explicitly dispose to avoid "Event loop is closed" errors from aiosqlite
    # when the test loop shuts down.
    await db.dispose_async_engine()


def test_get_async_engine_unavailable(monkeypatch):
    monkeypatch.setattr(db, "_ASYNC_SQLALCHEMY_AVAILABLE", False)
    monkeypatch.setattr(db, "_SQLALCHEMY_AVAILABLE", True)
    monkeypatch.setattr(db, "_ASYNC_ENGINE", None)

    with pytest.raises(RuntimeError, match="async engine is not available"):
        db.get_async_engine()


def test_reset_db_state_handles_async_dispose_errors(monkeypatch):
    class DummyEngine:
        def dispose(self) -> None:
            return None

    class DummySyncEngine:
        def dispose(self) -> None:
            raise RuntimeError("boom")

    class DummyAsyncEngine:
        def __init__(self):
            self.sync_engine = DummySyncEngine()

    monkeypatch.setattr(db, "_ENGINE", DummyEngine())
    monkeypatch.setattr(db, "_ASYNC_ENGINE", DummyAsyncEngine())
    monkeypatch.setattr(db, "_SessionLocal", object())
    monkeypatch.setattr(db, "_AsyncSessionLocal", object())

    db.reset_db_state()
    assert db._ENGINE is None
    assert db._ASYNC_ENGINE is None


def test_get_session_factory_initializes_engine(monkeypatch, tmp_path):
    monkeypatch.setenv("HBMON_DB_URL", f"sqlite:///{tmp_path/'db.sqlite'}")
    db.reset_db_state()
    factory = db.get_session_factory()
    assert callable(factory)


def test_get_async_session_factory_unavailable(monkeypatch):
    monkeypatch.setattr(db, "_ASYNC_SQLALCHEMY_AVAILABLE", False)
    monkeypatch.setattr(db, "_SQLALCHEMY_AVAILABLE", True)

    with pytest.raises(RuntimeError, match="async engine is not available"):
        db.get_async_session_factory()


@pytest.mark.asyncio
async def test_init_async_db_unavailable(monkeypatch):
    monkeypatch.setattr(db, "_ASYNC_SQLALCHEMY_AVAILABLE", False)
    monkeypatch.setattr(db, "_SQLALCHEMY_AVAILABLE", True)

    with pytest.raises(RuntimeError, match="async engine is not available"):
        await db.init_async_db()


class TestSqlalchemyAvailable:
    """Tests for SQLAlchemy availability detection."""

    def test_sqlalchemy_available_flag(self):
        """Test that _SQLALCHEMY_AVAILABLE flag is defined."""
        assert hasattr(db, "_SQLALCHEMY_AVAILABLE")
        assert isinstance(db._SQLALCHEMY_AVAILABLE, bool)


class TestGetEngine:
    """Tests for the get_engine function."""

    def test_get_engine_without_sqlalchemy(self, monkeypatch):
        """Test that get_engine raises if SQLAlchemy is unavailable."""
        monkeypatch.setattr(db, "_SQLALCHEMY_AVAILABLE", False)
        # Reset cached engine
        monkeypatch.setattr(db, "_ENGINE", None)

        with pytest.raises(RuntimeError, match="SQLAlchemy"):
            db.get_engine()


class TestSessionScope:
    """Tests for the session_scope context manager."""

    def test_session_scope_without_sqlalchemy(self, monkeypatch):
        """Test that session_scope raises if SQLAlchemy is unavailable."""
        monkeypatch.setattr(db, "_SQLALCHEMY_AVAILABLE", False)

        with pytest.raises(RuntimeError, match="SQLAlchemy"):
            with db.session_scope() as _session:
                pass


class TestGetDb:
    """Tests for the get_db FastAPI dependency."""

    def test_get_db_without_sqlalchemy(self, monkeypatch):
        """Test that get_db raises if SQLAlchemy is unavailable."""
        monkeypatch.setattr(db, "_SQLALCHEMY_AVAILABLE", False)

        with pytest.raises(RuntimeError, match="SQLAlchemy"):
            gen = db.get_db()
            next(gen)


class TestInitDb:
    """Tests for the init_db function."""

    def test_init_db_without_sqlalchemy(self, monkeypatch):
        """Test that init_db raises if SQLAlchemy is unavailable."""
        monkeypatch.setattr(db, "_SQLALCHEMY_AVAILABLE", False)

        with pytest.raises(RuntimeError, match="SQLAlchemy"):
            db.init_db()


class TestGetSessionFactory:
    """Tests for the get_session_factory function."""

    def test_get_session_factory_without_sqlalchemy(self, monkeypatch):
        """Test that get_session_factory raises if SQLAlchemy is unavailable."""
        monkeypatch.setattr(db, "_SQLALCHEMY_AVAILABLE", False)

        with pytest.raises(RuntimeError, match="SQLAlchemy"):
            db.get_session_factory()
