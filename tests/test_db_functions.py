"""
Tests for database module helper functions.

These tests cover the db module utility functions.
"""

from __future__ import annotations


import pytest

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
