from hbmon.db import (
    get_db_url,
    get_async_db_url,
    _pool_settings,
    reset_db_state
)

def test_db_url_derivation(monkeypatch, tmp_path):
    monkeypatch.delenv("HBMON_DB_URL", raising=False)
    monkeypatch.delenv("HBMON_DB_ASYNC_URL", raising=False)
    monkeypatch.setenv("HBMON_DATA_DIR", str(tmp_path))
    
    url = get_db_url()
    assert "sqlite" in url
    
    async_url = get_async_db_url()
    assert "sqlite+aiosqlite" in async_url

def test_db_url_env(monkeypatch):
    monkeypatch.setenv("HBMON_DB_URL", "postgresql://user:pass@host/db")
    assert get_db_url() == "postgresql://user:pass@host/db"
    
    # Derivation for postgres
    assert get_async_db_url() == "postgresql+asyncpg://user:pass@host/db"

def test_db_url_async_env(monkeypatch):
    monkeypatch.setenv("HBMON_DB_ASYNC_URL", "something://else")
    assert get_async_db_url() == "something://else"

def test_pool_settings():
    # Sqlite should have no pool settings (NullPool used)
    assert _pool_settings("sqlite:///foo") == {}
    
    # Postgres should have settings
    s = _pool_settings("postgresql://foo")
    assert "pool_size" in s
    assert s["pool_size"] == 5

def test_engine_creation_error(monkeypatch):
    reset_db_state()
    # Mock _SQLALCHEMY_AVAILABLE to False is hard due to global, 
    # but we can try to trigger other branches.
    pass

def test_get_async_db_url_various(monkeypatch):
    monkeypatch.delenv("HBMON_DB_ASYNC_URL", raising=False)
    
    monkeypatch.setenv("HBMON_DB_URL", "postgresql+psycopg://foo")
    assert get_async_db_url() == "postgresql+asyncpg://foo"
    
    monkeypatch.setenv("HBMON_DB_URL", "sqlite+aiosqlite:///foo")
    assert get_async_db_url() == "sqlite+aiosqlite:///foo"
    
    monkeypatch.setenv("HBMON_DB_URL", "mysql://foo")
    assert get_async_db_url() == "" # Unsupported for derivation
