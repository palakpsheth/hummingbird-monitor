# src/hbmon/db.py
"""
SQLAlchemy database helpers for hbmon.

- Uses SQLite by default, stored at /data/hbmon.sqlite (docker volume).
- Uses SQLAlchemy 2.x patterns (SessionLocal, Engine, create_all).
- Safe for use with FastAPI dependency injection.

Environment:
- HBMON_DB_URL: override the database URL (sync engine)
  default: sqlite:////data/hbmon.sqlite
- HBMON_DB_ASYNC_URL: override the async database URL (web app)
  default: derived from HBMON_DB_URL when possible (sqlite+aiosqlite or postgresql+asyncpg)
- HBMON_DB_POOL_SIZE: SQLAlchemy pool size for non-SQLite engines (default: 5)
- HBMON_DB_MAX_OVERFLOW: SQLAlchemy pool overflow for non-SQLite engines (default: 10)
- HBMON_DB_POOL_TIMEOUT: SQLAlchemy pool timeout seconds (default: 30)
- HBMON_DB_POOL_RECYCLE: SQLAlchemy pool recycle seconds (default: 1800)
- HBMON_SQLITE_BUSY_TIMEOUT_MS: PRAGMA busy_timeout value in ms for SQLite (default: 5000)
"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
import importlib.util
from typing import Iterator, Optional, Callable, Any, AsyncIterator

"""
Database helpers for hbmon.

This module provides functions to initialize and interact with a SQLAlchemy
database.  To ensure that the rest of the package can be imported without
SQLAlchemy installed, all imports of SQLAlchemy are wrapped in try/except
blocks.  When SQLAlchemy is unavailable, attempting to call any of the
database helper functions will raise a ``RuntimeError`` with a descriptive
message.  The ``_SQLALCHEMY_AVAILABLE`` flag may be inspected to determine if
the database layer is usable.
"""

try:
    from sqlalchemy import create_engine, event  # type: ignore
    from sqlalchemy.engine import Engine  # type: ignore
    from sqlalchemy.orm import Session, sessionmaker  # type: ignore
    from sqlalchemy.pool import NullPool  # type: ignore
    _SQLALCHEMY_AVAILABLE = True
except Exception:  # pragma: no cover
    # SQLAlchemy is not available.  Create stub types and mark flag.
    create_engine = None  # type: ignore
    Engine = object  # type: ignore
    Session = object  # type: ignore
    sessionmaker = None  # type: ignore
    NullPool = object  # type: ignore
    _SQLALCHEMY_AVAILABLE = False

from hbmon.config import db_path, env_int, env_str, ensure_dirs

if _SQLALCHEMY_AVAILABLE and importlib.util.find_spec("sqlalchemy.ext.asyncio"):
    from sqlalchemy.ext.asyncio import (  # type: ignore
        AsyncEngine,
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )
    _ASYNC_SQLALCHEMY_AVAILABLE = True
else:  # pragma: no cover - optional dependency
    AsyncEngine = object  # type: ignore
    AsyncSession = object  # type: ignore
    async_sessionmaker = None  # type: ignore
    create_async_engine = None  # type: ignore
    _ASYNC_SQLALCHEMY_AVAILABLE = False


# Session factory is initialized lazily to allow env overrides before first use.
_ENGINE: Optional[Engine] = None
_SessionLocal: Optional[Callable[..., Session]] = None  # sessionmaker type
_ASYNC_ENGINE: Optional[AsyncEngine] = None
_AsyncSessionLocal: Optional[Callable[..., AsyncSession]] = None


def get_db_url() -> str:
    """
    Choose DB URL from env, else default sqlite file in HBMON_DATA_DIR.
    """
    url = env_str("HBMON_DB_URL", "")
    if url:
        return url

    # Default sqlite file under /data
    ensure_dirs()
    p = db_path()
    # sqlite absolute path must be 4 slashes: sqlite:////abs/path
    return f"sqlite:////{p.as_posix().lstrip('/')}"

def get_async_db_url() -> str:
    """
    Choose async DB URL from env, else derive from HBMON_DB_URL when possible.
    """
    url = env_str("HBMON_DB_ASYNC_URL", "")
    if url:
        return url

    sync_url = get_db_url()
    if sync_url.startswith("sqlite:"):
        return sync_url.replace("sqlite:", "sqlite+aiosqlite:", 1)
    if sync_url.startswith("postgresql+psycopg"):
        return sync_url.replace("postgresql+psycopg", "postgresql+asyncpg", 1)
    if sync_url.startswith("postgresql://"):
        return sync_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if "+asyncpg" in sync_url or "+aiosqlite" in sync_url:
        return sync_url
    return ""


def _pool_settings(url: str) -> dict[str, Any]:
    if url.startswith("sqlite:"):
        return {}
    return {
        "pool_size": env_int("HBMON_DB_POOL_SIZE", 5),
        "max_overflow": env_int("HBMON_DB_MAX_OVERFLOW", 10),
        "pool_timeout": env_int("HBMON_DB_POOL_TIMEOUT", 30),
        "pool_recycle": env_int("HBMON_DB_POOL_RECYCLE", 1800),
    }


def get_engine() -> Engine:
    global _ENGINE, _SessionLocal
    if not _SQLALCHEMY_AVAILABLE:
        raise RuntimeError(
            "SQLAlchemy is not installed; database functions are unavailable."
        )

    if _ENGINE is not None:
        return _ENGINE  # type: ignore[return-value]

    url = get_db_url()

    busy_timeout_ms = env_int("HBMON_SQLITE_BUSY_TIMEOUT_MS", 5000)
    connect_args: dict[str, Any] = {}
    engine_kwargs = _pool_settings(url)
    if url.startswith("sqlite:"):
        # Required for SQLite + threads (FastAPI/uvicorn)
        connect_args = {"check_same_thread": False}
        engine_kwargs["poolclass"] = NullPool

    assert create_engine is not None  # for mypy
    _ENGINE = create_engine(
        url,
        future=True,
        pool_pre_ping=True,
        connect_args=connect_args,
        **engine_kwargs,
    )  # type: ignore[assignment]

    if url.startswith("sqlite:"):
        def _set_sqlite_pragmas(dbapi_connection: Any, connection_record: Any) -> None:  # type: ignore[override]
            try:
                cursor = dbapi_connection.cursor()
                # Use % formatting instead of f-string to avoid CodeQL SQL injection warning
                # busy_timeout_ms is validated as int via env_int() so this is safe
                cursor.execute("PRAGMA busy_timeout=%d" % int(busy_timeout_ms))
                cursor.close()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[db] failed to set PRAGMA busy_timeout: {exc}")

        event.listen(_ENGINE, "connect", _set_sqlite_pragmas)

    assert sessionmaker is not None  # for mypy
    _SessionLocal = sessionmaker(
        bind=_ENGINE,
        autoflush=False,
        autocommit=False,
        future=True,
    )
    return _ENGINE  # type: ignore[return-value]

def get_async_engine() -> AsyncEngine:
    global _ASYNC_ENGINE, _AsyncSessionLocal
    if not (_SQLALCHEMY_AVAILABLE and _ASYNC_SQLALCHEMY_AVAILABLE):
        raise RuntimeError(
            "SQLAlchemy async engine is not available; database functions are unavailable."
        )
    if _ASYNC_ENGINE is not None:
        return _ASYNC_ENGINE  # type: ignore[return-value]

    url = get_async_db_url()
    if not url:
        raise RuntimeError("Async DB URL is not configured.")

    busy_timeout_ms = env_int("HBMON_SQLITE_BUSY_TIMEOUT_MS", 5000)
    connect_args: dict[str, Any] = {}
    engine_kwargs = _pool_settings(url)
    if url.startswith("sqlite:"):
        connect_args = {"timeout": float(busy_timeout_ms) / 1000.0}
        engine_kwargs["poolclass"] = NullPool

    assert create_async_engine is not None
    _ASYNC_ENGINE = create_async_engine(
        url,
        future=True,
        pool_pre_ping=True,
        connect_args=connect_args,
        **engine_kwargs,
    )  # type: ignore[assignment]

    if url.startswith("sqlite:"):
        def _set_sqlite_pragmas(dbapi_connection: Any, connection_record: Any) -> None:  # type: ignore[override]
            try:
                cursor = dbapi_connection.cursor()
                # Use % formatting instead of f-string to avoid CodeQL SQL injection warning
                # busy_timeout_ms is validated as int via env_int() so this is safe
                cursor.execute("PRAGMA busy_timeout=%d" % int(busy_timeout_ms))
                cursor.close()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[db] failed to set async PRAGMA busy_timeout: {exc}")

        event.listen(_ASYNC_ENGINE.sync_engine, "connect", _set_sqlite_pragmas)

    assert async_sessionmaker is not None
    _AsyncSessionLocal = async_sessionmaker(
        bind=_ASYNC_ENGINE,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    return _ASYNC_ENGINE  # type: ignore[return-value]


def get_session_factory() -> Callable[..., Session]:
    global _SessionLocal
    if not _SQLALCHEMY_AVAILABLE:
        raise RuntimeError(
            "SQLAlchemy is not installed; database functions are unavailable."
        )
    if _SessionLocal is None:
        get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


def get_async_session_factory() -> Callable[..., AsyncSession]:
    global _AsyncSessionLocal
    if not (_SQLALCHEMY_AVAILABLE and _ASYNC_SQLALCHEMY_AVAILABLE):
        raise RuntimeError(
            "SQLAlchemy async engine is not available; database functions are unavailable."
        )
    if _AsyncSessionLocal is None:
        get_async_engine()
    assert _AsyncSessionLocal is not None
    return _AsyncSessionLocal


def is_async_db_available() -> bool:
    return _SQLALCHEMY_AVAILABLE and _ASYNC_SQLALCHEMY_AVAILABLE


def reset_db_state() -> None:
    """
    Reset cached SQLAlchemy engines and session factories.

    This is primarily intended for tests that override environment variables
    between runs and need a fresh engine/session setup.
    """
    global _ENGINE, _SessionLocal, _ASYNC_ENGINE, _AsyncSessionLocal
    if _ENGINE is not None:
        _ENGINE.dispose()
    if _ASYNC_ENGINE is not None:
        try:
            _ASYNC_ENGINE.sync_engine.dispose()
        except Exception:
            pass
    _ENGINE = None
    _SessionLocal = None
    _ASYNC_ENGINE = None
    _AsyncSessionLocal = None


async def dispose_async_engine() -> None:
    """
    Dispose the async engine if it is initialized.

    This is primarily used by the web app lifespan to ensure aiosqlite
    background threads are stopped before the event loop is closed.
    """
    global _ASYNC_ENGINE, _AsyncSessionLocal
    if _ASYNC_ENGINE is None:
        return
    try:
        await _ASYNC_ENGINE.dispose()
    except Exception:
        try:
            _ASYNC_ENGINE.sync_engine.dispose()
        except Exception:
            pass
    _ASYNC_ENGINE = None
    _AsyncSessionLocal = None


@contextmanager
def session_scope() -> Iterator[Session]:
    """
    Context manager for short-lived DB transactions.
    Commits on success, rolls back on exception.

    Raises:
        RuntimeError: if SQLAlchemy is not available.
    """
    if not _SQLALCHEMY_AVAILABLE:
        raise RuntimeError(
            "SQLAlchemy is not installed; cannot create a database session."
        )
    SessionLocal = get_session_factory()
    db = SessionLocal()  # type: ignore[call-arg]
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

@asynccontextmanager
async def async_session_scope() -> AsyncIterator[AsyncSession]:
    """
    Async context manager for short-lived DB transactions.
    Commits on success, rolls back on exception.
    """
    if not (_SQLALCHEMY_AVAILABLE and _ASYNC_SQLALCHEMY_AVAILABLE):
        raise RuntimeError(
            "SQLAlchemy async engine is not available; cannot create a database session."
        )
    SessionLocal = get_async_session_factory()
    db = SessionLocal()  # type: ignore[call-arg]
    try:
        yield db
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()


def init_db() -> None:
    """
    Create tables if they do not exist.

    IMPORTANT: models must be imported so SQLAlchemy knows them.
    """
    if not _SQLALCHEMY_AVAILABLE:
        raise RuntimeError(
            "SQLAlchemy is not installed; cannot initialize the database."
        )
    # Import side effect registers models with Base metadata.
    from hbmon import models  # noqa: F401  # type: ignore

    engine = get_engine()

    # models.Base must exist
    from hbmon.models import Base  # type: ignore

    Base.metadata.create_all(bind=engine)
    
    # Run schema migrations for existing tables
    _run_migrations(engine)


def _run_migrations(engine: Engine) -> None:
    """
    Run schema migrations to add missing columns to existing tables.
    
    SQLAlchemy's create_all only creates missing tables, not missing columns.
    This function adds columns that were defined after the table was created.
    """
    from sqlalchemy import text, inspect
    
    migrations = [
        # (table_name, column_name, column_definition)
        ("annotation_boxes", "confidence", "REAL"),
    ]
    
    with engine.connect() as conn:
        inspector = inspect(engine)
        
        for table, column, col_def in migrations:
            # Check if table exists
            if table not in inspector.get_table_names():
                continue
            
            # Check if column exists
            existing_cols = [c["name"] for c in inspector.get_columns(table)]
            if column in existing_cols:
                continue
            
            # Add the column
            try:
                # PostgreSQL and SQLite compatible syntax
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}"))
                conn.commit()
                print(f"[db] Migration: Added column {table}.{column}")
            except Exception as e:
                print(f"[db] Migration failed for {table}.{column}: {e}")


async def init_async_db() -> None:
    """
    Create tables if they do not exist (async engine).
    """
    if not (_SQLALCHEMY_AVAILABLE and _ASYNC_SQLALCHEMY_AVAILABLE):
        raise RuntimeError(
            "SQLAlchemy async engine is not available; cannot initialize the database."
        )
    from hbmon import models  # noqa: F401  # type: ignore
    from hbmon.models import Base  # type: ignore

    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# FastAPI dependency
def get_db() -> Iterator[Session]:
    """
    FastAPI dependency: yields a Session.
    """
    if not _SQLALCHEMY_AVAILABLE:
        raise RuntimeError(
            "SQLAlchemy is not installed; cannot provide a database session."
        )
    SessionLocal = get_session_factory()
    db = SessionLocal()  # type: ignore[call-arg]
    try:
        yield db
    finally:
        db.close()


# FastAPI dependency (async)
async def get_async_db() -> AsyncIterator[AsyncSession]:
    """
    FastAPI dependency: yields an AsyncSession.
    """
    if not (_SQLALCHEMY_AVAILABLE and _ASYNC_SQLALCHEMY_AVAILABLE):
        raise RuntimeError(
            "SQLAlchemy async engine is not available; cannot provide a database session."
        )
    SessionLocal = get_async_session_factory()
    db = SessionLocal()  # type: ignore[call-arg]
    try:
        yield db
    finally:
        await db.close()

# Alias for compatibility with annotation jobs
get_sync_session = session_scope
