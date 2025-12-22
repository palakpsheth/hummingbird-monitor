# src/hbmon/db.py
"""
SQLAlchemy database helpers for hbmon.

- Uses SQLite by default, stored at /data/hbmon.sqlite (docker volume).
- Uses SQLAlchemy 2.x patterns (SessionLocal, Engine, create_all).
- Safe for use with FastAPI dependency injection.

Environment:
- HBMON_DB_URL: override the database URL
  default: sqlite:////data/hbmon.sqlite
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional, Callable

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
    _SQLALCHEMY_AVAILABLE = True
except Exception:  # pragma: no cover
    # SQLAlchemy is not available.  Create stub types and mark flag.
    create_engine = None  # type: ignore
    Engine = object  # type: ignore
    Session = object  # type: ignore
    sessionmaker = None  # type: ignore
    _SQLALCHEMY_AVAILABLE = False

from hbmon.config import db_path, env_str, ensure_dirs


# Session factory is initialized lazily to allow env overrides before first use.
_ENGINE: Optional[Engine] = None
_SessionLocal: Optional[Callable[..., Session]] = None  # sessionmaker type


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
    connect_args = {}
    if url.startswith("sqlite:"):
        # Required for SQLite + threads (FastAPI/uvicorn)
        connect_args = {
            "check_same_thread": False,
            "timeout": max(1.0, float(busy_timeout_ms) / 1000.0),
        }

    assert create_engine is not None  # for mypy
    _ENGINE = create_engine(
        url,
        future=True,
        pool_pre_ping=True,
        connect_args=connect_args,
    )  # type: ignore[assignment]

    if url.startswith("sqlite:"):
        def _set_sqlite_pragmas(dbapi_connection, connection_record):  # type: ignore[override]
            try:
                cursor = dbapi_connection.cursor()
                cursor.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
                cursor.close()
            except Exception:
                pass

        event.listen(_ENGINE, "connect", _set_sqlite_pragmas)

    assert sessionmaker is not None  # for mypy
    _SessionLocal = sessionmaker(
        bind=_ENGINE,
        autoflush=False,
        autocommit=False,
        future=True,
    )
    return _ENGINE  # type: ignore[return-value]


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
