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
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from hbmon.config import db_path, env_str, ensure_dirs


# Session factory is initialized lazily to allow env overrides before first use.
_ENGINE: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


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
    if _ENGINE is not None:
        return _ENGINE

    url = get_db_url()

    connect_args = {}
    if url.startswith("sqlite:"):
        # Required for SQLite + threads (FastAPI/uvicorn)
        connect_args = {"check_same_thread": False}

    _ENGINE = create_engine(
        url,
        future=True,
        pool_pre_ping=True,
        connect_args=connect_args,
    )

    _SessionLocal = sessionmaker(
        bind=_ENGINE,
        autoflush=False,
        autocommit=False,
        future=True,
    )
    return _ENGINE


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    """
    Context manager for short-lived DB transactions.
    Commits on success, rolls back on exception.
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
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
    # Import side effect registers models with Base metadata.
    from hbmon import models  # noqa: F401

    engine = get_engine()

    # models.Base must exist
    from hbmon.models import Base  # type: ignore

    Base.metadata.create_all(bind=engine)


# FastAPI dependency
def get_db() -> Iterator[Session]:
    """
    FastAPI dependency: yields a Session.
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
