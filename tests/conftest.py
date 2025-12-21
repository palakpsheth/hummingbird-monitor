# tests/conftest.py
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
from PIL import Image


@dataclass(frozen=True)
class DummySettings:
    db_path: Path
    media_dir: Path
    pretrigger_seconds: float = 2.0
    posttrigger_seconds: float = 4.0
    clip_fps: int = 15


@pytest.fixture()
def tmp_media_dir(tmp_path: Path) -> Path:
    d = tmp_path / "media"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture()
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "hbmon.sqlite3"


@pytest.fixture()
def settings(tmp_db_path: Path, tmp_media_dir: Path) -> DummySettings:
    return DummySettings(db_path=tmp_db_path, media_dir=tmp_media_dir)


@pytest.fixture()
def sample_rgb_image() -> Image.Image:
    # A small, deterministic RGB image.
    w, h = 320, 240
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = 80
    arr[..., 1] = 120
    arr[..., 2] = 200
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture()
def sqlite_conn(tmp_db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(tmp_db_path))
    try:
        yield conn
    finally:
        conn.close()
