"""
Tests for Redis cache helper behavior.
"""

from __future__ import annotations

import pytest

import hbmon.cache as cache


class _FailingRedis:
    async def get(self, key: str) -> str:
        raise cache.RedisError("boom")

    async def setex(self, key: str, ttl: int, payload: str) -> None:
        raise cache.RedisError("boom")


@pytest.mark.anyio
async def test_cache_helpers_ignore_redis_errors(monkeypatch) -> None:
    monkeypatch.setattr(cache, "get_redis_client", lambda: _FailingRedis())

    assert await cache.cache_get_json("hbmon:test") is None
    await cache.cache_set_json("hbmon:test", {"ok": True})


class _RecordingRedis:
    def __init__(self, value: str | None):
        self.value = value
        self.calls: list[tuple[str, int, str]] = []

    async def get(self, key: str) -> str | None:
        return self.value

    async def setex(self, key: str, ttl: int, payload: str) -> None:
        self.calls.append((key, ttl, payload))


@pytest.mark.anyio
async def test_cache_helpers_json_roundtrip(monkeypatch) -> None:
    client = _RecordingRedis('{"ok": true}')
    monkeypatch.setattr(cache, "get_redis_client", lambda: client)

    payload = await cache.cache_get_json("hbmon:ok")
    assert payload == {"ok": True}

    await cache.cache_set_json("hbmon:ok", {"hello": "world"}, ttl_seconds=2)
    assert client.calls == [("hbmon:ok", 2, '{"hello": "world"}')]


@pytest.mark.anyio
async def test_cache_helpers_invalid_json(monkeypatch) -> None:
    client = _RecordingRedis("not-json")
    monkeypatch.setattr(cache, "get_redis_client", lambda: client)

    payload = await cache.cache_get_json("hbmon:bad")
    assert payload is None


def test_get_redis_client_disabled(monkeypatch):
    monkeypatch.setattr(cache, "_REDIS_AVAILABLE", False)
    assert cache.get_redis_client() is None


def test_get_redis_client_creates_client(monkeypatch):
    monkeypatch.setenv("HBMON_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setattr(cache, "_REDIS_AVAILABLE", True)
    monkeypatch.setattr(cache, "_REDIS_CLIENT", None)
    dummy = object()
    monkeypatch.setattr(cache.redis, "from_url", lambda *args, **kwargs: dummy)

    assert cache.get_redis_client() is dummy


@pytest.mark.anyio
async def test_cache_helpers_empty_payload(monkeypatch) -> None:
    client = _RecordingRedis("")
    monkeypatch.setattr(cache, "get_redis_client", lambda: client)

    payload = await cache.cache_get_json("hbmon:empty")
    assert payload is None
