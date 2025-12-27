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
