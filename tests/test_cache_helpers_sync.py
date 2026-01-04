
"""
Tests for synchronous Redis cache helper behavior.
"""
from __future__ import annotations

import hbmon.cache as cache

class _RecordingSyncRedis:
    def __init__(self, value: str | None):
        self.value = value
        self.calls: list[tuple[str, int, str]] = []

    def get(self, key: str) -> str | None:
        return self.value

    def setex(self, key: str, ttl: int, payload: str) -> None:
        self.calls.append((key, ttl, payload))

class _FailingSyncRedis:
    def get(self, key: str) -> str:
        raise cache.RedisError("boom")
    
    def setex(self, key: str, ttl: int, payload: str) -> None:
        raise cache.RedisError("boom")

def test_cache_get_json_sync_basic(monkeypatch) -> None:
    client = _RecordingSyncRedis('{"sync": true}')
    monkeypatch.setattr(cache, "get_redis_sync_client", lambda: client)
    
    payload = cache.cache_get_json_sync("hbmon:sync")
    assert payload == {"sync": True}

def test_cache_get_json_sync_missing(monkeypatch) -> None:
    client = _RecordingSyncRedis(None)
    monkeypatch.setattr(cache, "get_redis_sync_client", lambda: client)
    
    assert cache.cache_get_json_sync("hbmon:missing") is None

def test_cache_get_json_sync_invalid(monkeypatch) -> None:
    client = _RecordingSyncRedis("not-json")
    monkeypatch.setattr(cache, "get_redis_sync_client", lambda: client)
    
    # Should catch JSONDecodeError and return default (None)
    assert cache.cache_get_json_sync("hbmon:bad") is None

def test_cache_get_json_sync_error(monkeypatch) -> None:
    monkeypatch.setattr(cache, "get_redis_sync_client", lambda: _FailingSyncRedis())
    
    # SHould catch RedisError and return default (None)
    assert cache.cache_get_json_sync("hbmon:error") is None

def test_cache_set_json_sync_basic(monkeypatch) -> None:
    client = _RecordingSyncRedis(None)
    monkeypatch.setattr(cache, "get_redis_sync_client", lambda: client)
    
    success = cache.cache_set_json_sync("hbmon:key", {"val": 123}, ttl_seconds=10)
    assert success is True
    assert client.calls == [("hbmon:key", 10, '{"val":123}')]

def test_cache_set_json_sync_error(monkeypatch) -> None:
    monkeypatch.setattr(cache, "get_redis_sync_client", lambda: _FailingSyncRedis())
    
    success = cache.cache_set_json_sync("hbmon:key", {}, ttl_seconds=10)
    assert success is False
