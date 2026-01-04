from __future__ import annotations

"""
Redis cache helpers for hbmon.

The cache layer is optional. When Redis is unavailable or not configured, all
functions return ``None``/no-op to keep the rest of the app running.

Environment:
- HBMON_REDIS_URL: Redis connection URL (e.g., redis://hbmon-redis:6379/0)
- HBMON_REDIS_TTL_SECONDS: default cache TTL (seconds, default: 5)
"""

import importlib.util
import json
from typing import Any

from hbmon.config import env_int, env_str

_REDIS_AVAILABLE = importlib.util.find_spec("redis") is not None
if _REDIS_AVAILABLE:
    import redis.asyncio as redis  # type: ignore
    from redis.exceptions import RedisError  # type: ignore
else:  # pragma: no cover - optional dependency
    redis = None  # type: ignore
    RedisError = Exception  # type: ignore

_REDIS_CLIENT: "redis.Redis | None" = None
_REDIS_SYNC_CLIENT: "Any | None" = None


def get_cache_ttl() -> int:
    return env_int("HBMON_REDIS_TTL_SECONDS", 5)


def get_redis_url() -> str:
    return env_str("HBMON_REDIS_URL", "")


def get_redis_client() -> "redis.Redis | None":
    global _REDIS_CLIENT
    if not _REDIS_AVAILABLE:
        return None
    url = get_redis_url()
    if not url:
        return None
    if _REDIS_CLIENT is None:
        _REDIS_CLIENT = redis.from_url(url, decode_responses=True)
    return _REDIS_CLIENT


def get_redis_sync_client() -> "Any | None":
    """Get synchronous Redis client for use in non-async contexts (threads)."""
    global _REDIS_SYNC_CLIENT
    if not _REDIS_AVAILABLE:
        return None
    url = get_redis_url()
    if not url:
        return None
    if _REDIS_SYNC_CLIENT is None:
        # Import synchronous Redis client
        import redis as redis_sync
        _REDIS_SYNC_CLIENT = redis_sync.from_url(url, decode_responses=True)
    return _REDIS_SYNC_CLIENT


async def cache_get_json(key: str) -> Any | None:
    client = get_redis_client()
    if client is None:
        return None
    try:
        payload = await client.get(key)
        if not payload:
            return None
    except RedisError:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


async def cache_set_json(key: str, value: Any, ttl_seconds: int | None = None) -> None:
    client = get_redis_client()
    if client is None:
        return None
    ttl = int(ttl_seconds or get_cache_ttl())
    payload = json.dumps(value, default=str)
    try:
        await client.setex(key, ttl, payload)
    except RedisError:
        return None


def cache_set_json_sync(key: str, value: Any, ttl_seconds: int | None = None) -> bool:
    """
    Synchronous version for use in threads (annotator monitoring).
    Returns True if successful, False otherwise.
    """
    client = get_redis_sync_client()
    if not client:
        return False
    
    ttl = int(ttl_seconds or get_cache_ttl())
    try:
        data = json.dumps(value, separators=(",", ":"))
        client.setex(key, ttl, data)
        return True
    except (RedisError, Exception):
        return False


def cache_get_json_sync(key: str, default: Any = None) -> Any:
    """
    Synchronous get for use in threads or non-async contexts.
    """
    client = get_redis_sync_client()
    if not client:
        return default
    
    try:
        payload = client.get(key)
        if not payload:
            return default
        return json.loads(payload)
    except (RedisError, json.JSONDecodeError, Exception):
        return default
