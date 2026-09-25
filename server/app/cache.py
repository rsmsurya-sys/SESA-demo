from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock
from typing import Any

from app.config import Settings


class TtlCache:
    def __init__(self) -> None:
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._data.get(key)
            if row is None:
                return None
            exp, val = row
            if exp < time.time():
                self._data.pop(key, None)
                return None
            return val

    def set(self, key: str, value: Any, ttl: int) -> None:
        with self._lock:
            self._data[key] = (time.time() + ttl, value)


class RateLimiter:
    """Sliding-window limiter (in-process). Swap for Redis INCR in production."""

    def __init__(self, limit: int, window_s: int = 60) -> None:
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def hit(self, key: str) -> tuple[bool, int]:
        now = time.time()
        with self._lock:
            q = self._hits[key]
            cutoff = now - self.window_s
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.limit:
                return False, 0
            q.append(now)
            return True, self.limit - len(q)


def redis_client(settings: Settings):
    if not settings.redis_url:
        return None
    try:
        import redis

        return redis.Redis.from_url(settings.redis_url, decode_responses=True)
    except Exception:
        return None
