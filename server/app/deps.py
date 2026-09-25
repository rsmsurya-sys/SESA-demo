from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth import decode_token
from app.cache import RateLimiter, TtlCache, redis_client
from app.config import Settings, get_settings
from app.db import Store, get_store
from app.exceptions import RateLimitError, UnauthorizedError

bearer = HTTPBearer(auto_error=False)
_limiter = RateLimiter(limit=100)
_cache = TtlCache()


def get_cache() -> TtlCache:
    return _cache


def get_limiter() -> RateLimiter:
    return _limiter


def get_redis(settings: Annotated[Settings, Depends(get_settings)]):
    return redis_client(settings)


class Principal:
    def __init__(self, sub: str, role: str) -> None:
        self.sub = sub
        self.role = role


def get_principal(
    settings: Annotated[Settings, Depends(get_settings)],
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> Principal:
    if creds is None or creds.scheme.lower() != "bearer":
        raise UnauthorizedError("Missing bearer token")
    payload = decode_token(settings, creds.credentials)
    sub = payload.get("sub")
    role = payload.get("role")
    if not sub or not role:
        raise UnauthorizedError("Token missing claims")
    return Principal(str(sub), str(role))


def enforce_rate_limit(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    principal: Annotated[Principal, Depends(get_principal)],
    limiter: Annotated[RateLimiter, Depends(get_limiter)],
    redis: Annotated[Any, Depends(get_redis)],
) -> Principal:
    key = principal.sub
    limit = settings.rate_limit_per_minute
    limiter.limit = limit
    if redis is not None:
        try:
            import time

            bucket = int(time.time() // 60)
            rkey = f"rl:{key}:{bucket}"
            n = int(redis.incr(rkey))
            if n == 1:
                redis.expire(rkey, 70)
            if n > limit:
                raise RateLimitError("Rate limit exceeded (100 requests/minute)")
            request.state.rate_remaining = max(0, limit - n)
            return principal
        except RateLimitError:
            raise
        except Exception:
            pass
    ok, remaining = limiter.hit(key)
    if not ok:
        raise RateLimitError("Rate limit exceeded (100 requests/minute)")
    request.state.rate_remaining = remaining
    return principal


AuthUser = Annotated[Principal, Depends(enforce_rate_limit)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
StoreDep = Annotated[Store, Depends(get_store)]
CacheDep = Annotated[TtlCache, Depends(get_cache)]
