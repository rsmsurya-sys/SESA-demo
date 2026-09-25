import asyncio
import os

os.environ["JWT_SECRET"] = "test-secret"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000"

import httpx
import pytest

from app.config import get_settings
from app.db import reset_store
from app.main import create_app

TINY_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class SyncASGIClient:
    """httpx 0.28+ ASGITransport is async-only; wrap for pytest."""

    def __init__(self, app) -> None:
        self._app = app

    def request(self, method: str, url: str, **kwargs):
        async def _go():
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                return await ac.request(method, url, **kwargs)

        return asyncio.run(_go())

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)

    def close(self) -> None:
        return None


@pytest.fixture
def client():
    get_settings.cache_clear()
    reset_store()
    return SyncASGIClient(create_app())


def _token(client, user: str = "athlete", password: str = "athlete-demo") -> str:
    r = client.post("/api/v1/auth/token", json={"username": user, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _auth(client, user: str = "athlete", password: str = "athlete-demo"):
    return {"Authorization": f"Bearer {_token(client, user, password)}"}
