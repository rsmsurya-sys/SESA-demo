from __future__ import annotations

from datetime import timedelta
from typing import Any

from jose import JWTError, jwt

from app.config import Settings
from app.exceptions import UnauthorizedError
from app.util import utcnow


DEMO_USERS = {
    "athlete": {"password_env": "demo_athlete_password", "role": "athlete", "sub": "ath_001"},
    "clinician": {"password_env": "demo_clinician_password", "role": "clinician", "sub": "doc_001"},
}


def create_access_token(settings: Settings, *, sub: str, role: str) -> str:
    exp = utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": sub, "role": role, "exp": exp}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(settings: Settings, token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise UnauthorizedError("Invalid or expired token") from exc


def authenticate_demo(settings: Settings, username: str, password: str) -> tuple[str, str]:
    spec = DEMO_USERS.get(username)
    if spec is None:
        raise UnauthorizedError("Unknown user")
    expected = getattr(settings, spec["password_env"])
    if password != expected:
        raise UnauthorizedError("Invalid credentials")
    return spec["sub"], spec["role"]
