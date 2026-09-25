from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "SESA API"
    app_version: str = "0.1.0"
    model_version: str = "v2.3.1"
    debug: bool = False
    jwt_secret: str = "dev-only-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 12
    cors_origins: str = "*"
    rate_limit_per_minute: int = 100
    redis_url: str | None = None
    mongo_uri: str | None = None
    mongo_db: str = "sesa"
    max_image_bytes: int = 8 * 1024 * 1024
    max_video_bytes: int = 20 * 1024 * 1024
    cache_ttl_seconds: int = 300
    demo_athlete_password: str = "athlete-demo"
    demo_clinician_password: str = "clinician-demo"


@lru_cache
def get_settings() -> Settings:
    return Settings()
