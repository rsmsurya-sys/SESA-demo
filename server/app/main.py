
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.exceptions import SesaError
from app.logging_conf import configure_logging
from app.routers import auth_router, router
from app.util import redact

log = logging.getLogger("sesa")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    configure_logging(settings.debug)
    log.info("startup", extra={"event": "startup", "version": settings.app_version})
    yield
    log.info("shutdown", extra={"event": "shutdown"})


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Smart Emergency Sports App API",
        version=settings.app_version,
        description=(
            "Triage assistance API (not a medical device). "
            "OpenAPI/Swagger at `/docs`. JWT via `POST /api/v1/auth/token` "
            "(demo users: `athlete` / `clinician`)."
        ),
        lifespan=lifespan,
        contact={"name": "SESA"},
    )
    # Build allowed origins list.
    # allow_credentials=True + allow_origins=["*"] is illegal per the CORS spec and
    # causes browsers to block responses. JWT is in the Authorization header (not a
    # cookie), so allow_credentials=False is correct for this API.
    configured = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    demo_origins = [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",  # Vite
    ]
    if configured == ["*"]:
        # Wildcard mode — still need explicit list so credentials can stay False.
        final_origins: list[str] | str = "*"
    else:
        final_origins = list(dict.fromkeys(configured + demo_origins))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=final_origins,
        allow_credentials=False,  # JWT via Authorization header, not cookies
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-RateLimit-Remaining"],
    )

    @app.exception_handler(SesaError)
    async def sesa_handler(_: Request, exc: SesaError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": "error",
                "error": {"code": exc.code, "message": exc.message, "details": exc.details},
            },
        )

    @app.exception_handler(RequestValidationError)
    async def valid_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "error": {
                    "code": "validation_error",
                    "message": "Invalid request",
                    "details": {"errors": redact(exc.errors())},
                },
            },
        )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        response = await call_next(request)
        log.info(
            "request",
            extra={
                "event": "http",
                "path": request.url.path,
                "request_id": request.headers.get("x-request-id"),
            },
        )
        remaining = getattr(request.state, "rate_remaining", None)
        if remaining is not None:
            response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    @app.get("/health", tags=["meta"])
    def health():
        return {"status": "ok", "version": settings.app_version}

    @app.get("/", include_in_schema=False)
    def root_redirect():
        """Redirect root → demo web app."""
        return RedirectResponse(url="/app/")

    # Mount the web demo as static files at /app.
    # Resolves relative to this file: server/app/main.py → ../../web
    _web_dir = Path(__file__).resolve().parents[2] / "web"
    if _web_dir.is_dir():
        app.mount("/app", StaticFiles(directory=str(_web_dir), html=True), name="web")
        log.info("web_demo_mounted", extra={"event": "web_demo", "path": str(_web_dir)})

    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()