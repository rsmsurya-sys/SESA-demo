from __future__ import annotations


class SesaError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class BadRequestError(SesaError):
    status_code = 400
    code = "bad_request"


class InvalidImageError(BadRequestError):
    code = "invalid_image"


class InvalidVideoError(BadRequestError):
    code = "invalid_video"


class UnauthorizedError(SesaError):
    status_code = 401
    code = "unauthorized"


class ForbiddenError(SesaError):
    status_code = 403
    code = "forbidden"


class NotFoundError(SesaError):
    status_code = 404
    code = "not_found"


class RateLimitError(SesaError):
    status_code = 429
    code = "rate_limited"


class InferenceError(SesaError):
    status_code = 500
    code = "inference_failed"
