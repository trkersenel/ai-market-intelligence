"""Domain exception hierarchy and the HTTP translation layer.

Services and repositories raise domain exceptions that know nothing about HTTP.
A single set of handlers registered on the FastAPI app converts them into a
stable, documented error envelope, so transport concerns never leak into the
business layer and clients get one predictable error shape.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger

logger = get_logger(__name__)

#: Starlette renamed its 422 constant across versions; the numeric literal is
#: stable and keeps the module working on either.
HTTP_422_UNPROCESSABLE: int = 422


class AppError(Exception):
    """Base class for every expected, domain-level failure.

    Attributes:
        message: Human-readable description safe to return to the client.
        code: Stable machine-readable identifier for programmatic handling.
        status_code: HTTP status the transport layer should respond with.
        details: Optional structured context (field errors, identifiers, ...).
    """

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Build a domain error, defaulting to the subclass's code and status."""
        super().__init__(message)
        self.message = message
        self.code = code or type(self).code
        self.status_code = status_code or type(self).status_code
        self.details = details or {}


class NotFoundError(AppError):
    """A requested entity does not exist."""

    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class ConflictError(AppError):
    """The request conflicts with the current state of a resource."""

    status_code = status.HTTP_409_CONFLICT
    code = "conflict"


class ValidationError(AppError):
    """Input passed schema validation but violates a domain rule."""

    status_code = HTTP_422_UNPROCESSABLE
    code = "validation_error"


class AuthenticationError(AppError):
    """Credentials are missing, malformed or invalid."""

    status_code = status.HTTP_401_UNAUTHORIZED
    code = "authentication_error"


class AuthorizationError(AppError):
    """The caller is authenticated but not permitted to perform the action."""

    status_code = status.HTTP_403_FORBIDDEN
    code = "authorization_error"


class RateLimitError(AppError):
    """An upstream or internal rate limit was exceeded."""

    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "rate_limited"


class ExternalServiceError(AppError):
    """A third-party dependency failed or returned an unusable response."""

    status_code = status.HTTP_502_BAD_GATEWAY
    code = "external_service_error"


class ServiceUnavailableError(AppError):
    """A required internal dependency (database, index) is not available."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "service_unavailable"


def _envelope(
    *,
    code: str,
    message: str,
    status_code: int,
    request: Request,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """Build the canonical error response body."""
    body: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "request_id": getattr(request.state, "request_id", None),
        }
    }
    if details:
        body["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=body)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the platform-wide exception handlers to ``app``."""

    @app.exception_handler(AppError)
    async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        log = logger.bind(error_code=exc.code, status_code=exc.status_code)
        # 5xx indicates a defect or an unhealthy dependency: keep the traceback.
        if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
            log.error("domain_error", message=exc.message, exc_info=exc)
        else:
            log.info("domain_error", message=exc.message)
        return _envelope(
            code=exc.code,
            message=exc.message,
            status_code=exc.status_code,
            request=request,
            details=exc.details,
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_request_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _envelope(
            code="request_validation_error",
            message="Request payload failed validation.",
            status_code=HTTP_422_UNPROCESSABLE,
            request=request,
            details={"errors": exc.errors()},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _envelope(
            code="http_error",
            message=str(exc.detail),
            status_code=exc.status_code,
            request=request,
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception", exc_info=exc)
        return _envelope(
            code="internal_error",
            message="An unexpected error occurred.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            request=request,
        )
