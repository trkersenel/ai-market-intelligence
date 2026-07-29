"""HTTP middleware: request correlation and access logging.

Correlation is established once, at the edge, and propagated through
``structlog`` context variables. Every downstream log line -- in a service, a
repository, or an OpenAI call -- inherits the request id without any explicit
plumbing, which is what makes distributed debugging tractable.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from app.core.config import ObservabilitySettings
from app.core.logging import get_logger

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
RESPONSE_TIME_HEADER = "X-Response-Time-ms"

_CallNext = Callable[[Request], Awaitable[Response]]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Bind a correlation id and emit one structured access log per request."""

    def __init__(self, app: ASGIApp, *, settings: ObservabilitySettings) -> None:
        """Wrap ``app``, caching the paths excluded from access logging."""
        super().__init__(app)
        self._excluded_paths = frozenset(settings.excluded_access_paths)

    async def dispatch(self, request: Request, call_next: _CallNext) -> Response:
        """Process a single request, adding correlation headers and logging."""
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # The exception handlers own the response body; this branch only
            # guarantees the failure is timed and logged with its correlation id.
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.exception("request_failed", duration_ms=round(elapsed_ms, 2))
            raise
        finally:
            structlog.contextvars.unbind_contextvars("method", "path")

        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[RESPONSE_TIME_HEADER] = f"{elapsed_ms:.2f}"

        if request.url.path not in self._excluded_paths:
            logger.info(
                "request_completed",
                status_code=response.status_code,
                duration_ms=round(elapsed_ms, 2),
            )
        return response
