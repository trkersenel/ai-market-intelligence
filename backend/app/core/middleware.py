"""HTTP middleware: request correlation and access logging.

Correlation is established once, at the edge, and propagated through
``structlog`` context variables. Every downstream log line -- in a service, a
repository, or an OpenAI call -- inherits the request id without any explicit
plumbing, which is what makes distributed debugging tractable.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import ObservabilitySettings
from app.core.logging import get_logger
from app.core.metrics import MetricsRegistry

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


class MetricsMiddleware:
    """Records request metrics, as pure ASGI rather than ``BaseHTTPMiddleware``.

    The distinction is load-bearing here. ``BaseHTTPMiddleware`` runs the
    downstream app in a separate task and does not surface the scope mutations
    the router makes, so the matched route is invisible to it -- the first
    version of this recorded every request under a relative path like "/live"
    and lost the mount prefix entirely.

    A pure-ASGI middleware holds the same ``scope`` dict the router writes
    ``route`` and ``root_path`` into, so the template is readable after the call.
    It also avoids the per-request task ``BaseHTTPMiddleware`` spawns.
    """

    def __init__(self, app: ASGIApp, *, metrics: MetricsRegistry) -> None:
        """Wrap ``app``, recording into ``metrics``."""
        self._app = app
        self._metrics = metrics
        # Built lazily on the first request: middleware is constructed before
        # every router has been included, so building it here would miss routes.
        self._templates: dict[int, str] | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Time the request and record its outcome."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        if self._templates is None:
            self._templates = build_route_templates(scope.get("app"))

        started = time.perf_counter()
        status = 500

        async def capture(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            await self._app(scope, receive, capture)
        finally:
            # In the `finally` so a request that raises is still counted; an
            # endpoint that only fails would otherwise be invisible in metrics.
            self._metrics.observe_request(
                method=str(scope.get("method", "GET")),
                route=_route_template(scope, self._templates or {}),
                status=status,
                duration_seconds=time.perf_counter() - started,
            )


def build_route_templates(app: Any) -> dict[int, str]:
    """Map each route object to its fully-prefixed path template.

    Needed because this FastAPI version keeps an included router as a wrapper
    rather than flattening its routes into the parent. At request time
    ``scope["route"]`` is therefore the *inner* route, whose ``path`` is relative
    to its mount -- "/live", not "/health/live" -- and neither ``root_path`` nor
    the route itself carries the prefix.

    Walking the tree once at startup and keying by object identity avoids both
    re-implementing routing and depending on scope internals that have already
    changed shape once.
    """
    templates: dict[int, str] = {}

    def walk(routes: Iterable[Any], prefix: str) -> None:
        for route in routes:
            context = getattr(route, "include_context", None)
            if context is not None:
                inner = getattr(context, "included_router", None)
                walk(getattr(inner, "routes", ()), prefix + str(getattr(context, "prefix", "")))
                continue
            path = getattr(route, "path_format", None)
            if path is None:
                path = getattr(route, "path", None)
            # `is not None`, not truthiness: a collection endpoint is declared as
            # `@router.get("")` under a prefix, so its own path is the empty
            # string. Testing for truth skipped every such route and recorded it
            # as "unmatched" -- with a 200 status, which is how it was spotted.
            if path is not None:
                templates[id(route)] = f"{prefix}{path}"

    walk(getattr(app, "routes", ()), "")
    return templates


def _route_template(scope: Scope, templates: dict[int, str]) -> str:
    """Return the matched route pattern rather than the concrete path.

    ``/api/v1/prices/{symbol}`` is one time series; the raw path would be one
    per ticker, and on any endpoint taking a free-form identifier that is
    unbounded label cardinality -- the standard way an instrumented service
    takes down its own metrics backend.

    Unmatched requests collapse to a single bucket for the same reason: a 404
    scanner probing random URLs must not be able to create series at will.
    """
    route = scope.get("route")
    if route is None:
        return "unmatched"

    resolved = templates.get(id(route))
    if resolved:
        return resolved

    # Fallback for a route registered after startup: better a relative template
    # than an unbounded one.
    path = getattr(route, "path_format", None) or getattr(route, "path", None)
    return str(path) if path else "unmatched"
