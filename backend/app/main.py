"""Application entrypoint and composition root.

Exposes an application *factory* rather than a module-level singleton. The
factory takes its settings as an argument, so tests, the CLI and the scheduler
can each build an app with different configuration in the same process -- the
practical payoff of not reading globals at import time.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints import health as health_endpoints
from app.api.v1.endpoints import metrics as metrics_endpoints
from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.metrics import MetricsRegistry
from app.core.middleware import MetricsMiddleware, RequestContextMiddleware
from app.db.mongo import MongoDatabase
from app.db.mongo_indexes import create_indexes
from app.db.postgres import PostgresDatabase
from app.marketdata.cache import ResponseCache
from app.marketdata.registry import build_registry
from app.marketdata.service import MarketDataService

logger = get_logger(__name__)


def _build_lifespan(
    settings: Settings,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Create the lifespan context manager bound to ``settings``.

    Long-lived resources (connection pools, HTTP clients, the scheduler) are
    created once on startup and released on shutdown. Binding them to
    ``app.state`` -- instead of module globals -- is what keeps multiple app
    instances isolated inside one interpreter.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "application_starting",
            environment=settings.environment.value,
            version=settings.version,
        )
        app.state.postgres = PostgresDatabase(settings.postgres)
        app.state.mongo = MongoDatabase(settings.mongo)

        # One registry and one cache for the process. Both are stateful in ways
        # that make per-request construction actively harmful: the cache exists
        # to be shared, and each provider holds a rate limiter whose budget is
        # only meaningful if every caller draws from the same one.
        app.state.market_data = MarketDataService(
            registry=build_registry(settings),
            cache=ResponseCache(),
            settings=settings.marketdata,
        )

        # MongoDB has no migration tool, so index creation runs at startup. It
        # is idempotent and non-fatal: a failure is logged and the API still
        # serves, because a missing index degrades latency rather than
        # correctness. PostgreSQL schema changes go through Alembic instead --
        # they are not safe to apply implicitly on boot.
        try:
            await create_indexes(app.state.mongo)
        except Exception:  # startup must survive a slow or unavailable index build
            logger.warning("mongo_index_setup_skipped", exc_info=True)

        try:
            yield
        finally:
            await app.state.market_data.aclose()
            await app.state.postgres.dispose()
            await app.state.mongo.close()
            logger.info("application_stopped")

    return lifespan


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and wire the FastAPI application.

    Args:
        settings: Configuration to use. Defaults to the process-wide settings.

    Returns:
        A fully configured application: logging, middleware, exception handlers
        and routers attached.
    """
    settings = settings or get_settings()
    configure_logging(settings.observability)

    app = FastAPI(
        title=settings.project_name,
        version=settings.version,
        debug=settings.debug,
        lifespan=_build_lifespan(settings),
        openapi_url=f"{settings.api_v1_prefix}/openapi.json" if settings.expose_docs else None,
        docs_url="/docs" if settings.expose_docs else None,
        redoc_url="/redoc" if settings.expose_docs else None,
        summary="Real-time intelligence over the AI infrastructure and semiconductor ecosystem.",
    )

    # Order matters: CORS is added last so it wraps outermost and can attach
    # headers to responses produced by the inner middleware and handlers.
    metrics = MetricsRegistry()
    app.state.metrics = metrics
    app.add_middleware(RequestContextMiddleware, settings=settings.observability)
    app.add_middleware(MetricsMiddleware, metrics=metrics)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time-ms"],
    )

    register_exception_handlers(app)

    # Attached at construction rather than in the lifespan so that settings are
    # available to dependencies even when the lifespan is not run (unit tests).
    app.state.settings = settings

    app.include_router(health_endpoints.router, prefix="/health")
    app.include_router(metrics_endpoints.router)
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    return app


app = create_app()
