"""Shared pytest fixtures.

Unit tests must run with no database, no network and no Docker -- otherwise the
suite stops being run. Infrastructure adapters are therefore replaced with test
doubles through ``app.dependency_overrides``, which is the payoff of routing all
construction through :mod:`app.api.deps`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_health_service
from app.core.config import Environment, Settings
from app.main import create_app
from app.schemas.health import DependencyHealth, DependencyStatus, ReadinessResponse


class StubHealthService:
    """Health service double returning a scripted readiness report."""

    def __init__(self, *, healthy: bool = True) -> None:
        """Configure whether the stub reports dependencies as up or down."""
        self._healthy = healthy

    async def check_readiness(self) -> ReadinessResponse:
        """Return a deterministic readiness report."""
        status = DependencyStatus.UP if self._healthy else DependencyStatus.DOWN
        return ReadinessResponse(
            status=status,
            checked_at=datetime.now(UTC),
            dependencies=[
                DependencyHealth(
                    name=name,
                    status=status,
                    latency_ms=1.0,
                    error=None if self._healthy else "connection refused",
                )
                for name in ("postgres", "mongodb")
            ],
        )


def healthy_health_service() -> StubHealthService:
    """Dependency override factory.

    A plain function rather than the class itself: FastAPI would otherwise
    inspect ``StubHealthService.__init__`` and expose ``healthy`` as a query
    parameter.
    """
    return StubHealthService()


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Settings pinned to the test environment with console-free JSON logs."""
    return Settings(
        environment=Environment.TEST,
        debug=True,
        cors_origins=["http://testserver"],
    )


@pytest.fixture
def app(settings: Settings) -> Iterator[FastAPI]:
    """Application instance with a healthy health-service double installed."""
    application = create_app(settings)
    application.dependency_overrides[get_health_service] = healthy_health_service
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """HTTP client speaking directly to the ASGI app -- no socket, no server.

    ``lifespan`` is not triggered by ``ASGITransport``, so no real database
    connection is ever opened by the unit suite.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client
