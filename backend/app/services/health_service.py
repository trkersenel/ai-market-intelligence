"""Health probing service.

Demonstrates the layering rule the whole backend follows: the service depends on
*infrastructure abstractions* (``PostgresDatabase``, ``MongoDatabase``) injected
through its constructor, never on FastAPI, request objects or global state. That
is what lets it be unit-tested with fakes and reused from the scheduler.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from app.core.logging import get_logger
from app.db.mongo import MongoDatabase
from app.db.postgres import PostgresDatabase
from app.schemas.health import DependencyHealth, DependencyStatus, ReadinessResponse

logger = get_logger(__name__)

#: A dependency probe: succeeds silently, raises on failure.
type Probe = Callable[[], Coroutine[Any, Any, None]]


class HealthService:
    """Probes every backing store and aggregates the result."""

    def __init__(self, postgres: PostgresDatabase, mongo: MongoDatabase) -> None:
        """Store the injected infrastructure adapters.

        Args:
            postgres: Relational store wrapper.
            mongo: Document store wrapper.
        """
        self._postgres = postgres
        self._mongo = mongo

    async def check_readiness(self) -> ReadinessResponse:
        """Probe all dependencies concurrently.

        Returns:
            A readiness report whose overall status is ``up`` only if every
            dependency responded successfully.
        """
        dependencies = await asyncio.gather(
            self._probe("postgres", self._ping_postgres),
            self._probe("mongodb", self._mongo.ping),
        )
        overall = (
            DependencyStatus.UP
            if all(dep.status is DependencyStatus.UP for dep in dependencies)
            else DependencyStatus.DOWN
        )
        return ReadinessResponse(
            status=overall,
            checked_at=datetime.now(UTC),
            dependencies=list(dependencies),
        )

    async def _ping_postgres(self) -> None:
        """Execute the cheapest possible round-trip against PostgreSQL."""
        async with self._postgres.session() as session:
            await session.execute(text("SELECT 1"))

    @staticmethod
    async def _probe(name: str, probe: Probe) -> DependencyHealth:
        """Run a probe, timing it and converting failures into a status.

        Args:
            name: Dependency identifier used in the report.
            probe: Zero-argument coroutine function that raises on failure.

        Returns:
            The health record for this dependency. Never raises: a failing
            dependency must produce a report, not a 500.
        """
        started = time.perf_counter()
        try:
            await probe()
        except Exception as exc:  # noqa: BLE001 - any failure means "down"
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.warning("dependency_probe_failed", dependency=name, error=str(exc))
            return DependencyHealth(
                name=name,
                status=DependencyStatus.DOWN,
                latency_ms=round(elapsed_ms, 2),
                error=f"{type(exc).__name__}: {exc}",
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return DependencyHealth(
            name=name,
            status=DependencyStatus.UP,
            latency_ms=round(elapsed_ms, 2),
        )
