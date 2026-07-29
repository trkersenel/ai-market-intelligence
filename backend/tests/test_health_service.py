"""Unit tests for :class:`app.services.health_service.HealthService`.

The service is exercised with fakes rather than live databases -- the point of
injecting infrastructure through the constructor is that these tests need no
Docker, no network and no fixtures beyond plain objects.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from app.schemas.health import DependencyStatus
from app.services.health_service import HealthService


class FakeSession:
    """Minimal async session double recording the statements it executed."""

    def __init__(self, *, fail: bool = False) -> None:
        """Configure whether ``execute`` raises."""
        self._fail = fail
        self.executed: list[Any] = []

    async def execute(self, statement: Any) -> None:
        """Record the statement, or raise if configured to fail."""
        if self._fail:
            msg = "connection refused"
            raise ConnectionError(msg)
        self.executed.append(statement)


class FakePostgres:
    """Stands in for :class:`app.db.postgres.PostgresDatabase`."""

    def __init__(self, *, fail: bool = False) -> None:
        """Create a fake whose sessions succeed or fail as configured."""
        self.last_session = FakeSession(fail=fail)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[FakeSession]:
        """Yield the fake session, mirroring the real context-manager API."""
        yield self.last_session


class FakeMongo:
    """Stands in for :class:`app.db.mongo.MongoDatabase`."""

    def __init__(self, *, fail: bool = False) -> None:
        """Create a fake whose ping succeeds or fails as configured."""
        self._fail = fail
        self.ping_count = 0

    async def ping(self) -> None:
        """Count the probe, raising if configured to fail."""
        self.ping_count += 1
        if self._fail:
            msg = "server selection timed out"
            raise TimeoutError(msg)


def _service(*, postgres_fails: bool = False, mongo_fails: bool = False) -> HealthService:
    """Build a service wired to fakes with the requested failure modes."""
    return HealthService(
        postgres=FakePostgres(fail=postgres_fails),  # type: ignore[arg-type]
        mongo=FakeMongo(fail=mongo_fails),  # type: ignore[arg-type]
    )


async def test_all_dependencies_up_reports_overall_up() -> None:
    report = await _service().check_readiness()

    assert report.status is DependencyStatus.UP
    assert all(dep.status is DependencyStatus.UP for dep in report.dependencies)
    assert all(dep.error is None for dep in report.dependencies)


@pytest.mark.parametrize(
    ("postgres_fails", "mongo_fails", "expected_down"),
    [
        (True, False, "postgres"),
        (False, True, "mongodb"),
    ],
)
async def test_single_failure_marks_only_that_dependency_down(
    postgres_fails: bool,
    mongo_fails: bool,
    expected_down: str,
) -> None:
    service = _service(postgres_fails=postgres_fails, mongo_fails=mongo_fails)

    report = await service.check_readiness()

    assert report.status is DependencyStatus.DOWN
    down = {dep.name for dep in report.dependencies if dep.status is DependencyStatus.DOWN}
    assert down == {expected_down}


async def test_probe_failure_is_captured_not_raised() -> None:
    """A dead dependency must produce a report, never propagate an exception."""
    report = await _service(postgres_fails=True, mongo_fails=True).check_readiness()

    assert report.status is DependencyStatus.DOWN
    for dep in report.dependencies:
        assert dep.error
        assert dep.latency_ms >= 0


async def test_latency_is_recorded_for_healthy_dependencies() -> None:
    report = await _service().check_readiness()

    assert all(dep.latency_ms >= 0 for dep in report.dependencies)
