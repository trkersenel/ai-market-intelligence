"""Fixtures for tests that need a real PostgreSQL instance.

Isolation strategy: the suite opens one connection per test, starts a
transaction on it, binds the session to that connection, and rolls back
afterwards. Nothing a test writes is ever committed, so tests cannot leak state
into each other and the schema is created exactly once for the session.

Truncating tables between tests would be the alternative; it is slower and it
cannot exercise the rollback behaviour the application itself relies on.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import sqlalchemy
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import PostgresSettings, get_settings
from app.models import Base

#: Tests run against a dedicated database so a careless run can never touch
#: development data.
TEST_DATABASE_SUFFIX = "_test"

pytestmark = pytest.mark.integration


def _test_settings() -> PostgresSettings:
    """Return PostgreSQL settings pointed at the dedicated test database."""
    settings = get_settings().postgres
    return settings.model_copy(update={"db": f"{settings.db}{TEST_DATABASE_SUFFIX}"})


async def _ensure_database_exists(settings: PostgresSettings) -> None:
    """Create the test database if it is not there yet.

    ``CREATE DATABASE`` cannot run inside a transaction, hence the AUTOCOMMIT
    isolation level and the connection to the maintenance database.
    """
    maintenance = settings.model_copy(update={"db": "postgres"})
    engine = create_async_engine(maintenance.async_dsn, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as connection:
            exists = await connection.scalar(
                sqlalchemy.text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": settings.db},
            )
            if not exists:
                await connection.execute(sqlalchemy.text(f'CREATE DATABASE "{settings.db}"'))
    finally:
        await engine.dispose()


@pytest.fixture(scope="session")
async def engine() -> AsyncIterator[AsyncEngine]:
    """Session-scoped engine bound to a freshly created test schema.

    The schema is built from ``Base.metadata`` rather than by running Alembic:
    these tests verify repository behaviour, and coupling them to the migration
    history would make every unrelated migration a reason for them to fail.
    Migration correctness is covered separately by ``alembic check``.
    """
    settings = _test_settings()
    try:
        await _ensure_database_exists(settings)
    except OSError as exc:  # pragma: no cover - environment without PostgreSQL
        pytest.skip(f"PostgreSQL is not reachable: {exc}")

    test_engine = create_async_engine(settings.async_dsn, poolclass=sqlalchemy.pool.NullPool)
    async with test_engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)

    yield test_engine
    await test_engine.dispose()


@pytest.fixture
async def connection(engine: AsyncEngine) -> AsyncIterator[AsyncConnection]:
    """Yield a connection inside a transaction that is always rolled back."""
    async with engine.connect() as conn:
        transaction = await conn.begin()
        try:
            yield conn
        finally:
            await transaction.rollback()


@pytest.fixture
async def session(connection: AsyncConnection) -> AsyncIterator[AsyncSession]:
    """Yield a session enrolled in the test's outer transaction.

    ``join_transaction_mode="create_savepoint"`` means a ``session.commit()``
    inside the code under test releases a SAVEPOINT rather than committing for
    real -- so repositories and services can be tested exactly as they run in
    production, and the outer rollback still discards everything.
    """
    factory = async_sessionmaker(
        bind=connection,
        class_=AsyncSession,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    async with factory() as test_session:
        yield test_session
