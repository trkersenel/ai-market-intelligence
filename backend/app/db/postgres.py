"""Async PostgreSQL engine and session lifecycle.

The engine is a process-wide resource created once during application startup
and disposed on shutdown; sessions are short-lived and scoped to a single unit
of work. Wrapping session creation in a context manager -- rather than letting
callers construct sessions ad hoc -- guarantees that every unit of work either
commits or rolls back, and never leaks a connection back to the pool.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.core.config import PostgresSettings
from app.core.logging import get_logger

logger = get_logger(__name__)


class PostgresDatabase:
    """Owns the async engine and session factory for the relational store.

    Instances are held on ``app.state`` and injected into request handlers, which
    keeps the module import-time free of side effects and makes it trivial to
    substitute a test database.
    """

    def __init__(self, settings: PostgresSettings, *, use_null_pool: bool = False) -> None:
        """Create the engine and session factory.

        Args:
            settings: Connection and pooling configuration.
            use_null_pool: Disable pooling. Required for test suites and for
                serverless runtimes where connections must not outlive a call.
        """
        self._settings = settings
        pool_kwargs: dict[str, object] = (
            {"poolclass": NullPool}
            if use_null_pool
            else {
                "pool_size": settings.pool_size,
                "max_overflow": settings.max_overflow,
                "pool_recycle": settings.pool_recycle_seconds,
                "pool_pre_ping": settings.pool_pre_ping,
            }
        )

        self._engine: AsyncEngine = create_async_engine(
            settings.async_dsn,
            echo=settings.echo_sql,
            future=True,
            connect_args={
                "server_settings": {
                    "application_name": "market-intel-api",
                    "statement_timeout": str(settings.statement_timeout_ms),
                }
            },
            **pool_kwargs,
        )
        self._session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

    @property
    def engine(self) -> AsyncEngine:
        """The underlying async engine."""
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Factory used by dependencies and background jobs to open sessions."""
        return self._session_factory

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a session bound to a single unit of work.

        The session is committed when the block exits normally and rolled back
        if it raises, so callers never manage transaction boundaries by hand.

        Yields:
            An :class:`AsyncSession` that is closed on exit.
        """
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def dispose(self) -> None:
        """Close every pooled connection. Called on application shutdown."""
        await self._engine.dispose()
        logger.info("postgres_engine_disposed")
