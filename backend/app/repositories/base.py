"""Generic repository base.

The repository is the only layer that knows SQL exists. Services ask domain
questions -- "the last 90 sessions for these tickers" -- and repositories decide
how to answer them, so replacing a query with a materialised view or a cache
later touches one class instead of every caller.

Generic over both the model and its primary-key type, because the platform uses
``BIGINT`` keys for high-volume internal rows and ``UUID`` keys for anything a
client can see. One base would otherwise have to lie about one of them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from sqlalchemy import CursorResult, Executable, Select, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute

from app.core.exceptions import NotFoundError
from app.db.base import Base


class BaseRepository[ModelT: Base, IdT]:
    """CRUD operations shared by every concrete repository.

    Subclasses set :attr:`model` and add the queries that express their own
    domain vocabulary. Nothing here flushes or commits: transaction boundaries
    belong to the caller's unit of work, so a service can compose several
    repository calls into one atomic operation.
    """

    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to a unit of work.

        Args:
            session: The session owning the current transaction.
        """
        self._session = session

    @property
    def session(self) -> AsyncSession:
        """The session this repository writes through."""
        return self._session

    # --- Reads ------------------------------------------------------------

    async def get(self, entity_id: IdT) -> ModelT | None:
        """Return the entity with this primary key, or ``None``."""
        return await self._session.get(self.model, entity_id)

    async def get_or_raise(self, entity_id: IdT) -> ModelT:
        """Return the entity with this primary key.

        Raises:
            NotFoundError: If no row has that key. Translated to a 404 by the
                API layer, so handlers never write their own existence checks.
        """
        entity = await self.get(entity_id)
        if entity is None:
            msg = f"{self.model.__name__} {entity_id!r} was not found."
            raise NotFoundError(msg, details={"entity": self.model.__name__})
        return entity

    async def list(
        self,
        *,
        limit: int | None = 100,
        offset: int = 0,
        order_by: InstrumentedAttribute[Any] | None = None,
    ) -> Sequence[ModelT]:
        """Return a page of entities.

        Args:
            limit: Maximum rows to return; ``None`` disables the cap.
            offset: Rows to skip.
            order_by: Sort column. Defaults to the primary key so that paging is
                deterministic -- without an ORDER BY, PostgreSQL may return the
                same row on two consecutive pages.
        """
        statement: Select[tuple[ModelT]] = select(self.model)
        statement = statement.order_by(order_by if order_by is not None else self.model.id)  # type: ignore[attr-defined]
        if offset:
            statement = statement.offset(offset)
        if limit is not None:
            statement = statement.limit(limit)
        result = await self._session.execute(statement)
        return result.scalars().all()

    async def count(self) -> int:
        """Return the total number of rows."""
        result = await self._session.execute(select(func.count()).select_from(self.model))
        return result.scalar_one()

    async def exists(self, entity_id: IdT) -> bool:
        """Return whether a row with this primary key exists.

        Cheaper than :meth:`get`: the row is never loaded into the identity map.
        """
        statement = select(
            select(self.model).where(self.model.id == entity_id).exists()  # type: ignore[attr-defined]
        )
        result = await self._session.execute(statement)
        return bool(result.scalar())

    # --- Writes -----------------------------------------------------------

    def add(self, entity: ModelT) -> ModelT:
        """Stage a new entity for insertion.

        Returns the same instance for call-site convenience. Server-generated
        columns (id, timestamps) are unset until the session flushes.
        """
        self._session.add(entity)
        return entity

    def add_all(self, entities: Sequence[ModelT]) -> Sequence[ModelT]:
        """Stage several entities for insertion."""
        self._session.add_all(entities)
        return entities

    async def delete(self, entity: ModelT) -> None:
        """Stage an entity for deletion."""
        await self._session.delete(entity)

    async def delete_by_id(self, entity_id: IdT) -> int:
        """Delete by primary key without loading the row.

        Returns:
            The number of rows deleted: 0 or 1.
        """
        return await self._execute_dml(
            delete(self.model).where(self.model.id == entity_id)  # type: ignore[attr-defined]
        )

    async def _execute_dml(self, statement: Executable) -> int:
        """Execute an INSERT, UPDATE or DELETE and return the affected row count.

        ``AsyncSession.execute`` is typed as returning ``Result``, but DML
        always yields a ``CursorResult``, which is the only one carrying
        ``rowcount``. The narrowing happens once here instead of at every call
        site that needs to know how many rows an upsert touched.
        """
        result = await self._session.execute(statement)
        return cast("CursorResult[Any]", result).rowcount

    async def flush(self) -> None:
        """Emit pending SQL without committing.

        Used when the caller needs server-generated values -- a primary key for
        a child row -- before the transaction ends.
        """
        await self._session.flush()
