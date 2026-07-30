"""Queries over the browsable universe."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from sqlalchemy import Select, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models.company import Ticker
from app.models.listing import Listing
from app.repositories.base import BaseRepository

#: Columns a sync refreshes on an existing row. ``symbol`` is excluded because
#: it is the conflict key, and the timestamp columns are handled separately.
_UPSERT_COLUMNS = (
    "name",
    "exchange",
    "currency",
    "security_type",
    "figi",
    "is_active",
    "source",
    "synced_at",
)

#: Chunk size for the bulk upsert. PostgreSQL's parameter limit is 65535 and
#: each row binds ten of them, so this leaves a wide margin while still cutting
#: a full NASDAQ sync to a handful of statements.
_BATCH_SIZE = 2000


class ListingRepository(BaseRepository[Listing, int]):
    """Reads and writes over the exchange universe."""

    model = Listing

    async def upsert_many(self, rows: Sequence[dict[str, object]]) -> int:
        """Insert or refresh listings, keyed on symbol.

        Idempotent by construction: a sync that runs twice leaves the same rows,
        and one that is interrupted can simply be run again. Conflict resolution
        happens inside PostgreSQL rather than as a read-then-write, which would
        open a race between the scheduler and a manual trigger.

        Args:
            rows: Column dictionaries, each carrying at least ``symbol``.

        Returns:
            The number of rows written.
        """
        if not rows:
            return 0

        written = 0
        for start in range(0, len(rows), _BATCH_SIZE):
            batch = list(rows[start : start + _BATCH_SIZE])
            statement = pg_insert(Listing).values(batch)
            statement = statement.on_conflict_do_update(
                index_elements=[Listing.symbol],
                set_={column: statement.excluded[column] for column in _UPSERT_COLUMNS},
            )
            written += await self._execute_dml(statement)
        return written

    async def deactivate_missing(self, symbols: Iterable[str], *, source: str) -> int:
        """Mark rows absent from the latest sync as inactive.

        Delisted symbols are flagged, never deleted. They still appear in stored
        news, in anomaly history and in users' watchlists, and a dangling
        reference is a worse outcome than a row marked inactive.

        Scoped to ``source`` so one provider's sync cannot deactivate rows
        another provider supplied -- which is what would happen the first time a
        second provider covering a different exchange was added.
        """
        present = {symbol.upper() for symbol in symbols}
        if not present:
            return 0

        return await self._execute_dml(
            update(Listing)
            .where(
                Listing.source == source,
                Listing.is_active.is_(True),
                Listing.symbol.not_in(present),
            )
            .values(is_active=False, synced_at=datetime.now(UTC))
        )

    async def get_by_symbol(self, symbol: str) -> Listing | None:
        """Return one listing, or ``None`` when the symbol is unknown."""
        result = await self._session.execute(
            select(Listing).where(Listing.symbol == symbol.upper())
        )
        return result.scalar_one_or_none()

    async def search(
        self,
        query: str,
        *,
        limit: int = 20,
        active_only: bool = True,
    ) -> Sequence[Listing]:
        """Return listings matching a symbol or name fragment.

        Ordering is what makes this feel like a real search box. An exact symbol
        match ranks first, then a symbol prefix, then a name match -- so typing
        "MU" surfaces Micron rather than the several dozen companies with "mu"
        somewhere in their name. Within a tier, shorter symbols come first,
        which reliably puts the primary listing above its warrants and units.
        """
        term = query.strip().upper()
        if not term:
            return []

        # ``term`` is upper-cased and both columns are folded to match, so the
        # search is case-insensitive without needing a functional index -- at a
        # few thousand rows the scan costs less than maintaining one would.
        pattern = f"%{term}%"
        statement = (
            select(Listing)
            .where(
                func.upper(Listing.symbol).like(pattern) | func.upper(Listing.name).like(pattern)
            )
            .order_by(
                # Booleans sort false-first in PostgreSQL, so each expression is
                # written to be false for the tier that should rank higher.
                Listing.symbol != term,
                ~func.upper(Listing.symbol).like(f"{term}%"),
                func.length(Listing.symbol),
                Listing.symbol,
            )
            .limit(limit)
        )
        if active_only:
            statement = statement.where(Listing.is_active.is_(True))

        result = await self._session.execute(statement)
        return result.scalars().all()

    async def count(self, *, active_only: bool = True) -> int:
        """Return how many listings are stored."""
        statement: Select[tuple[int]] = select(func.count()).select_from(Listing)
        if active_only:
            statement = statement.where(Listing.is_active.is_(True))
        result = await self._session.execute(statement)
        return int(result.scalar_one())

    async def tracked_symbols(self) -> set[str]:
        """Return the symbols that also exist as tracked tickers.

        The join between breadth and depth. A page uses it to show whether a
        listing has analysis behind it or is browse-only, which is the one
        distinction a user needs to understand about this platform.
        """
        result = await self._session.execute(select(Ticker.symbol))
        return {symbol for (symbol,) in result.all()}
