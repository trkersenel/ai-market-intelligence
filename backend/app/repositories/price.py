"""Repositories for price bars and derived technical indicators.

These carry the platform's idempotency guarantee. Ingestion jobs re-run: on a
retry after a timeout, on a manual backfill, on an overlapping schedule window.
Every write here is an upsert keyed on the table's natural unique constraint, so
running a job twice produces the same database state as running it once.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

from sqlalchemy import Row, delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models.company import Ticker
from app.models.price import DailyPrice, TechnicalIndicator
from app.repositories.base import BaseRepository


class DailyPriceRepository(BaseRepository[DailyPrice, int]):
    """Reads and idempotent writes over the OHLCV table."""

    model = DailyPrice

    #: Columns refreshed when an upsert hits an existing row. Excludes the key
    #: columns and ``created_at`` -- the first-seen timestamp must survive a
    #: correction to the bar itself.
    _UPSERT_COLUMNS = (
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "source",
        # Refreshed so yesterday's provisional bar is cleared when the next
        # run re-fetches it as a completed session.
        "is_provisional",
    )

    async def bulk_upsert(self, rows: Sequence[dict[str, Any]]) -> int:
        """Insert price bars, updating any that already exist.

        Args:
            rows: Mappings with at least ``ticker_id``, ``trade_date`` and the
                OHLCV columns.

        Returns:
            The number of rows inserted or updated.

        Notes:
            Compiles to a single ``INSERT ... ON CONFLICT (ticker_id,
            trade_date) DO UPDATE``. One statement per batch rather than one per
            row matters at this table's volume, and pushing conflict resolution
            into PostgreSQL avoids the read-then-write race that a
            select-then-insert would open between concurrent workers.
        """
        if not rows:
            return 0

        statement = pg_insert(DailyPrice).values(list(rows))
        statement = statement.on_conflict_do_update(
            index_elements=[DailyPrice.ticker_id, DailyPrice.trade_date],
            set_={column: statement.excluded[column] for column in self._UPSERT_COLUMNS},
        )
        return await self._execute_dml(statement)

    async def get_range(
        self,
        ticker_id: int,
        *,
        start: date,
        end: date,
    ) -> Sequence[DailyPrice]:
        """Return bars for one ticker within an inclusive date window, oldest first."""
        result = await self._session.execute(
            select(DailyPrice)
            .where(
                DailyPrice.ticker_id == ticker_id,
                DailyPrice.trade_date >= start,
                DailyPrice.trade_date <= end,
            )
            .order_by(DailyPrice.trade_date)
        )
        return result.scalars().all()

    async def get_latest(self, ticker_id: int) -> DailyPrice | None:
        """Return the most recent bar for one ticker."""
        result = await self._session.execute(
            select(DailyPrice)
            .where(DailyPrice.ticker_id == ticker_id)
            .order_by(DailyPrice.trade_date.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_recent(
        self,
        ticker_id: int,
        *,
        sessions: int | None = None,
        completed_only: bool = False,
    ) -> Sequence[DailyPrice]:
        """Return the last ``sessions`` bars for one ticker, oldest first.

        Args:
            ticker_id: Listing to read.
            sessions: How many recent bars to return; ``None`` returns the whole
                stored history, which the feature pipeline needs when an
                indicator definition changes and every row must be recomputed.
            completed_only: Exclude the still-trading session. Statistics must
                set this; a chart showing today's move must not.

        Notes:
            Fetched in descending order to use the index, then reversed in
            Python -- cheaper than making PostgreSQL sort the result set again.
        """
        statement = select(DailyPrice).where(DailyPrice.ticker_id == ticker_id)
        if completed_only:
            statement = statement.where(DailyPrice.is_provisional.is_(False))
        statement = statement.order_by(DailyPrice.trade_date.desc())
        if sessions is not None:
            statement = statement.limit(sessions)
        result = await self._session.execute(statement)
        return list(reversed(result.scalars().all()))

    async def get_cross_section(self, trade_date: date) -> Sequence[Row[tuple[str, DailyPrice]]]:
        """Return every ticker's bar for one session, with its symbol.

        Powers the market heatmap. Returns rows rather than ORM objects so the
        symbol arrives in the same query instead of triggering a load per bar.
        """
        result = await self._session.execute(
            select(Ticker.symbol, DailyPrice)
            .join(DailyPrice, DailyPrice.ticker_id == Ticker.id)
            .where(DailyPrice.trade_date == trade_date)
            .order_by(Ticker.symbol)
        )
        return result.all()

    async def get_sessions_by_exchange(self) -> Sequence[tuple[str, date]]:
        """Return every distinct ``(exchange, session)`` pair observed.

        The input to calendar reconstruction. Provisional bars are excluded: a
        session that is still trading has not yet happened in full, and marking
        it closed-or-open on partial evidence is what the flag exists to prevent.

        Aggregated in the database rather than by loading bars and grouping in
        Python -- this is one DISTINCT over an indexed column against millions
        of rows.
        """
        result = await self._session.execute(
            select(Ticker.exchange, DailyPrice.trade_date)
            .join(DailyPrice, DailyPrice.ticker_id == Ticker.id)
            .where(
                Ticker.exchange.is_not(None),
                DailyPrice.is_provisional.is_(False),
            )
            .distinct()
        )
        # `exchange` is nullable on the model but excluded by the WHERE clause
        # above, so the rows are rebuilt as plain tuples with the narrower type
        # rather than pushed to callers as `str | None`.
        return [(exchange, session) for exchange, session in result.all()]

    async def get_date_bounds(self, ticker_id: int) -> tuple[date, date] | None:
        """Return the oldest and newest session stored for one ticker.

        Returns ``None`` when no bars exist. Used after ingestion to refresh the
        watermarks on ``ticker``.
        """
        result = await self._session.execute(
            select(
                func.min(DailyPrice.trade_date),
                func.max(DailyPrice.trade_date),
            ).where(DailyPrice.ticker_id == ticker_id)
        )
        first, last = result.one()
        if first is None or last is None:
            return None
        return first, last


class TechnicalIndicatorRepository(BaseRepository[TechnicalIndicator, int]):
    """Reads and idempotent writes over the derived-feature table."""

    model = TechnicalIndicator

    #: Every column except the identity, the natural key and ``created_at``:
    #: recomputing features must overwrite whatever a previous run wrote.
    _EXCLUDED_FROM_UPSERT = frozenset({"id", "ticker_id", "trade_date", "created_at"})

    async def bulk_upsert(self, rows: Sequence[dict[str, Any]]) -> int:
        """Insert indicator rows, overwriting any already computed.

        Args:
            rows: Mappings keyed by ``ticker_id`` and ``trade_date`` plus any
                subset of the indicator columns.

        Returns:
            The number of rows inserted or updated.
        """
        if not rows:
            return 0

        updatable = [
            column.name
            for column in TechnicalIndicator.__table__.columns
            if column.name not in self._EXCLUDED_FROM_UPSERT
        ]
        statement = pg_insert(TechnicalIndicator).values(list(rows))
        statement = statement.on_conflict_do_update(
            index_elements=[TechnicalIndicator.ticker_id, TechnicalIndicator.trade_date],
            set_={column: statement.excluded[column] for column in updatable},
        )
        return await self._execute_dml(statement)

    async def delete_after(self, ticker_id: int, *, after: date) -> int:
        """Delete indicator rows for sessions later than ``after``.

        Returns:
            The number of rows removed.

        Notes:
            Upserts can only correct rows a run actually writes. When a session
            is reclassified -- a bar computed yesterday while trading, now
            excluded as provisional -- the row it produced would otherwise
            survive forever as the newest, wrong, answer. This is the sweep that
            retires it.
        """
        return await self._execute_dml(
            delete(TechnicalIndicator).where(
                TechnicalIndicator.ticker_id == ticker_id,
                TechnicalIndicator.trade_date > after,
            )
        )

    async def get_recent(self, ticker_id: int, *, sessions: int) -> Sequence[TechnicalIndicator]:
        """Return the last ``sessions`` feature rows for one ticker, oldest first.

        The baseline window the anomaly detectors establish "normal" from.
        """
        result = await self._session.execute(
            select(TechnicalIndicator)
            .where(TechnicalIndicator.ticker_id == ticker_id)
            .order_by(TechnicalIndicator.trade_date.desc())
            .limit(sessions)
        )
        return list(reversed(result.scalars().all()))

    async def get_latest(self, ticker_id: int) -> TechnicalIndicator | None:
        """Return the most recently computed feature row for one ticker."""
        result = await self._session.execute(
            select(TechnicalIndicator)
            .where(TechnicalIndicator.ticker_id == ticker_id)
            .order_by(TechnicalIndicator.trade_date.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_range(
        self,
        ticker_id: int,
        *,
        start: date,
        end: date,
    ) -> Sequence[TechnicalIndicator]:
        """Return feature rows within an inclusive date window, oldest first."""
        result = await self._session.execute(
            select(TechnicalIndicator)
            .where(
                TechnicalIndicator.ticker_id == ticker_id,
                TechnicalIndicator.trade_date >= start,
                TechnicalIndicator.trade_date <= end,
            )
            .order_by(TechnicalIndicator.trade_date)
        )
        return result.scalars().all()
