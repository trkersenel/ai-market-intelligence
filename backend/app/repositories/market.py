"""Repositories for the exchange calendar and generated market summaries."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models.market import DailyMarketSummary, MarketCalendar
from app.repositories.base import BaseRepository


class MarketCalendarRepository(BaseRepository[MarketCalendar, int]):
    """Queries over per-exchange trading sessions."""

    model = MarketCalendar

    async def bulk_upsert(self, rows: Sequence[dict[str, Any]]) -> int:
        """Insert or refresh calendar rows keyed by ``(exchange, session_date)``."""
        if not rows:
            return 0

        statement = pg_insert(MarketCalendar).values(list(rows))
        statement = statement.on_conflict_do_update(
            index_elements=[MarketCalendar.exchange, MarketCalendar.session_date],
            set_={
                "is_trading_day": statement.excluded.is_trading_day,
                "open_time": statement.excluded.open_time,
                "close_time": statement.excluded.close_time,
                "note": statement.excluded.note,
            },
        )
        return await self._execute_dml(statement)

    async def is_trading_day(self, exchange: str, session_date: date) -> bool | None:
        """Return whether an exchange traded on a date.

        Returns ``None`` when the date is not in the calendar at all -- an
        important distinction from ``False``: the ingestion pipeline should
        retry an unknown date but never a known holiday.
        """
        result = await self._session.execute(
            select(MarketCalendar.is_trading_day).where(
                MarketCalendar.exchange == exchange,
                MarketCalendar.session_date == session_date,
            )
        )
        return result.scalar_one_or_none()

    async def list_trading_days(self, exchange: str, *, start: date, end: date) -> Sequence[date]:
        """Return the sessions an exchange traded within an inclusive window."""
        result = await self._session.execute(
            select(MarketCalendar.session_date)
            .where(
                MarketCalendar.exchange == exchange,
                MarketCalendar.session_date >= start,
                MarketCalendar.session_date <= end,
                MarketCalendar.is_trading_day.is_(True),
            )
            .order_by(MarketCalendar.session_date)
        )
        return list(result.scalars().all())

    async def most_recent_trading_day(self, exchange: str, *, on_or_before: date) -> date | None:
        """Return the latest session at or before a date.

        Resolves "today's market" on a weekend or holiday, so the dashboard
        shows the last real session instead of an empty page.
        """
        result = await self._session.execute(
            select(MarketCalendar.session_date)
            .where(
                MarketCalendar.exchange == exchange,
                MarketCalendar.session_date <= on_or_before,
                MarketCalendar.is_trading_day.is_(True),
            )
            .order_by(MarketCalendar.session_date.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()


class MarketSummaryRepository(BaseRepository[DailyMarketSummary, int]):
    """Queries over generated daily briefings."""

    model = DailyMarketSummary

    async def get_by_date(self, summary_date: date) -> DailyMarketSummary | None:
        """Return the briefing for one session, or ``None``."""
        result = await self._session.execute(
            select(DailyMarketSummary).where(DailyMarketSummary.summary_date == summary_date)
        )
        return result.scalar_one_or_none()

    async def get_latest(self) -> DailyMarketSummary | None:
        """Return the most recent briefing."""
        result = await self._session.execute(
            select(DailyMarketSummary).order_by(DailyMarketSummary.summary_date.desc()).limit(1)
        )
        return result.scalar_one_or_none()

    async def list_recent(self, *, limit: int = 30) -> Sequence[DailyMarketSummary]:
        """Return recent briefings, newest first."""
        result = await self._session.execute(
            select(DailyMarketSummary).order_by(DailyMarketSummary.summary_date.desc()).limit(limit)
        )
        return result.scalars().all()
