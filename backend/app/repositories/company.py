"""Repositories for reference data: companies and tickers."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

from sqlalchemy import func, select, update
from sqlalchemy.orm import selectinload

from app.models.company import Company, Ticker
from app.models.enums import AssetType, EcosystemTag
from app.repositories.base import BaseRepository


class CompanyRepository(BaseRepository[Company, int]):
    """Queries over the tracked issuer universe."""

    model = Company

    async def get_by_slug(self, slug: str) -> Company | None:
        """Return the company with this slug, or ``None``."""
        result = await self._session.execute(select(Company).where(Company.slug == slug))
        return result.scalar_one_or_none()

    async def get_with_tickers(self, company_id: int) -> Company | None:
        """Return a company with its listings eagerly loaded.

        Uses ``selectinload`` rather than a lazy load: relationships are
        configured ``lazy="raise_on_sql"``, so touching ``company.tickers``
        without loading it raises instead of silently emitting an N+1 query.
        """
        result = await self._session.execute(
            select(Company).where(Company.id == company_id).options(selectinload(Company.tickers))
        )
        return result.scalar_one_or_none()

    async def list_by_tags(
        self, tags: Sequence[EcosystemTag], *, match_all: bool = False
    ) -> Sequence[Company]:
        """Return companies exposed to the given value-chain segments.

        Args:
            tags: Segments to match, e.g. ``[EcosystemTag.HBM]``.
            match_all: Require every tag rather than any of them.

        Notes:
            Both branches compile to a PostgreSQL array operator (``@>`` or
            ``&&``) served by the GIN index on ``company.tags`` -- no join table
            and no sequential scan.
        """
        values = [tag.value for tag in tags]
        condition = Company.tags.contains(values) if match_all else Company.tags.overlap(values)
        result = await self._session.execute(
            select(Company).where(condition).order_by(Company.name)
        )
        return result.scalars().all()

    async def list_tracked(self) -> Sequence[Company]:
        """Return every company currently included in ingestion."""
        result = await self._session.execute(
            select(Company).where(Company.is_tracked.is_(True)).order_by(Company.name)
        )
        return result.scalars().all()

    async def search(self, term: str, *, limit: int = 20) -> Sequence[Company]:
        """Return companies whose name or slug contains ``term``.

        Case-insensitive substring matching. Deliberately simple: semantic
        company search is served by the vector index, not by SQL.
        """
        pattern = f"%{term.strip()}%"
        result = await self._session.execute(
            select(Company)
            .where(Company.name.ilike(pattern) | Company.slug.ilike(pattern))
            .order_by(Company.name)
            .limit(limit)
        )
        return result.scalars().all()


class TickerRepository(BaseRepository[Ticker, int]):
    """Queries over tradable listings."""

    model = Ticker

    async def get_by_symbol(self, symbol: str) -> Ticker | None:
        """Return the listing with this symbol, or ``None``.

        Symbols are stored and compared upper-case so ``nvda`` and ``NVDA``
        resolve to the same row.
        """
        result = await self._session.execute(
            select(Ticker).where(Ticker.symbol == symbol.strip().upper())
        )
        return result.scalar_one_or_none()

    async def get_many_by_symbols(self, symbols: Sequence[str]) -> Sequence[Ticker]:
        """Return every listing matching the given symbols, in one query."""
        normalised = [symbol.strip().upper() for symbol in symbols]
        result = await self._session.execute(
            select(Ticker).where(Ticker.symbol.in_(normalised)).order_by(Ticker.symbol)
        )
        return result.scalars().all()

    async def list_active(self, *, asset_type: AssetType | None = None) -> Sequence[Ticker]:
        """Return the listings the ingestion scheduler should fetch."""
        statement = select(Ticker).where(Ticker.is_active.is_(True))
        if asset_type is not None:
            statement = statement.where(Ticker.asset_type == asset_type)
        result = await self._session.execute(statement.order_by(Ticker.symbol))
        return result.scalars().all()

    async def list_stale(self, *, before: date) -> Sequence[Ticker]:
        """Return active listings whose prices are missing or older than ``before``.

        This is the ingestion scheduler's work queue. Reading the denormalised
        watermark keeps it a single indexed scan instead of a MAX() over the
        price table.
        """
        result = await self._session.execute(
            select(Ticker)
            .where(
                Ticker.is_active.is_(True),
                (Ticker.last_price_date.is_(None)) | (Ticker.last_price_date < before),
            )
            .order_by(Ticker.symbol)
        )
        return result.scalars().all()

    async def update_watermarks(
        self,
        ticker_id: int,
        *,
        first_price_date: date | None,
        last_price_date: date,
        ingested_at: datetime,
    ) -> None:
        """Record how far price ingestion has progressed for one listing.

        The window only ever widens: ``LEAST``/``GREATEST`` are evaluated in the
        database, so a backfill of older history and a same-day incremental run
        can interleave -- in any order, from concurrent workers -- without either
        one narrowing the range the other recorded.
        """
        values: dict[str, object] = {
            "last_price_date": func.greatest(
                func.coalesce(Ticker.last_price_date, last_price_date), last_price_date
            ),
            "last_ingested_at": ingested_at,
        }
        if first_price_date is not None:
            values["first_price_date"] = func.least(
                func.coalesce(Ticker.first_price_date, first_price_date), first_price_date
            )
        await self._session.execute(update(Ticker).where(Ticker.id == ticker_id).values(**values))
