"""Seed the reference universe into PostgreSQL and ensure MongoDB indexes.

Run as a module:

    python -m app.db.seed

Idempotent by construction: companies are matched on ``slug`` and tickers on
``symbol``, so re-running updates the existing rows instead of failing on a
unique violation or duplicating the universe. That property is what lets this
run unattended as a deployment step.
"""

from __future__ import annotations

import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db.mongo import MongoDatabase
from app.db.mongo_indexes import create_indexes
from app.db.postgres import PostgresDatabase
from app.db.seed_data import COMPANIES, ETFS, CompanySeed, TickerSeed
from app.models.company import Company, Ticker
from app.repositories.company import CompanyRepository, TickerRepository

logger = get_logger(__name__)


async def _upsert_company(repository: CompanyRepository, seed: CompanySeed) -> Company:
    """Create or refresh one company row, returning the persisted instance."""
    company = await repository.get_by_slug(seed.slug)
    if company is None:
        company = Company(slug=seed.slug)
        repository.add(company)

    company.name = seed.name
    company.sector = seed.sector
    company.industry = seed.industry
    company.country = seed.country
    company.website = seed.website
    company.description = seed.description
    company.tags = [tag.value for tag in seed.tags]
    company.is_tracked = True

    # Flush so the identity is available for the ticker rows below; the
    # transaction still commits as a whole.
    await repository.flush()
    return company


async def _upsert_ticker(
    repository: TickerRepository, seed: TickerSeed, *, company_id: int | None
) -> Ticker:
    """Create or refresh one listing row, leaving ingestion watermarks untouched."""
    ticker = await repository.get_by_symbol(seed.symbol)
    if ticker is None:
        ticker = Ticker(symbol=seed.symbol.upper())
        repository.add(ticker)

    ticker.company_id = company_id
    ticker.display_name = seed.display_name
    ticker.exchange = seed.exchange
    ticker.currency = seed.currency
    ticker.asset_type = seed.asset_type
    ticker.is_active = True
    return ticker


async def seed_reference_data(session: AsyncSession) -> tuple[int, int]:
    """Write the tracked universe into an open session.

    Args:
        session: Session owning the transaction. Not committed here -- the
            caller decides the boundary, which lets tests roll the whole seed
            back and lets a deployment script batch it with other work.

    Returns:
        The number of companies and tickers written.
    """
    companies = CompanyRepository(session)
    tickers = TickerRepository(session)

    ticker_count = 0
    for seed in COMPANIES:
        company = await _upsert_company(companies, seed)
        for ticker_seed in seed.tickers:
            await _upsert_ticker(tickers, ticker_seed, company_id=company.id)
            ticker_count += 1

    for etf_seed in ETFS:
        await _upsert_ticker(tickers, etf_seed, company_id=None)
        ticker_count += 1

    await session.flush()
    logger.info("reference_data_seeded", companies=len(COMPANIES), tickers=ticker_count)
    return len(COMPANIES), ticker_count


async def main() -> None:
    """Seed PostgreSQL and ensure MongoDB indexes, then close both connections."""
    settings = get_settings()
    configure_logging(settings.observability)

    postgres = PostgresDatabase(settings.postgres)
    mongo = MongoDatabase(settings.mongo)
    try:
        async with postgres.session() as session:
            await seed_reference_data(session)
        await create_indexes(mongo)
    finally:
        await postgres.dispose()
        await mongo.close()


if __name__ == "__main__":
    asyncio.run(main())
