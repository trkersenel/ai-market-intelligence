"""Job functions the scheduler invokes.

Each job is a thin composition root: it opens a unit of work, assembles the
services it needs, runs one, and closes everything. The business logic lives in
the services, so a job can be invoked from the scheduler, from a CLI, or from a
test with no scheduler at all.

Jobs never raise. An unhandled exception inside APScheduler kills the job's
next run in some configurations and is invisible in others; catching and logging
here means a failed run is loud, recorded, and followed by another attempt on
the next tick.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.clients.news_clients import NewsApiProvider, RssProvider
from app.clients.protocols import NewsProvider
from app.clients.yfinance_client import YFinancePriceProvider
from app.core.config import Settings
from app.core.logging import get_logger
from app.db.mongo import MongoDatabase
from app.db.postgres import PostgresDatabase
from app.repositories.company import CompanyRepository, TickerRepository
from app.repositories.documents import NewsRepository
from app.repositories.price import DailyPriceRepository
from app.services.ingestion import NewsIngestionService, PriceIngestionService

logger = get_logger(__name__)


@dataclass
class JobContext:
    """Long-lived resources shared by every job in the worker process.

    Connection pools are created once for the process, not per job run. A daily
    job that built its own engine would leave a pool behind on every tick.
    """

    settings: Settings
    postgres: PostgresDatabase
    mongo: MongoDatabase

    @classmethod
    def create(cls, settings: Settings) -> JobContext:
        """Open the worker's database connections."""
        return cls(
            settings=settings,
            postgres=PostgresDatabase(settings.postgres),
            mongo=MongoDatabase(settings.mongo),
        )

    async def aclose(self) -> None:
        """Release every pooled connection."""
        await self.postgres.dispose()
        await self.mongo.close()


async def ingest_prices_job(context: JobContext) -> None:
    """Fetch and store missing price history for every active listing."""
    log = logger.bind(job="ingest_prices")
    try:
        provider = YFinancePriceProvider(context.settings.ingestion)
        async with context.postgres.session() as session:
            service = PriceIngestionService(
                provider=provider,
                tickers=TickerRepository(session),
                prices=DailyPriceRepository(session),
                settings=context.settings.ingestion,
            )
            report = await service.ingest_all()

        log.info(
            "job_succeeded",
            bars_written=report.bars_written,
            failures=[failure.symbol for failure in report.failures],
            duration_seconds=round(report.duration_seconds, 2),
        )
    except Exception:
        # Never propagate: a raising job is a silently dead schedule.
        log.exception("job_failed")


async def ingest_news_job(context: JobContext) -> None:
    """Poll every configured news provider and store relevant articles."""
    log = logger.bind(job="ingest_news")
    providers = _build_news_providers(context.settings)
    if not providers:
        log.warning("job_skipped", reason="no news providers are configured")
        return

    try:
        async with context.postgres.session() as session:
            service = NewsIngestionService(
                providers=providers,
                news=NewsRepository(context.mongo),
                companies=CompanyRepository(session),
                tickers=TickerRepository(session),
            )
            report = await service.ingest_all()

        log.info(
            "job_succeeded",
            stored=report.stored,
            failures=[failure.provider for failure in report.failures],
        )
    except Exception:
        log.exception("job_failed")
    finally:
        for provider in providers:
            await provider.aclose()  # type: ignore[attr-defined]


def _build_news_providers(settings: Settings) -> list[NewsProvider]:
    """Instantiate the news providers that are actually usable.

    NewsAPI is included only when a key is configured. The platform is designed
    to run without one -- RSS needs no credential -- so a missing key degrades
    coverage rather than breaking ingestion.
    """
    providers: list[NewsProvider] = [RssProvider(settings.ingestion)]
    if NewsApiProvider.is_configured(settings.ingestion):
        providers.append(NewsApiProvider(settings.ingestion))
    else:
        logger.info("newsapi_disabled", reason="INGEST_NEWSAPI_KEY is not set")
    return providers
