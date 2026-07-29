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

from collections.abc import Sequence
from dataclasses import dataclass

from app.clients.news_clients import (
    NewsApiProvider,
    RssProvider,
    YahooFinanceNewsProvider,
)
from app.clients.protocols import NewsProvider
from app.clients.yfinance_client import YFinancePriceProvider
from app.core.config import Settings
from app.core.logging import get_logger
from app.db.mongo import MongoDatabase
from app.db.postgres import PostgresDatabase
from app.repositories.anomaly import AnomalyRepository
from app.repositories.company import CompanyRepository, TickerRepository
from app.repositories.documents import NewsRepository, RagChunkRepository
from app.repositories.market import MarketCalendarRepository
from app.repositories.price import DailyPriceRepository, TechnicalIndicatorRepository
from app.services.anomalies import AnomalyDetectionService
from app.services.anomalies.detectors import (
    IsolationForestDetector,
    ZScoreDetector,
)
from app.services.features import FeatureEngineeringService
from app.services.ingestion import NewsIngestionService, PriceIngestionService
from app.services.market_calendar_service import MarketCalendarService
from app.services.rag import (
    CorrelationEngine,
    DocumentIndexingService,
    build_embedding_provider,
)
from app.services.sentiment import build_analyzer
from app.services.sentiment_service import SentimentScoringService

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

    # The tracked symbols are read first: the Yahoo provider fetches one feed
    # per ticker, so it cannot be built without them.
    async with context.postgres.session() as session:
        symbols = [listing.symbol for listing in await TickerRepository(session).list_active()]

    providers = _build_news_providers(context.settings, symbols)
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


def _build_news_providers(settings: Settings, symbols: Sequence[str]) -> list[NewsProvider]:
    """Instantiate the news providers that are actually usable.

    NewsAPI is included only when a key is configured. The platform is designed
    to run without one -- RSS needs no credential -- so a missing key degrades
    coverage rather than breaking ingestion.
    """
    providers: list[NewsProvider] = [RssProvider(settings.ingestion)]
    if settings.ingestion.yahoo_news_enabled and symbols:
        providers.append(YahooFinanceNewsProvider(settings.ingestion, symbols))
    if NewsApiProvider.is_configured(settings.ingestion):
        providers.append(NewsApiProvider(settings.ingestion))
    else:
        logger.info("newsapi_disabled", reason="INGEST_NEWSAPI_KEY is not set")
    return providers


async def compute_features_job(context: JobContext) -> None:
    """Recompute technical indicators from the stored price history.

    Scheduled to run *after* price ingestion, not concurrently: features are a
    pure function of prices, so computing them against a half-written batch
    would produce values that the next run silently corrects. Sequencing them
    means every stored indicator corresponds to a complete session.
    """
    log = logger.bind(job="compute_features")
    try:
        async with context.postgres.session() as session:
            service = FeatureEngineeringService(
                tickers=TickerRepository(session),
                prices=DailyPriceRepository(session),
                features=TechnicalIndicatorRepository(session),
            )
            report = await service.compute_all()

        log.info(
            "job_succeeded",
            rows_written=report.rows_written,
            failures=[failure.symbol for failure in report.failures],
            duration_seconds=round(report.duration_seconds, 2),
        )
    except Exception:
        log.exception("job_failed")


async def detect_anomalies_job(context: JobContext) -> None:
    """Rebuild the exchange calendar, then run both anomaly detectors.

    The calendar is rebuilt first and in the same job, because detection depends
    on it: without knowing which sessions the market was open, a holiday reads as
    a collapse in trading activity. Running them as separate scheduled jobs would
    let detection fire against a stale calendar the one time it matters -- the
    session after a new holiday.
    """
    log = logger.bind(job="detect_anomalies")
    analysis = context.settings.analysis
    try:
        async with context.postgres.session() as session:
            calendar = MarketCalendarService(
                prices=DailyPriceRepository(session),
                calendar=MarketCalendarRepository(session),
            )
            calendar_report = await calendar.rebuild()

            service = AnomalyDetectionService(
                tickers=TickerRepository(session),
                indicators=TechnicalIndicatorRepository(session),
                anomalies=AnomalyRepository(session),
                calendar=MarketCalendarRepository(session),
                z_score=ZScoreDetector(threshold=analysis.z_score_threshold),
                isolation_forest=IsolationForestDetector(
                    contamination=analysis.isolation_forest_contamination
                ),
                lookback_sessions=analysis.anomaly_lookback_sessions,
            )
            report = await service.detect_all()

        log.info(
            "job_succeeded",
            calendar_sessions=calendar_report.trading_days,
            detections=report.detections,
            failures=[failure.symbol for failure in report.failures],
            duration_seconds=round(report.duration_seconds, 2),
        )
    except Exception:
        log.exception("job_failed")


async def score_sentiment_job(context: JobContext) -> None:
    """Score any recent article that has no sentiment yet."""
    log = logger.bind(job="score_sentiment")
    analysis = context.settings.analysis
    try:
        service = SentimentScoringService(
            analyzer=build_analyzer(prefer_finbert=analysis.use_finbert),
            news=NewsRepository(context.mongo),
        )
        report = await service.score_pending()

        if report.succeeded:
            log.info(
                "job_succeeded",
                scored=report.scored,
                model=report.model_name,
                duration_seconds=round(report.duration_seconds, 2),
            )
        else:
            log.error("job_failed", error=report.error)
    except Exception:
        log.exception("job_failed")


async def index_documents_job(context: JobContext) -> None:
    """Chunk and embed any article not yet indexed by the active model.

    Runs after sentiment scoring so each chunk carries its article's sentiment
    as filterable metadata -- making "bearish news about Micron last week" a
    single query rather than a join.
    """
    log = logger.bind(job="index_documents")
    embedding = context.settings.embedding
    try:
        service = DocumentIndexingService(
            embedder=build_embedding_provider(embedding, context.settings.ingestion),
            news=NewsRepository(context.mongo),
            chunks=RagChunkRepository(context.mongo),
            chunk_size=embedding.chunk_size,
            chunk_overlap=embedding.chunk_overlap,
            documents_per_run=embedding.documents_per_run,
        )
        report = await service.index_pending()

        if report.succeeded:
            log.info(
                "job_succeeded",
                documents=report.documents,
                chunks=report.chunks,
                model=report.model_name,
                duration_seconds=round(report.duration_seconds, 2),
            )
        else:
            log.error("job_failed", error=report.error)
    except Exception:
        log.exception("job_failed")


async def explain_anomalies_job(context: JobContext) -> None:
    """Correlate unexplained anomalies with news published around their session.

    Runs after detection, because it explains what the detectors found and has
    nothing to do until they have run.
    """
    log = logger.bind(job="explain_anomalies")
    llm = context.settings.llm
    try:
        async with context.postgres.session() as session:
            engine = CorrelationEngine(
                news=NewsRepository(context.mongo),
                anomalies=AnomalyRepository(session),
                tickers=TickerRepository(session),
                lookback_hours=llm.correlation_lookback_hours,
                lookahead_hours=llm.correlation_lookahead_hours,
            )
            results = await engine.explain_pending()

        log.info("job_succeeded", explained=len(results))
    except Exception:
        log.exception("job_failed")
