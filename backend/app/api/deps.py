"""FastAPI dependency providers -- the composition root of the API layer.

Every request-scoped object is assembled here. Endpoints declare *what* they
need (a service, a session) and this module decides *how* it is built, which
keeps handlers free of construction logic and makes any collaborator trivially
replaceable in tests via ``app.dependency_overrides``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from functools import cache
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.protocols import PriceProvider
from app.clients.yfinance_client import YFinancePriceProvider
from app.core.config import Settings, get_settings
from app.core.exceptions import ServiceUnavailableError
from app.db.mongo import MongoDatabase
from app.db.postgres import PostgresDatabase
from app.repositories import (
    AnomalyRepository,
    CompanyRepository,
    DailyPriceRepository,
    MarketCalendarRepository,
    MarketSummaryRepository,
    PortfolioRepository,
    TechnicalIndicatorRepository,
    TickerRepository,
    UserRepository,
    WatchlistRepository,
)
from app.repositories.documents import NewsRepository
from app.services.features import FeatureEngineeringService
from app.services.health_service import HealthService
from app.services.ingestion import PriceIngestionService


def get_app_settings(request: Request) -> Settings:
    """Provide the settings the *current app instance* was built with.

    Reads from ``app.state`` rather than calling :func:`get_settings` directly,
    so an app constructed with explicit settings -- a test app, or a second app
    in the same process -- is never served the process-wide defaults.
    """
    settings: Settings | None = getattr(request.app.state, "settings", None)
    return settings if settings is not None else get_settings()


def get_postgres(request: Request) -> PostgresDatabase:
    """Return the process-wide PostgreSQL adapter created during startup."""
    postgres: PostgresDatabase | None = getattr(request.app.state, "postgres", None)
    if postgres is None:  # pragma: no cover - indicates a lifespan wiring bug
        msg = "PostgreSQL is not initialised."
        raise ServiceUnavailableError(msg)
    return postgres


def get_mongo(request: Request) -> MongoDatabase:
    """Return the process-wide MongoDB adapter created during startup."""
    mongo: MongoDatabase | None = getattr(request.app.state, "mongo", None)
    if mongo is None:  # pragma: no cover - indicates a lifespan wiring bug
        msg = "MongoDB is not initialised."
        raise ServiceUnavailableError(msg)
    return mongo


async def get_db_session(
    postgres: Annotated[PostgresDatabase, Depends(get_postgres)],
) -> AsyncIterator[AsyncSession]:
    """Yield a transactional session scoped to the current request.

    The session commits when the handler returns and rolls back if it raises, so
    a request is an atomic unit of work by default.
    """
    async with postgres.session() as session:
        yield session


@cache
def repository_provider[RepoT](
    repository_type: type[RepoT],
) -> Callable[[AsyncSession], RepoT]:
    """Build a FastAPI dependency that constructs ``repository_type``.

    Every repository takes exactly one argument -- the request-scoped session --
    so declaring twelve near-identical provider functions would be pure
    duplication. The returned callable keeps an explicit ``Depends`` on the
    session, which is what FastAPI needs to resolve the graph.

    Notes:
        ``@cache`` is load-bearing, not an optimisation. FastAPI matches
        ``dependency_overrides`` by callable *identity*, so an uncached factory
        would mint a new function on every call and no test could ever override
        the dependency the route actually declared.
    """

    def provider(session: Annotated[AsyncSession, Depends(get_db_session)]) -> RepoT:
        return repository_type(session)  # type: ignore[call-arg]

    provider.__name__ = f"get_{repository_type.__name__}"
    return provider


def get_health_service(
    postgres: Annotated[PostgresDatabase, Depends(get_postgres)],
    mongo: Annotated[MongoDatabase, Depends(get_mongo)],
) -> HealthService:
    """Assemble the health service from its infrastructure dependencies."""
    return HealthService(postgres=postgres, mongo=mongo)


def get_news_repository(
    mongo: Annotated[MongoDatabase, Depends(get_mongo)],
) -> NewsRepository:
    """Assemble the document repository for news."""
    return NewsRepository(mongo)


def get_price_provider(settings: Annotated[Settings, Depends(get_app_settings)]) -> PriceProvider:
    """Provide the price data source.

    Typed as the *protocol*, not the concrete class, so swapping yfinance for a
    paid vendor later is a change in this one function.
    """
    return YFinancePriceProvider(settings.ingestion)


def get_price_ingestion_service(
    provider: Annotated[PriceProvider, Depends(get_price_provider)],
    tickers: Annotated[TickerRepository, Depends(repository_provider(TickerRepository))],
    prices: Annotated[DailyPriceRepository, Depends(repository_provider(DailyPriceRepository))],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> PriceIngestionService:
    """Assemble the price ingestion service for on-demand API triggers."""
    return PriceIngestionService(
        provider=provider,
        tickers=tickers,
        prices=prices,
        settings=settings.ingestion,
    )


def get_feature_service(
    tickers: Annotated[TickerRepository, Depends(repository_provider(TickerRepository))],
    prices: Annotated[DailyPriceRepository, Depends(repository_provider(DailyPriceRepository))],
    features: Annotated[
        TechnicalIndicatorRepository,
        Depends(repository_provider(TechnicalIndicatorRepository)),
    ],
) -> FeatureEngineeringService:
    """Assemble the feature engineering service for on-demand API triggers."""
    return FeatureEngineeringService(tickers=tickers, prices=prices, features=features)


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_db_session)]
MongoDep = Annotated[MongoDatabase, Depends(get_mongo)]
HealthServiceDep = Annotated[HealthService, Depends(get_health_service)]
NewsRepoDep = Annotated[NewsRepository, Depends(get_news_repository)]
PriceIngestionDep = Annotated[PriceIngestionService, Depends(get_price_ingestion_service)]
FeatureServiceDep = Annotated[FeatureEngineeringService, Depends(get_feature_service)]

# Repository dependencies. Endpoints and services annotate with these aliases
# rather than constructing repositories, so a test can swap any one of them for
# a fake through `app.dependency_overrides`.
CompanyRepoDep = Annotated[CompanyRepository, Depends(repository_provider(CompanyRepository))]
TickerRepoDep = Annotated[TickerRepository, Depends(repository_provider(TickerRepository))]
PriceRepoDep = Annotated[DailyPriceRepository, Depends(repository_provider(DailyPriceRepository))]
IndicatorRepoDep = Annotated[
    TechnicalIndicatorRepository,
    Depends(repository_provider(TechnicalIndicatorRepository)),
]
AnomalyRepoDep = Annotated[AnomalyRepository, Depends(repository_provider(AnomalyRepository))]
CalendarRepoDep = Annotated[
    MarketCalendarRepository, Depends(repository_provider(MarketCalendarRepository))
]
SummaryRepoDep = Annotated[
    MarketSummaryRepository, Depends(repository_provider(MarketSummaryRepository))
]
UserRepoDep = Annotated[UserRepository, Depends(repository_provider(UserRepository))]
WatchlistRepoDep = Annotated[WatchlistRepository, Depends(repository_provider(WatchlistRepository))]
PortfolioRepoDep = Annotated[PortfolioRepository, Depends(repository_provider(PortfolioRepository))]
