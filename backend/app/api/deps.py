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
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.protocols import PriceProvider
from app.clients.yfinance_client import YFinancePriceProvider
from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError, ServiceUnavailableError
from app.core.security import TokenType, decode_token
from app.db.mongo import MongoDatabase
from app.db.postgres import PostgresDatabase
from app.marketdata.service import MarketDataService
from app.models.user import User
from app.repositories import (
    AnomalyRepository,
    CompanyRepository,
    DailyPriceRepository,
    EntityRepository,
    ListingRepository,
    MarketCalendarRepository,
    MarketSummaryRepository,
    PortfolioRepository,
    RelationshipRepository,
    TechnicalIndicatorRepository,
    TickerRepository,
    UserRepository,
    WatchlistRepository,
)
from app.repositories.documents import (
    AiReportRepository,
    ChatRepository,
    NewsRepository,
    RagChunkRepository,
)
from app.services.anomalies import AnomalyDetectionService
from app.services.anomalies.detectors import IsolationForestDetector, ZScoreDetector
from app.services.auth_service import AuthService
from app.services.features import FeatureEngineeringService
from app.services.graph import GraphService
from app.services.health_service import HealthService
from app.services.ingestion import PriceIngestionService
from app.services.portfolio_service import PortfolioService, WatchlistService
from app.services.rag import (
    ChatService,
    CorrelationEngine,
    DocumentIndexingService,
    HybridSearchService,
    RagService,
    build_vector_store,
)
from app.services.rag.embeddings import EmbeddingProvider
from app.services.rag.llm import LlmClient
from app.services.reports import CompanyReportService
from app.services.universe import UniverseSyncService

#: `auto_error=False` so a missing header reaches our own dependency and is
#: translated into the platform's error envelope, rather than FastAPI emitting a
#: bare 403 that no other endpoint produces.
_bearer_scheme = HTTPBearer(auto_error=False, description="JWT access token")


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


def get_llm_client(request: Request) -> LlmClient:
    """Return the language model client selected once during startup.

    Selection probes the network, so doing it per request would both cost a
    round trip on every call and let the *same* endpoint answer from different
    models minute to minute as the probe succeeded or timed out.
    """
    client: LlmClient | None = getattr(request.app.state, "llm_client", None)
    if client is None:  # pragma: no cover - indicates a lifespan wiring bug
        msg = "The language model client is not initialised."
        raise ServiceUnavailableError(msg)
    return client


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    """Return the embedding provider selected once during startup."""
    provider: EmbeddingProvider | None = getattr(request.app.state, "embedding_provider", None)
    if provider is None:  # pragma: no cover - indicates a lifespan wiring bug
        msg = "The embedding provider is not initialised."
        raise ServiceUnavailableError(msg)
    return provider


def get_market_data(request: Request) -> MarketDataService:
    """Return the process-wide market data facade built during startup."""
    service: MarketDataService | None = getattr(request.app.state, "market_data", None)
    if service is None:  # pragma: no cover - indicates a lifespan wiring bug
        msg = "Market data is not initialised."
        raise ServiceUnavailableError(msg)
    return service


def get_universe_sync_service(
    market_data: Annotated[MarketDataService, Depends(get_market_data)],
    listings: Annotated[ListingRepository, Depends(repository_provider(ListingRepository))],
) -> UniverseSyncService:
    """Assemble the universe sync service for on-demand API triggers."""
    return UniverseSyncService(market_data=market_data, listings=listings)


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


def get_anomaly_service(
    tickers: Annotated[TickerRepository, Depends(repository_provider(TickerRepository))],
    indicators: Annotated[
        TechnicalIndicatorRepository,
        Depends(repository_provider(TechnicalIndicatorRepository)),
    ],
    anomalies: Annotated[AnomalyRepository, Depends(repository_provider(AnomalyRepository))],
    calendar: Annotated[
        MarketCalendarRepository, Depends(repository_provider(MarketCalendarRepository))
    ],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> AnomalyDetectionService:
    """Assemble the anomaly detection service with configured thresholds."""
    analysis = settings.analysis
    return AnomalyDetectionService(
        tickers=tickers,
        indicators=indicators,
        anomalies=anomalies,
        calendar=calendar,
        z_score=ZScoreDetector(threshold=analysis.z_score_threshold),
        isolation_forest=IsolationForestDetector(
            contamination=analysis.isolation_forest_contamination
        ),
        lookback_sessions=analysis.anomaly_lookback_sessions,
    )


def get_chunk_repository(
    mongo: Annotated[MongoDatabase, Depends(get_mongo)],
) -> RagChunkRepository:
    """Assemble the repository over embedded chunks."""
    return RagChunkRepository(mongo)


def get_indexing_service(
    settings: Annotated[Settings, Depends(get_app_settings)],
    embedder: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    news: Annotated[NewsRepository, Depends(get_news_repository)],
    chunks: Annotated[RagChunkRepository, Depends(get_chunk_repository)],
) -> DocumentIndexingService:
    """Assemble the document indexing service."""
    embedding = settings.embedding
    return DocumentIndexingService(
        embedder=embedder,
        news=news,
        chunks=chunks,
        chunk_size=embedding.chunk_size,
        chunk_overlap=embedding.chunk_overlap,
        documents_per_run=embedding.documents_per_run,
    )


async def get_search_service(
    request: Request,
    settings: Annotated[Settings, Depends(get_app_settings)],
    embedder: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    news: Annotated[NewsRepository, Depends(get_news_repository)],
) -> HybridSearchService:
    """Assemble the hybrid search service.

    The vector backend is probed once and cached on ``app.state``: the check is
    a round trip to the database, and repeating it per request would add latency
    to answer a question whose answer cannot change while the process runs.
    """
    store = getattr(request.app.state, "vector_store", None)
    if store is None:
        store = await build_vector_store(get_mongo(request))
        request.app.state.vector_store = store

    embedding = settings.embedding
    return HybridSearchService(
        embedder=embedder,
        vector_store=store,
        news=news,
        candidates=embedding.vector_candidates,
        rrf_k=embedding.rrf_k,
    )


def get_graph_service(
    entities: Annotated[EntityRepository, Depends(repository_provider(EntityRepository))],
    relationships: Annotated[
        RelationshipRepository, Depends(repository_provider(RelationshipRepository))
    ],
) -> GraphService:
    """Assemble the knowledge graph service."""
    return GraphService(entities=entities, relationships=relationships)


def get_ai_report_repository(
    mongo: Annotated[MongoDatabase, Depends(get_mongo)],
) -> AiReportRepository:
    """Assemble the briefing cache."""
    return AiReportRepository(mongo)


def get_report_service(
    settings: Annotated[Settings, Depends(get_app_settings)],
    market_data: Annotated[MarketDataService, Depends(get_market_data)],
    llm_client: Annotated[LlmClient, Depends(get_llm_client)],
    reports: Annotated[AiReportRepository, Depends(get_ai_report_repository)],
) -> CompanyReportService:
    """Assemble the on-demand company briefing service."""
    return CompanyReportService(
        market_data=market_data,
        llm=llm_client,
        reports=reports,
        ttl_hours=settings.llm.report_ttl_hours,
    )


def get_chat_repository(
    mongo: Annotated[MongoDatabase, Depends(get_mongo)],
) -> ChatRepository:
    """Assemble the conversation repository."""
    return ChatRepository(mongo)


def get_rag_service(
    settings: Annotated[Settings, Depends(get_app_settings)],
    search: Annotated[HybridSearchService, Depends(get_search_service)],
    embedder: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    llm_client: Annotated[LlmClient, Depends(get_llm_client)],
) -> RagService:
    """Assemble the question-answering pipeline."""
    llm = settings.llm
    return RagService(
        search=search,
        llm=llm_client,
        relevance_floor=embedder.relevance_floor,
        context_passages=llm.context_passages,
        passage_chars=llm.passage_chars,
    )


async def get_chat_service(
    rag: Annotated[RagService, Depends(get_rag_service)],
    chat: Annotated[ChatRepository, Depends(get_chat_repository)],
) -> ChatService:
    """Assemble the conversational wrapper."""
    return ChatService(rag=rag, chat=chat)


def get_correlation_engine(
    settings: Annotated[Settings, Depends(get_app_settings)],
    news: Annotated[NewsRepository, Depends(get_news_repository)],
    anomalies: Annotated[AnomalyRepository, Depends(repository_provider(AnomalyRepository))],
    tickers: Annotated[TickerRepository, Depends(repository_provider(TickerRepository))],
) -> CorrelationEngine:
    """Assemble the news-correlation engine."""
    llm = settings.llm
    return CorrelationEngine(
        news=news,
        anomalies=anomalies,
        tickers=tickers,
        lookback_hours=llm.correlation_lookback_hours,
        lookahead_hours=llm.correlation_lookahead_hours,
    )


def get_auth_service(
    users: Annotated[UserRepository, Depends(repository_provider(UserRepository))],
    watchlists: Annotated[WatchlistRepository, Depends(repository_provider(WatchlistRepository))],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> AuthService:
    """Assemble the authentication service."""
    return AuthService(users=users, watchlists=watchlists, settings=settings.security)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
    users: Annotated[UserRepository, Depends(repository_provider(UserRepository))],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> User:
    """Resolve the account the presented access token belongs to.

    Raises:
        AuthenticationError: If the header is missing, the token is invalid or
            expired, it is a refresh token rather than an access token, or the
            account no longer exists or has been deactivated.

    Notes:
        The user is loaded from the database on every request rather than
        reconstructed from the token's claims. A token is a bearer of identity,
        not a snapshot of state: deactivating an account has to take effect
        immediately, not when the token happens to expire.
    """
    if credentials is None or not credentials.credentials:
        msg = "Not authenticated."
        raise AuthenticationError(msg)

    claims = decode_token(settings.security, credentials.credentials, expected=TokenType.ACCESS)
    user = await users.get(claims.subject)
    if user is None or not user.is_active:
        msg = "Could not validate credentials."
        raise AuthenticationError(msg)
    return user


async def get_current_user_from_token(
    token: str | None, app_state: object, postgres: PostgresDatabase
) -> User:
    """Resolve a user from a raw token, outside the HTTP dependency graph.

    WebSocket handshakes carry no ``Authorization`` header that browsers can set,
    so the token arrives as a query parameter and cannot flow through
    :func:`get_current_user`. The verification is identical -- same signature
    check, same access-token-only rule, same fresh database read -- so a revoked
    or deactivated account is rejected on the socket exactly as on HTTP.

    Raises:
        AuthenticationError: If the token is absent, invalid, of the wrong type,
            or its account is gone or deactivated.
    """
    if not token:
        msg = "Not authenticated."
        raise AuthenticationError(msg)

    settings: Settings = getattr(app_state, "settings", None) or get_settings()
    claims = decode_token(settings.security, token, expected=TokenType.ACCESS)

    async with postgres.session() as session:
        user = await UserRepository(session).get(claims.subject)
        if user is None or not user.is_active:
            msg = "Could not validate credentials."
            raise AuthenticationError(msg)
        return user


def get_watchlist_service(
    watchlists: Annotated[WatchlistRepository, Depends(repository_provider(WatchlistRepository))],
    tickers: Annotated[TickerRepository, Depends(repository_provider(TickerRepository))],
) -> WatchlistService:
    """Assemble the watchlist service."""
    return WatchlistService(watchlists=watchlists, tickers=tickers)


def get_portfolio_service(
    portfolios: Annotated[PortfolioRepository, Depends(repository_provider(PortfolioRepository))],
    tickers: Annotated[TickerRepository, Depends(repository_provider(TickerRepository))],
    prices: Annotated[DailyPriceRepository, Depends(repository_provider(DailyPriceRepository))],
) -> PortfolioService:
    """Assemble the portfolio service."""
    return PortfolioService(portfolios=portfolios, tickers=tickers, prices=prices)


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_db_session)]
MongoDep = Annotated[MongoDatabase, Depends(get_mongo)]
HealthServiceDep = Annotated[HealthService, Depends(get_health_service)]
MarketDataDep = Annotated[MarketDataService, Depends(get_market_data)]
UniverseSyncDep = Annotated[UniverseSyncService, Depends(get_universe_sync_service)]
ReportServiceDep = Annotated[CompanyReportService, Depends(get_report_service)]
GraphServiceDep = Annotated[GraphService, Depends(get_graph_service)]
NewsRepoDep = Annotated[NewsRepository, Depends(get_news_repository)]
PriceIngestionDep = Annotated[PriceIngestionService, Depends(get_price_ingestion_service)]
FeatureServiceDep = Annotated[FeatureEngineeringService, Depends(get_feature_service)]
AnomalyServiceDep = Annotated[AnomalyDetectionService, Depends(get_anomaly_service)]
AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
CurrentUserDep = Annotated[User, Depends(get_current_user)]
WatchlistServiceDep = Annotated[WatchlistService, Depends(get_watchlist_service)]
PortfolioServiceDep = Annotated[PortfolioService, Depends(get_portfolio_service)]
ChunkRepoDep = Annotated[RagChunkRepository, Depends(get_chunk_repository)]
IndexingServiceDep = Annotated[DocumentIndexingService, Depends(get_indexing_service)]
SearchServiceDep = Annotated[HybridSearchService, Depends(get_search_service)]
RagServiceDep = Annotated[RagService, Depends(get_rag_service)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
CorrelationEngineDep = Annotated[CorrelationEngine, Depends(get_correlation_engine)]

# Repository dependencies. Endpoints and services annotate with these aliases
# rather than constructing repositories, so a test can swap any one of them for
# a fake through `app.dependency_overrides`.
CompanyRepoDep = Annotated[CompanyRepository, Depends(repository_provider(CompanyRepository))]
ListingRepoDep = Annotated[ListingRepository, Depends(repository_provider(ListingRepository))]
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
