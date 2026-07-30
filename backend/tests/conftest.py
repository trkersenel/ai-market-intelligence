"""Shared pytest fixtures.

Unit tests must run with no database, no network and no Docker -- otherwise the
suite stops being run. Infrastructure adapters are therefore replaced with test
doubles through ``app.dependency_overrides``, which is the payoff of routing all
construction through :mod:`app.api.deps`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_health_service, get_news_repository, repository_provider
from app.core.config import Environment, Settings
from app.main import create_app
from app.models.enums import AssetType, DataSource
from app.repositories import CompanyRepository, DailyPriceRepository, TickerRepository
from app.schemas.documents import NewsArticle
from app.schemas.health import DependencyHealth, DependencyStatus, ReadinessResponse


class StubHealthService:
    """Health service double returning a scripted readiness report."""

    def __init__(self, *, healthy: bool = True) -> None:
        """Configure whether the stub reports dependencies as up or down."""
        self._healthy = healthy

    async def check_readiness(self) -> ReadinessResponse:
        """Return a deterministic readiness report."""
        status = DependencyStatus.UP if self._healthy else DependencyStatus.DOWN
        return ReadinessResponse(
            status=status,
            checked_at=datetime.now(UTC),
            dependencies=[
                DependencyHealth(
                    name=name,
                    status=status,
                    latency_ms=1.0,
                    error=None if self._healthy else "connection refused",
                )
                for name in ("postgres", "mongodb")
            ],
        )


def healthy_health_service() -> StubHealthService:
    """Dependency override factory.

    A plain function rather than the class itself: FastAPI would otherwise
    inspect ``StubHealthService.__init__`` and expose ``healthy`` as a query
    parameter.
    """
    return StubHealthService()


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Settings pinned to the test environment with console-free JSON logs."""
    return Settings(
        environment=Environment.TEST,
        debug=True,
        cors_origins=["http://testserver"],
    )


@pytest.fixture
def app(settings: Settings) -> Iterator[FastAPI]:
    """Application instance with a healthy health-service double installed."""
    application = create_app(settings)
    application.dependency_overrides[get_health_service] = healthy_health_service
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """HTTP client speaking directly to the ASGI app -- no socket, no server.

    ``lifespan`` is not triggered by ``ASGITransport``, so no real database
    connection is ever opened by the unit suite.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client


# --- Market API fixtures ---------------------------------------------------

#: Anchor date for the synthetic price series the fakes serve.
TODAY = date(2026, 7, 29)
#
# Shared by the endpoint and metrics suites. A fixture imported from another
# test module works but couples the two files; conftest is where pytest
# expects to find it.


class FakeCompany:
    """Minimal stand-in with the attributes the schemas read."""

    def __init__(self, company_id: int, slug: str, name: str, tags: list[str]) -> None:
        self.id = company_id
        self.slug = slug
        self.name = name
        self.sector = "Information Technology"
        self.industry = "Semiconductors"
        self.country = "US"
        self.website = "https://example.test"
        self.description = "A tracked company."
        self.tags = tags
        self.is_tracked = True
        self.tickers: list[FakeTicker] = []


class FakeTicker:
    """Minimal stand-in for a listing."""

    def __init__(
        self, ticker_id: int, symbol: str, asset_type: AssetType = AssetType.EQUITY
    ) -> None:
        self.id = ticker_id
        self.symbol = symbol
        self.display_name = f"{symbol} Inc."
        self.exchange = "NASDAQ"
        self.currency = "USD"
        self.asset_type = asset_type
        self.is_active = True
        self.last_price_date = TODAY


class FakePrice:
    """Minimal stand-in for a daily bar."""

    def __init__(self, trade_date: date, close: str) -> None:
        price = Decimal(close)
        self.trade_date = trade_date
        self.open = price
        self.high = price + 1
        self.low = price - 1
        self.close = price
        self.adjusted_close = price
        self.volume = 1_000_000


NVIDIA = FakeCompany(1, "nvidia", "NVIDIA", ["gpu", "networking"])
MICRON = FakeCompany(2, "micron", "Micron Technology", ["hbm", "dram"])
NVDA = FakeTicker(1, "NVDA")
MU = FakeTicker(2, "MU")
SMH = FakeTicker(3, "SMH", AssetType.ETF)
NVIDIA.tickers = [NVDA]
MICRON.tickers = [MU]


class FakeCompanyRepository:
    """In-memory company repository."""

    async def list_tracked(self) -> list[FakeCompany]:
        return [MICRON, NVIDIA]

    async def get_by_slug(self, slug: str) -> FakeCompany | None:
        return {"nvidia": NVIDIA, "micron": MICRON}.get(slug)

    async def get_with_tickers(self, company_id: int) -> FakeCompany | None:
        return {1: NVIDIA, 2: MICRON}.get(company_id)

    async def list_by_tags(
        self, tags: Sequence[Any], *, match_all: bool = False
    ) -> list[FakeCompany]:
        wanted = {tag.value for tag in tags}
        matches = [
            company
            for company in (MICRON, NVIDIA)
            if (wanted <= set(company.tags) if match_all else wanted & set(company.tags))
        ]
        return matches

    async def search(self, term: str, *, limit: int = 20) -> list[FakeCompany]:
        lowered = term.lower()
        return [c for c in (MICRON, NVIDIA) if lowered in c.name.lower() or lowered in c.slug]


class FakeTickerRepository:
    """In-memory ticker repository."""

    async def list_active(self, *, asset_type: AssetType | None = None) -> list[FakeTicker]:
        listings = [MU, NVDA, SMH]
        if asset_type is not None:
            listings = [t for t in listings if t.asset_type is asset_type]
        return listings

    async def get_by_symbol(self, symbol: str) -> FakeTicker | None:
        return {"NVDA": NVDA, "MU": MU, "SMH": SMH}.get(symbol.strip().upper())


class FakePriceRepository:
    """In-memory price repository returning a deterministic series."""

    def __init__(self) -> None:
        self.bars = [
            FakePrice(TODAY - timedelta(days=offset), str(100 + offset))
            for offset in reversed(range(5))
        ]

    async def get_range(self, ticker_id: int, *, start: date, end: date) -> list[FakePrice]:
        return [bar for bar in self.bars if start <= bar.trade_date <= end]

    async def get_recent(self, ticker_id: int, *, sessions: int) -> list[FakePrice]:
        return self.bars[-sessions:]


class FakeNewsRepository:
    """In-memory news repository."""

    def __init__(self) -> None:
        self.articles = [
            NewsArticle(
                _id="64f0c0ffee",
                url_hash="abc123",
                url="https://x.test/mu-hbm",
                title="Micron raises HBM guidance",
                summary="Capacity sold out through next year.",
                content="Long body that must not be returned by the API.",
                source=DataSource.NEWSAPI,
                source_name="Reuters",
                published_at=datetime(2026, 7, 28, tzinfo=UTC),
                ingested_at=datetime(2026, 7, 28, 1, tzinfo=UTC),
                tickers=["MU"],
                tags=["hbm", "dram"],
            )
        ]

    async def list_recent(self, **kwargs: Any) -> list[NewsArticle]:
        tickers = kwargs.get("tickers")
        if tickers:
            wanted = {t.upper() for t in tickers}
            return [a for a in self.articles if wanted & set(a.tickers)]
        return list(self.articles)

    async def search_text(self, term: str, *, limit: int = 25) -> list[NewsArticle]:
        return [a for a in self.articles if term.lower() in a.title.lower()]


# --- Fixtures --------------------------------------------------------------


@pytest.fixture
def market_app() -> FastAPI:
    """App with every data dependency replaced by an in-memory fake."""
    application = create_app(Settings(environment=Environment.TEST, debug=True))
    # `repository_provider` is cached, so these are the same callable objects
    # the routes declared -- which is what makes the override match.
    overrides = {
        repository_provider(CompanyRepository): FakeCompanyRepository,
        repository_provider(TickerRepository): FakeTickerRepository,
        repository_provider(DailyPriceRepository): FakePriceRepository,
        get_news_repository: FakeNewsRepository,
    }
    for dependency, fake in overrides.items():
        application.dependency_overrides[dependency] = fake
    return application


@pytest.fixture
async def market_client(market_app: FastAPI) -> AsyncIterator[AsyncClient]:
    """HTTP client bound to the fake-backed app."""
    transport = ASGITransport(app=market_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
