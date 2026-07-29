"""Endpoint tests using dependency overrides.

Repositories are replaced with in-memory fakes, so these verify the HTTP layer
in isolation: routing, query validation, error translation and the response
contract. Repository behaviour is covered by the integration suite against real
PostgreSQL -- testing it twice would only make the fakes drift.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_news_repository, repository_provider
from app.core.config import Environment, Settings
from app.main import create_app
from app.models.enums import AssetType, DataSource
from app.repositories import CompanyRepository, DailyPriceRepository, TickerRepository
from app.schemas.documents import NewsArticle

TODAY = date(2026, 7, 29)


# --- Fakes -----------------------------------------------------------------


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


# --- Companies -------------------------------------------------------------


class TestCompanyEndpoints:
    """Listing, filtering and detail retrieval."""

    async def test_list_returns_the_tracked_universe(self, market_client: AsyncClient) -> None:
        response = await market_client.get("/api/v1/companies")

        assert response.status_code == 200
        slugs = [company["slug"] for company in response.json()]
        assert slugs == ["micron", "nvidia"]

    async def test_filter_by_tag(self, market_client: AsyncClient) -> None:
        response = await market_client.get("/api/v1/companies", params={"tags": "hbm"})

        assert response.status_code == 200
        assert [c["slug"] for c in response.json()] == ["micron"]

    async def test_unknown_tag_is_rejected_by_validation(self, market_client: AsyncClient) -> None:
        """The enum is the contract; a typo must not silently return everything."""
        response = await market_client.get("/api/v1/companies", params={"tags": "quantum"})

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "request_validation_error"

    async def test_search_by_name(self, market_client: AsyncClient) -> None:
        response = await market_client.get("/api/v1/companies", params={"search": "micron"})

        assert [c["slug"] for c in response.json()] == ["micron"]

    async def test_detail_includes_listings(self, market_client: AsyncClient) -> None:
        response = await market_client.get("/api/v1/companies/nvidia")

        assert response.status_code == 200
        body = response.json()
        assert body["slug"] == "nvidia"
        assert [t["symbol"] for t in body["tickers"]] == ["NVDA"]
        assert body["description"]

    async def test_unknown_company_returns_the_error_envelope(
        self, market_client: AsyncClient
    ) -> None:
        response = await market_client.get("/api/v1/companies/does-not-exist")

        assert response.status_code == 404
        error = response.json()["error"]
        assert error["code"] == "not_found"
        assert error["request_id"]


class TestTickerEndpoints:
    """Listing and lookup of tradable instruments."""

    async def test_list_all(self, market_client: AsyncClient) -> None:
        response = await market_client.get("/api/v1/tickers")

        assert [t["symbol"] for t in response.json()] == ["MU", "NVDA", "SMH"]

    async def test_filter_by_asset_type(self, market_client: AsyncClient) -> None:
        response = await market_client.get("/api/v1/tickers", params={"asset_type": "etf"})

        assert [t["symbol"] for t in response.json()] == ["SMH"]

    async def test_lookup_is_case_insensitive(self, market_client: AsyncClient) -> None:
        response = await market_client.get("/api/v1/tickers/nvda")

        assert response.status_code == 200
        assert response.json()["symbol"] == "NVDA"

    async def test_unknown_ticker_returns_404(self, market_client: AsyncClient) -> None:
        response = await market_client.get("/api/v1/tickers/ZZZZ")

        assert response.status_code == 404


class TestPriceEndpoints:
    """Series retrieval, window validation and quote computation."""

    async def test_series_returns_bars_and_a_period_return(
        self, market_client: AsyncClient
    ) -> None:
        response = await market_client.get("/api/v1/prices/NVDA")

        assert response.status_code == 200
        body = response.json()
        assert body["symbol"] == "NVDA"
        assert body["count"] == len(body["bars"]) == 5
        # Closes run 104 -> 100 over the window, so the period return is negative.
        assert float(body["period_return"]) < 0

    async def test_window_is_honoured(self, market_client: AsyncClient) -> None:
        response = await market_client.get(
            "/api/v1/prices/NVDA",
            params={"start": str(TODAY - timedelta(days=1)), "end": str(TODAY)},
        )

        assert response.json()["count"] == 2

    async def test_inverted_window_is_rejected(self, market_client: AsyncClient) -> None:
        response = await market_client.get(
            "/api/v1/prices/NVDA",
            params={"start": str(TODAY), "end": str(TODAY - timedelta(days=10))},
        )

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    async def test_oversized_window_is_rejected(self, market_client: AsyncClient) -> None:
        """An unbounded window would be an unbounded scan."""
        response = await market_client.get(
            "/api/v1/prices/NVDA",
            params={"start": "2000-01-01", "end": str(TODAY)},
        )

        assert response.status_code == 422
        assert "max_days" in response.json()["error"]["details"]

    async def test_latest_quote_computes_the_session_change(
        self, market_client: AsyncClient
    ) -> None:
        response = await market_client.get("/api/v1/prices/NVDA/latest")

        assert response.status_code == 200
        body = response.json()
        assert body["close"] == "100.000000" or float(body["close"]) == 100
        assert float(body["change_percent"]) < 0

    async def test_prices_for_unknown_ticker_return_404(self, market_client: AsyncClient) -> None:
        response = await market_client.get("/api/v1/prices/ZZZZ")

        assert response.status_code == 404


class TestNewsEndpoints:
    """News listing, filtering and the response contract."""

    async def test_list_returns_articles_without_bodies(self, market_client: AsyncClient) -> None:
        """The body is for embedding, not for the feed -- it would dominate the payload."""
        response = await market_client.get("/api/v1/news")

        assert response.status_code == 200
        article = response.json()[0]
        assert article["title"] == "Micron raises HBM guidance"
        assert "content" not in article

    async def test_filter_by_ticker(self, market_client: AsyncClient) -> None:
        matching = await market_client.get("/api/v1/news", params={"tickers": "MU"})
        other = await market_client.get("/api/v1/news", params={"tickers": "NVDA"})

        assert len(matching.json()) == 1
        assert other.json() == []

    async def test_lookback_window_is_bounded(self, market_client: AsyncClient) -> None:
        response = await market_client.get("/api/v1/news", params={"days": 500})

        assert response.status_code == 422

    async def test_search_requires_a_meaningful_term(self, market_client: AsyncClient) -> None:
        response = await market_client.get("/api/v1/news/search", params={"q": "a"})

        assert response.status_code == 422

    async def test_search_matches_titles(self, market_client: AsyncClient) -> None:
        response = await market_client.get("/api/v1/news/search", params={"q": "HBM"})

        assert len(response.json()) == 1


class TestOpenApi:
    """The published contract."""

    async def test_every_router_is_documented(self, market_client: AsyncClient) -> None:
        response = await market_client.get("/api/v1/openapi.json")

        paths = response.json()["paths"]
        assert "/api/v1/companies" in paths
        assert "/api/v1/tickers" in paths
        assert "/api/v1/prices/{symbol}" in paths
        assert "/api/v1/news" in paths
        assert "/api/v1/ingestion/prices" in paths
