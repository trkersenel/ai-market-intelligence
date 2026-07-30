"""Endpoint tests using dependency overrides.

Repositories are replaced with in-memory fakes, so these verify the HTTP layer
in isolation: routing, query validation, error translation and the response
contract. Repository behaviour is covered by the integration suite against real
PostgreSQL -- testing it twice would only make the fakes drift.
"""

from __future__ import annotations

from datetime import date, timedelta

from httpx import AsyncClient

TODAY = date(2026, 7, 29)


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
