"""Tests for the news providers, driven by recorded responses.

The fixtures are real response shapes with the awkward cases left in --
NewsAPI's ``[Removed]`` placeholder, an RSS item with no author, an entry dated
years before the window. Those are exactly the rows that break a parser in
production, so a fixture that omits them tests nothing worth testing.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest

from app.clients.news_clients import NewsApiProvider, RssProvider, SecFilingsProvider
from app.core.config import IngestionSettings
from app.core.exceptions import ExternalServiceError
from app.models.enums import DataSource

FIXTURES = Path(__file__).parent / "fixtures"
SINCE = datetime(2026, 7, 27, tzinfo=UTC)


def _settings(**overrides: object) -> IngestionSettings:
    """Ingestion settings with fast retries for tests."""
    return IngestionSettings(
        max_retries=1,
        retry_backoff_seconds=0.001,
        newsapi_key="test-key",
        **overrides,  # type: ignore[arg-type]
    )


def _transport(response: httpx.Response) -> httpx.MockTransport:
    """A transport that returns the same response for every request."""
    return httpx.MockTransport(lambda _: response)


class TestNewsApiProvider:
    """NewsAPI parsing and its failure modes."""

    def _provider(self, transport: httpx.MockTransport) -> NewsApiProvider:
        return NewsApiProvider(
            _settings(),
            client=httpx.AsyncClient(transport=transport, base_url="https://newsapi.test"),
        )

    async def test_articles_are_normalised(self) -> None:
        payload = json.loads((FIXTURES / "newsapi_everything.json").read_text())
        provider = self._provider(_transport(httpx.Response(200, json=payload)))

        articles = await provider.fetch_articles(since=SINCE)

        # Four in the fixture, one is the [Removed] placeholder.
        assert len(articles) == 3
        first = articles[0]
        assert first.title.startswith("Micron raises HBM output")
        assert first.source is DataSource.NEWSAPI
        assert first.source_name == "Reuters"
        assert first.published_at.tzinfo is not None

    async def test_removed_placeholder_articles_are_dropped(self) -> None:
        payload = json.loads((FIXTURES / "newsapi_everything.json").read_text())
        provider = self._provider(_transport(httpx.Response(200, json=payload)))

        articles = await provider.fetch_articles(since=SINCE)

        assert all(article.title != "[Removed]" for article in articles)

    async def test_api_level_error_is_surfaced(self) -> None:
        """NewsAPI returns HTTP 200 with a failure body when the quota runs out."""
        body = {
            "status": "error",
            "code": "rateLimited",
            "message": "You have made too many requests",
        }
        provider = self._provider(_transport(httpx.Response(200, json=body)))

        with pytest.raises(ExternalServiceError, match="too many requests"):
            await provider.fetch_articles(since=SINCE)

    async def test_missing_credential_is_detectable_before_construction(self) -> None:
        settings = IngestionSettings(newsapi_key=None)

        assert NewsApiProvider.is_configured(settings) is False
        with pytest.raises(ExternalServiceError, match="not configured"):
            NewsApiProvider(settings)

    async def test_the_api_key_is_sent_as_a_header_not_a_query_parameter(self) -> None:
        """A key in the query string leaks into access logs and referrers."""
        captured: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(request.headers)
            captured["_url"] = str(request.url)
            return httpx.Response(200, json={"status": "ok", "articles": []})

        provider = self._provider(httpx.MockTransport(handler))
        await provider.fetch_articles(since=SINCE)

        assert captured["x-api-key"] == "test-key"
        assert "test-key" not in captured["_url"]


class TestRssProvider:
    """Feed parsing, date handling and per-feed failure isolation."""

    def _provider(
        self, transport: httpx.MockTransport, feeds: list[str] | None = None
    ) -> RssProvider:
        return RssProvider(
            _settings(rss_feeds=feeds or ["https://feed.test/rss"]),
            client=httpx.AsyncClient(transport=transport),
        )

    async def test_entries_within_the_window_are_returned(self) -> None:
        feed = (FIXTURES / "semiconductor_feed.xml").read_text()
        provider = self._provider(_transport(httpx.Response(200, text=feed)))

        articles = await provider.fetch_articles(since=SINCE)

        titles = [article.title for article in articles]
        assert "SK Hynix begins HBM4 sampling ahead of schedule" in titles
        assert "ASML books record High-NA EUV orders" in titles

    async def test_entries_older_than_the_window_are_excluded(self) -> None:
        """The 2020 item must not resurface as if it were breaking news."""
        feed = (FIXTURES / "semiconductor_feed.xml").read_text()
        provider = self._provider(_transport(httpx.Response(200, text=feed)))

        articles = await provider.fetch_articles(since=SINCE)

        assert all(article.published_at >= SINCE for article in articles)
        assert "stale-dram-item" not in " ".join(a.url for a in articles)

    async def test_results_are_newest_first(self) -> None:
        feed = (FIXTURES / "semiconductor_feed.xml").read_text()
        provider = self._provider(_transport(httpx.Response(200, text=feed)))

        articles = await provider.fetch_articles(since=SINCE)

        timestamps = [article.published_at for article in articles]
        assert timestamps == sorted(timestamps, reverse=True)

    async def test_a_failing_feed_does_not_abort_the_others(self) -> None:
        """One publisher's outage must not cost the other feeds' articles."""
        feed = (FIXTURES / "semiconductor_feed.xml").read_text()

        def handler(request: httpx.Request) -> httpx.Response:
            if "broken" in str(request.url):
                return httpx.Response(404)
            return httpx.Response(200, text=feed)

        provider = self._provider(
            httpx.MockTransport(handler),
            feeds=["https://broken.test/rss", "https://working.test/rss"],
        )

        articles = await provider.fetch_articles(since=SINCE)

        assert articles, "the working feed's articles should still be returned"

    async def test_published_timestamps_are_timezone_aware(self) -> None:
        feed = (FIXTURES / "semiconductor_feed.xml").read_text()
        provider = self._provider(_transport(httpx.Response(200, text=feed)))

        articles = await provider.fetch_articles(since=SINCE)

        assert all(article.published_at.tzinfo is not None for article in articles)


class TestSecFilingsProvider:
    """EDGAR's column-oriented payload and its required headers."""

    SUBMISSIONS: ClassVar[dict[str, Any]] = {
        "name": "MICRON TECHNOLOGY INC",
        "filings": {
            "recent": {
                "form": ["8-K", "10-Q", "4"],
                # The last entry predates the window and must be excluded.
                "filingDate": ["2026-07-28", "2026-07-27", "2020-01-05"],
                "accessionNumber": [
                    "0000723125-26-000045",
                    "0000723125-26-000040",
                    "0000723125-20-000001",
                ],
                "primaryDocument": ["mu-8k.htm", "mu-10q.htm", "mu-4.xml"],
                "primaryDocDescription": ["8-K", "10-Q", "FORM 4"],
                "cik": [723125, 723125, 723125],
            }
        },
    }

    def _provider(self, transport: httpx.MockTransport) -> SecFilingsProvider:
        return SecFilingsProvider(
            _settings(),
            client=httpx.AsyncClient(transport=transport, base_url="https://sec.test"),
        )

    async def test_recent_filings_are_converted_to_articles(self) -> None:
        provider = self._provider(_transport(httpx.Response(200, json=self.SUBMISSIONS)))

        filings = await provider.fetch_filings("723125", since=SINCE)

        assert len(filings) == 2  # the 2020 Form 4 is outside the window
        assert filings[0].title == "MICRON TECHNOLOGY INC filed 8-K"
        assert filings[0].source is DataSource.SEC_EDGAR
        assert filings[0].url.startswith("https://www.sec.gov/Archives/edgar/data/723125/")

    async def test_cik_is_zero_padded_to_ten_digits(self) -> None:
        """EDGAR 404s on an unpadded CIK, which is easy to get wrong by hand."""
        captured: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(str(request.url))
            return httpx.Response(200, json=self.SUBMISSIONS)

        provider = self._provider(httpx.MockTransport(handler))
        await provider.fetch_filings("723125", since=SINCE)

        assert "CIK0000723125.json" in captured[0]

    async def test_a_descriptive_user_agent_is_sent(self) -> None:
        """SEC blocks anonymous clients outright."""
        captured: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(request.headers)
            return httpx.Response(200, json=self.SUBMISSIONS)

        provider = self._provider(httpx.MockTransport(handler))
        await provider.fetch_filings("723125", since=SINCE)

        assert "AI Market Intelligence Platform" in captured["user-agent"]

    async def test_malformed_payload_is_reported(self) -> None:
        provider = self._provider(_transport(httpx.Response(200, json=["unexpected"])))

        with pytest.raises(ExternalServiceError, match="malformed"):
            await provider.fetch_filings("723125", since=SINCE)
