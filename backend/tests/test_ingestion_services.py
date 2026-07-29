"""Tests for the ingestion services.

The services are exercised with fake providers and fake repositories. That is
the whole payoff of depending on protocols: the policy under test -- window
selection, concurrency limits, failure isolation, relevance filtering -- is
verified without a network, a database, or a fixture that takes a second to
build.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from app.clients.protocols import PriceBar, RawArticle
from app.core.config import IngestionSettings
from app.core.exceptions import ExternalServiceError, NotFoundError
from app.models.enums import DataSource
from app.schemas.documents import NewsArticle
from app.services.ingestion import NewsIngestionService, PriceIngestionService

TODAY = date(2026, 7, 29)
NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


# --- Fakes -----------------------------------------------------------------


class FakeTicker:
    """Stands in for the Ticker ORM model."""

    def __init__(self, ticker_id: int, symbol: str, last_price_date: date | None = None) -> None:
        self.id = ticker_id
        self.symbol = symbol
        self.last_price_date = last_price_date
        self.first_price_date: date | None = None
        self.last_ingested_at: datetime | None = None


class FakeTickerRepository:
    """Records watermark updates instead of writing them."""

    def __init__(self, tickers: Sequence[FakeTicker]) -> None:
        self._tickers = list(tickers)
        self.watermarks: dict[int, tuple[date | None, date]] = {}

    async def list_active(self, *, asset_type: object = None) -> list[FakeTicker]:
        return list(self._tickers)

    async def get_by_symbol(self, symbol: str) -> FakeTicker | None:
        wanted = symbol.strip().upper()
        return next((t for t in self._tickers if t.symbol == wanted), None)

    async def update_watermarks(
        self,
        ticker_id: int,
        *,
        first_price_date: date | None,
        last_price_date: date,
        ingested_at: datetime,
    ) -> None:
        self.watermarks[ticker_id] = (first_price_date, last_price_date)


class FakePriceRepository:
    """Collects the rows it was asked to upsert."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    async def bulk_upsert(self, rows: Sequence[dict[str, Any]]) -> int:
        self.rows.extend(rows)
        return len(rows)


class FakePriceProvider:
    """Returns scripted bars, and records the windows it was asked for."""

    def __init__(
        self,
        bars_by_symbol: dict[str, list[PriceBar]] | None = None,
        failing_symbols: set[str] | None = None,
    ) -> None:
        self._bars = bars_by_symbol or {}
        self._failing = failing_symbols or set()
        self.requests: list[tuple[str, date, date]] = []
        self.max_concurrent = 0
        self._in_flight = 0

    @property
    def source(self) -> DataSource:
        return DataSource.YFINANCE

    async def fetch_daily_bars(self, symbol: str, *, start: date, end: date) -> list[PriceBar]:
        self.requests.append((symbol, start, end))
        self._in_flight += 1
        self.max_concurrent = max(self.max_concurrent, self._in_flight)
        try:
            await asyncio.sleep(0)  # yield, so concurrency is observable
            if symbol in self._failing:
                msg = f"provider is down for {symbol}"
                raise ExternalServiceError(msg)
            return self._bars.get(symbol, [])
        finally:
            self._in_flight -= 1


class FakeNewsProvider:
    """Returns scripted articles, or fails."""

    def __init__(
        self, name: str, articles: list[RawArticle] | None = None, fails: bool = False
    ) -> None:
        self._name = name
        self._articles = articles or []
        self._fails = fails

    @property
    def source(self) -> DataSource:
        return DataSource.RSS

    @property
    def name(self) -> str:
        return self._name

    async def fetch_articles(
        self, *, since: datetime, query: str | None = None, limit: int = 100
    ) -> list[RawArticle]:
        if self._fails:
            msg = f"{self._name} is unavailable"
            raise ExternalServiceError(msg)
        return list(self._articles)


class FakeNewsRepository:
    """Deduplicates on url_hash in memory."""

    def __init__(self) -> None:
        self.stored: dict[str, NewsArticle] = {}

    async def bulk_upsert(self, articles: Sequence[NewsArticle]) -> tuple[int, int]:
        inserted = 0
        for article in articles:
            if article.url_hash not in self.stored:
                self.stored[article.url_hash] = article
                inserted += 1
        return (inserted, len(articles) - inserted)


class FakeCompany:
    """Stands in for the Company ORM model."""

    def __init__(self, company_id: int, slug: str) -> None:
        self.id = company_id
        self.slug = slug


class FakeCompanyRepository:
    """Returns a fixed tracked universe."""

    def __init__(self, companies: Sequence[FakeCompany]) -> None:
        self._companies = list(companies)

    async def list_tracked(self) -> list[FakeCompany]:
        return list(self._companies)


class FakeCompanyTicker(FakeTicker):
    """A listing that knows its parent company."""

    def __init__(self, ticker_id: int, symbol: str, company_id: int | None) -> None:
        super().__init__(ticker_id, symbol)
        self.company_id = company_id


class FakeListingRepository(FakeTickerRepository):
    """Ticker repository whose listings carry a company_id."""


def _bar(day: date, close: str = "100") -> PriceBar:
    price = Decimal(close)
    return PriceBar(
        trade_date=day,
        open=price,
        high=price + 1,
        low=price - 1,
        close=price,
        adjusted_close=price,
        volume=1_000_000,
    )


def _settings(**overrides: object) -> IngestionSettings:
    """Ingestion settings with explicit defaults that any test may override."""
    values: dict[str, object] = {
        "initial_backfill_days": 730,
        "incremental_overlap_days": 5,
        "max_concurrent_fetches": 2,
    }
    values.update(overrides)
    return IngestionSettings(**values)  # type: ignore[arg-type]


def _service(
    provider: FakePriceProvider,
    tickers: FakeTickerRepository,
    prices: FakePriceRepository,
    settings: IngestionSettings | None = None,
) -> PriceIngestionService:
    return PriceIngestionService(
        provider=provider,  # type: ignore[arg-type]
        tickers=tickers,  # type: ignore[arg-type]
        prices=prices,  # type: ignore[arg-type]
        settings=settings or _settings(),
    )


# --- Price ingestion -------------------------------------------------------


class TestPriceIngestionWindows:
    """Window selection is the policy that makes re-runs cheap and safe."""

    async def test_a_new_listing_gets_the_full_backfill(self) -> None:
        tickers = FakeTickerRepository([FakeTicker(1, "NVDA", last_price_date=None)])
        provider = FakePriceProvider()

        await _service(provider, tickers, FakePriceRepository()).ingest_all(as_of=TODAY)

        _, start, end = provider.requests[0]
        assert (end - start).days == 730
        assert end == TODAY

    async def test_a_current_listing_gets_an_overlapping_window(self) -> None:
        """The overlap is what lets vendor corrections land, for free."""
        last_stored = date(2026, 7, 28)
        tickers = FakeTickerRepository([FakeTicker(1, "NVDA", last_price_date=last_stored)])
        provider = FakePriceProvider()

        await _service(provider, tickers, FakePriceRepository()).ingest_all(as_of=TODAY)

        _, start, _ = provider.requests[0]
        assert start == last_stored - timedelta(days=5)
        assert start < last_stored, "window must re-fetch already-stored sessions"

    async def test_a_listing_ahead_of_the_target_date_is_skipped(self) -> None:
        """No request at all when the computed window ends before it starts."""
        tickers = FakeTickerRepository([FakeTicker(1, "NVDA", last_price_date=date(2027, 1, 1))])
        provider = FakePriceProvider()
        settings = _settings(incremental_overlap_days=0)

        report = await _service(provider, tickers, FakePriceRepository(), settings).ingest_all(
            as_of=TODAY
        )

        assert provider.requests == []
        assert report.bars_written == 0


class TestPriceIngestionOutcomes:
    """Persistence, watermarks and failure isolation."""

    async def test_bars_are_written_and_watermarks_advanced(self) -> None:
        bars = [_bar(date(2026, 7, 27)), _bar(date(2026, 7, 28)), _bar(date(2026, 7, 29))]
        tickers = FakeTickerRepository([FakeTicker(1, "NVDA")])
        prices = FakePriceRepository()
        provider = FakePriceProvider({"NVDA": bars})

        report = await _service(provider, tickers, prices).ingest_all(as_of=TODAY)

        assert report.bars_written == 3
        assert len(prices.rows) == 3
        assert prices.rows[0]["ticker_id"] == 1
        assert tickers.watermarks[1] == (date(2026, 7, 27), date(2026, 7, 29))

    async def test_one_failing_symbol_does_not_abandon_the_others(self) -> None:
        """The single most important property of a nightly batch job."""
        tickers = FakeTickerRepository(
            [FakeTicker(1, "NVDA"), FakeTicker(2, "BROKEN"), FakeTicker(3, "MU")]
        )
        provider = FakePriceProvider(
            bars_by_symbol={"NVDA": [_bar(TODAY)], "MU": [_bar(TODAY)]},
            failing_symbols={"BROKEN"},
        )

        report = await _service(provider, tickers, FakePriceRepository()).ingest_all(as_of=TODAY)

        assert report.bars_written == 2
        assert [failure.symbol for failure in report.failures] == ["BROKEN"]
        assert len(report.results) == 3

    async def test_the_current_session_is_marked_provisional(self) -> None:
        """A bar dated today may still be trading; its OHLCV is partial."""
        tickers = FakeTickerRepository([FakeTicker(1, "NVDA")])
        prices = FakePriceRepository()
        bars = [_bar(date(2026, 7, 28)), _bar(TODAY)]
        provider = FakePriceProvider({"NVDA": bars})

        await _service(provider, tickers, prices).ingest_all(as_of=TODAY)

        by_date = {row["trade_date"]: row["is_provisional"] for row in prices.rows}
        assert by_date[date(2026, 7, 28)] is False
        assert by_date[TODAY] is True

    async def test_the_flag_clears_once_the_session_has_closed(self) -> None:
        """Self-healing: tomorrow's overlapping window rewrites today's bar.

        This is the property that makes the conservative marking safe -- no
        separate reconciliation job is needed to clear a stale flag.
        """
        tickers = FakeTickerRepository([FakeTicker(1, "NVDA")])
        prices = FakePriceRepository()
        provider = FakePriceProvider({"NVDA": [_bar(TODAY)]})

        # Run again the next day, when TODAY is a completed session.
        tomorrow = TODAY + timedelta(days=1)
        await _service(provider, tickers, prices).ingest_all(as_of=tomorrow)

        assert prices.rows[-1]["trade_date"] == TODAY
        assert prices.rows[-1]["is_provisional"] is False

    async def test_a_quiet_symbol_is_not_an_error(self) -> None:
        """An empty window -- a holiday week -- must not be reported as failure."""
        tickers = FakeTickerRepository([FakeTicker(1, "QUIET")])

        report = await _service(FakePriceProvider(), tickers, FakePriceRepository()).ingest_all(
            as_of=TODAY
        )

        assert report.failures == ()
        assert report.bars_written == 0
        assert 1 not in tickers.watermarks, "no bars means no watermark advance"

    async def test_concurrency_is_bounded(self) -> None:
        """A backfill must not open one connection per tracked symbol."""
        tickers = FakeTickerRepository([FakeTicker(index, f"SYM{index}") for index in range(1, 9)])
        provider = FakePriceProvider()

        await _service(
            provider, tickers, FakePriceRepository(), _settings(max_concurrent_fetches=2)
        ).ingest_all(as_of=TODAY)

        assert provider.max_concurrent <= 2
        assert len(provider.requests) == 8

    async def test_single_symbol_ingestion(self) -> None:
        tickers = FakeTickerRepository([FakeTicker(1, "MU")])
        provider = FakePriceProvider({"MU": [_bar(TODAY)]})

        result = await _service(provider, tickers, FakePriceRepository()).ingest_symbol(
            "mu", as_of=TODAY
        )

        assert result.succeeded
        assert result.bars_written == 1

    async def test_unknown_symbol_raises_not_found(self) -> None:
        service = _service(FakePriceProvider(), FakeTickerRepository([]), FakePriceRepository())

        with pytest.raises(NotFoundError):
            await service.ingest_symbol("NOPE", as_of=TODAY)


# --- News ingestion --------------------------------------------------------


def _article(title: str, url: str, published_at: datetime | None = None) -> RawArticle:
    return RawArticle(
        url=url,
        title=title,
        summary=None,
        content=None,
        published_at=published_at or NOW,
        source=DataSource.RSS,
        source_name="Test Feed",
    )


def _news_service(
    providers: Sequence[FakeNewsProvider], news: FakeNewsRepository
) -> NewsIngestionService:
    return NewsIngestionService(
        providers=providers,  # type: ignore[arg-type]
        news=news,  # type: ignore[arg-type]
        companies=FakeCompanyRepository([FakeCompany(1, "micron"), FakeCompany(2, "nvidia")]),  # type: ignore[arg-type]
        tickers=FakeListingRepository(  # type: ignore[arg-type]
            [FakeCompanyTicker(1, "MU", 1), FakeCompanyTicker(2, "NVDA", 2)]
        ),
    )


class TestNewsIngestion:
    """Relevance filtering, deduplication and provider isolation."""

    async def test_relevant_articles_are_tagged_and_stored(self) -> None:
        provider = FakeNewsProvider(
            "rss",
            [_article("Micron raises HBM guidance", "https://x.test/mu-hbm")],
        )
        news = FakeNewsRepository()

        report = await _news_service([provider], news).ingest_all(since=NOW - timedelta(days=1))

        assert report.stored == 1
        stored = next(iter(news.stored.values()))
        assert stored.tickers == ["MU"]
        assert "hbm" in stored.tags

    async def test_irrelevant_articles_are_dropped_before_storage(self) -> None:
        provider = FakeNewsProvider(
            "rss",
            [
                _article("Micron raises HBM guidance", "https://x.test/mu"),
                _article("Best houseplants for low light", "https://x.test/plants"),
            ],
        )
        news = FakeNewsRepository()

        report = await _news_service([provider], news).ingest_all()

        assert report.stored == 1
        assert report.results[0].irrelevant == 1

    async def test_the_same_story_from_two_feeds_is_stored_once(self) -> None:
        """The deduplication guarantee: one event must not become two."""
        url = "https://reuters.test/micron-hbm-sold-out"
        first = FakeNewsProvider("rss", [_article("Micron HBM sold out", url)])
        second = FakeNewsProvider("newsapi", [_article("Micron HBM sold out", url)])
        news = FakeNewsRepository()

        report = await _news_service([first, second], news).ingest_all()

        assert len(news.stored) == 1
        assert report.stored == 1

    async def test_a_failing_provider_does_not_stop_the_others(self) -> None:
        working = FakeNewsProvider(
            "rss", [_article("NVIDIA GPU demand accelerates", "https://x.test/nvda")]
        )
        broken = FakeNewsProvider("newsapi", fails=True)
        news = FakeNewsRepository()

        report = await _news_service([working, broken], news).ingest_all()

        assert report.stored == 1
        assert [failure.provider for failure in report.failures] == ["newsapi"]

    async def test_the_url_hash_is_the_deduplication_key(self) -> None:
        provider = FakeNewsProvider("rss", [_article("Micron HBM update", "https://X.TEST/MU-HBM")])
        news = FakeNewsRepository()

        await _news_service([provider], news).ingest_all()

        stored = next(iter(news.stored.values()))
        assert stored.url_hash == NewsArticle.hash_url("https://x.test/mu-hbm")

    async def test_ingested_at_is_recorded_separately_from_published_at(self) -> None:
        """Both timestamps matter: one orders the news, the other audits the run."""
        published = NOW - timedelta(hours=6)
        provider = FakeNewsProvider(
            "rss", [_article("DRAM contract prices rise", "https://x.test/dram", published)]
        )
        news = FakeNewsRepository()

        await _news_service([provider], news).ingest_all()

        stored = next(iter(news.stored.values()))
        assert stored.published_at == published
        assert stored.ingested_at > published
