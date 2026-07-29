"""News ingestion: fetch from every provider, tag, deduplicate, store.

The pipeline is deliberately ordered fetch → tag → filter → store. Tagging
before storing means the deduplication key is computed once and irrelevant
articles are dropped before they cost a write, an embedding and a place in the
retrieval index.

Providers are polled independently and their failures are contained: NewsAPI
being down or out of quota must still leave RSS and SEC ingestion working.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.clients.protocols import NewsProvider, RawArticle
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.repositories.company import CompanyRepository, TickerRepository
from app.repositories.documents import NewsRepository
from app.schemas.documents import NewsArticle
from app.services.tagging import ArticleTagger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ProviderResult:
    """Outcome of polling one news provider."""

    provider: str
    fetched: int = 0
    stored: int = 0
    duplicates: int = 0
    irrelevant: int = 0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the provider responded without an error."""
        return self.error is None


@dataclass(frozen=True)
class NewsIngestionReport:
    """Aggregate outcome of one news ingestion run."""

    started_at: datetime
    finished_at: datetime
    results: tuple[ProviderResult, ...]

    @property
    def stored(self) -> int:
        """Articles newly written across every provider."""
        return sum(result.stored for result in self.results)

    @property
    def failures(self) -> tuple[ProviderResult, ...]:
        """Providers that failed."""
        return tuple(result for result in self.results if not result.succeeded)


class NewsIngestionService:
    """Polls every configured news provider and stores relevant articles."""

    def __init__(
        self,
        *,
        providers: Sequence[NewsProvider],
        news: NewsRepository,
        companies: CompanyRepository,
        tickers: TickerRepository,
        default_lookback_hours: int = 24,
    ) -> None:
        """Wire the service to its collaborators.

        Args:
            providers: Anything satisfying :class:`NewsProvider`.
            news: Document repository for articles.
            companies: Used to build the slug-to-symbol map for tagging.
            tickers: Used to resolve company slugs to tradable symbols.
            default_lookback_hours: Window polled when no explicit ``since`` is
                given. Wider than the hourly schedule on purpose, so a missed
                run self-heals on the next one instead of leaving a gap.
        """
        self._providers = list(providers)
        self._news = news
        self._companies = companies
        self._tickers = tickers
        self._default_lookback = timedelta(hours=default_lookback_hours)

    async def ingest_all(self, *, since: datetime | None = None) -> NewsIngestionReport:
        """Poll every provider and store what is relevant and new."""
        started = datetime.now(UTC)
        window_start = since or (started - self._default_lookback)
        tagger = await self._build_tagger()

        results = await asyncio.gather(
            *(self._ingest_provider(provider, window_start, tagger) for provider in self._providers)
        )

        report = NewsIngestionReport(
            started_at=started, finished_at=datetime.now(UTC), results=tuple(results)
        )
        logger.info(
            "news_ingestion_complete",
            providers=len(self._providers),
            stored=report.stored,
            failures=len(report.failures),
        )
        return report

    async def _ingest_provider(
        self, provider: NewsProvider, since: datetime, tagger: ArticleTagger
    ) -> ProviderResult:
        """Poll one provider, converting any failure into a result."""
        try:
            raw = await provider.fetch_articles(since=since)
        except AppError as exc:
            logger.warning("news_provider_failed", provider=provider.name, error=exc.message)
            return ProviderResult(provider=provider.name, error=exc.message)

        relevant: list[NewsArticle] = []
        irrelevant = 0
        for item in raw:
            article = self._normalise(item, tagger)
            if article is None:
                irrelevant += 1
                continue
            relevant.append(article)

        stored, duplicates = await self._news.bulk_upsert(relevant)
        return ProviderResult(
            provider=provider.name,
            fetched=len(raw),
            stored=stored,
            duplicates=duplicates,
            irrelevant=irrelevant,
        )

    def _normalise(self, raw: RawArticle, tagger: ArticleTagger) -> NewsArticle | None:
        """Tag an article and convert it, or drop it as irrelevant.

        Returns ``None`` when neither the source nor the tagger can attach the
        article to anything tracked. Storing those would inflate the retrieval
        index with material the platform can never usefully cite.

        A source-asserted ticker outranks the tagger. A per-symbol feed has
        already established what the article is about, and an analysis of
        Micron's quarter that never writes "Micron" is exactly the article worth
        keeping -- and exactly the one keyword matching drops.
        """
        tags = tagger.tag(raw.searchable_text)
        asserted = [symbol.strip().upper() for symbol in raw.tickers if symbol.strip()]
        if not tags.is_relevant and not asserted:
            return None

        tickers = list(dict.fromkeys([*asserted, *tags.tickers]))

        return NewsArticle(
            url_hash=NewsArticle.hash_url(raw.url),
            url=raw.url,
            title=raw.title,
            summary=raw.summary,
            content=raw.content,
            source=raw.source,
            source_name=raw.source_name,
            author=raw.author,
            published_at=raw.published_at,
            ingested_at=datetime.now(UTC),
            tickers=tickers,
            company_slugs=list(tags.company_slugs),
            keywords=list(tags.keywords),
            tags=list(tags.tags),
            language=raw.language,
        )

    async def _build_tagger(self) -> ArticleTagger:
        """Construct a tagger that knows the current slug-to-symbol mapping.

        Built per run rather than cached: adding a listing should affect the
        next ingestion, not require a restart.
        """
        companies = await self._companies.list_tracked()
        listings = await self._tickers.list_active()

        slug_by_id = {company.id: company.slug for company in companies}
        symbols_by_slug: dict[str, list[str]] = {}
        for listing in listings:
            slug = slug_by_id.get(listing.company_id or -1)
            if slug is not None:
                symbols_by_slug.setdefault(slug, []).append(listing.symbol)

        return ArticleTagger(symbols_by_slug=symbols_by_slug)
