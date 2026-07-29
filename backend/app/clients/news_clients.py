"""News providers: NewsAPI, RSS feeds and SEC EDGAR.

Three sources with three completely different response shapes, normalised to a
single :class:`~app.clients.protocols.RawArticle` at the boundary. Downstream --
tagging, sentiment, embedding, retrieval -- never learns which feed an item came
from, beyond the ``source`` field kept for provenance.

Each provider degrades independently: a failing RSS feed must not stop NewsAPI
ingestion, so per-source failures are contained rather than propagated.
"""

from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from app.clients.http import HttpClient
from app.clients.protocols import RawArticle
from app.core.config import IngestionSettings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.models.enums import DataSource

logger = get_logger(__name__)

#: Query used against keyword-search providers. Broad enough to catch the
#: ecosystem, narrow enough to keep the free NewsAPI quota useful.
DEFAULT_NEWS_QUERY = (
    "semiconductor OR HBM OR DRAM OR GPU OR NVIDIA OR AMD OR Micron OR TSMC "
    'OR foundry OR "AI infrastructure" OR "data center"'
)


class NewsApiProvider:
    """Keyword search over NewsAPI's ``/everything`` endpoint."""

    def __init__(
        self,
        settings: IngestionSettings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the provider.

        Args:
            settings: Credentials, rate limit and retry configuration.
            client: Injected transport, used by tests to serve recorded responses.

        Raises:
            ExternalServiceError: If no API key is configured. Callers check
                :meth:`is_configured` first rather than relying on this.
        """
        self._settings = settings
        if settings.newsapi_key is None:
            msg = "NewsAPI key is not configured"
            raise ExternalServiceError(msg)

        self._http = HttpClient(
            settings=settings,
            base_url=settings.newsapi_base_url,
            rate_limit=settings.newsapi_rate_limit,
            headers={"X-Api-Key": settings.newsapi_key.get_secret_value()},
            provider="newsapi",
            client=client,
        )

    @staticmethod
    def is_configured(settings: IngestionSettings) -> bool:
        """Return whether a credential is available for this provider."""
        return settings.newsapi_key is not None

    @property
    def source(self) -> DataSource:
        """Provenance recorded on every article."""
        return DataSource.NEWSAPI

    @property
    def name(self) -> str:
        """Human-readable provider name."""
        return "newsapi"

    async def fetch_articles(
        self, *, since: datetime, query: str | None = None, limit: int = 100
    ) -> list[RawArticle]:
        """Return articles matching the query, published at or after ``since``."""
        payload = await self._http.get_json(
            "/everything",
            params={
                "q": query or DEFAULT_NEWS_QUERY,
                "from": since.astimezone(UTC).isoformat(timespec="seconds"),
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": min(limit, 100),
            },
        )

        if not isinstance(payload, dict) or payload.get("status") != "ok":
            message = (
                payload.get("message", "unknown error")
                if isinstance(payload, dict)
                else "malformed response"
            )
            msg = f"NewsAPI rejected the request: {message}"
            raise ExternalServiceError(msg)

        articles = [
            article
            for item in payload.get("articles", [])
            if (article := self._parse(item)) is not None
        ]
        logger.info("newsapi_fetched", articles=len(articles))
        return articles

    def _parse(self, item: dict[str, Any]) -> RawArticle | None:
        """Convert one NewsAPI item, returning ``None`` if it is unusable."""
        url = item.get("url")
        title = item.get("title")
        published = item.get("publishedAt")
        if not url or not title or not published:
            return None

        # NewsAPI marks paywalled or pulled items with this literal title.
        if title.strip() == "[Removed]":
            return None

        try:
            published_at = datetime.fromisoformat(published.replace("Z", "+00:00"))
        except ValueError:
            logger.debug("newsapi_unparseable_date", published=published)
            return None

        return RawArticle(
            url=url,
            title=title.strip(),
            summary=(item.get("description") or None),
            content=(item.get("content") or None),
            published_at=published_at,
            source=self.source,
            source_name=(item.get("source") or {}).get("name"),
            author=item.get("author"),
        )

    async def aclose(self) -> None:
        """Release the transport."""
        await self._http.aclose()


class RssProvider:
    """Reads a fixed set of industry RSS and Atom feeds."""

    def __init__(
        self,
        settings: IngestionSettings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the provider with the feeds from settings."""
        self._settings = settings
        self._feeds = list(settings.rss_feeds)
        self._http = HttpClient(
            settings=settings,
            rate_limit=settings.rss_rate_limit,
            headers={"User-Agent": settings.sec_user_agent},
            provider="rss",
            client=client,
        )

    @property
    def source(self) -> DataSource:
        """Provenance recorded on every article."""
        return DataSource.RSS

    @property
    def name(self) -> str:
        """Human-readable provider name."""
        return "rss"

    async def fetch_articles(
        self, *, since: datetime, query: str | None = None, limit: int = 100
    ) -> list[RawArticle]:
        """Return recent entries across every configured feed.

        A feed that fails is logged and skipped. One publisher's outage must not
        cost the platform the other three feeds' articles.
        """
        articles: list[RawArticle] = []
        for feed_url in self._feeds:
            try:
                body = await self._http.get_text(feed_url)
            except ExternalServiceError as exc:
                logger.warning("rss_feed_failed", feed=feed_url, error=str(exc))
                continue
            articles.extend(self._parse_feed(feed_url, body, since=since))

        articles.sort(key=lambda article: article.published_at, reverse=True)
        logger.info("rss_fetched", articles=len(articles), feeds=len(self._feeds))
        return articles[:limit]

    def _parse_feed(self, feed_url: str, body: str, *, since: datetime) -> list[RawArticle]:
        """Parse one feed body into articles newer than ``since``."""
        # Lazy, as with yfinance: feedparser drags in a large legacy parsing
        # stack that only news ingestion needs.
        import feedparser  # noqa: PLC0415

        parsed = feedparser.parse(body)
        feed_title = getattr(parsed.feed, "title", None)

        articles: list[RawArticle] = []
        for entry in parsed.entries:
            published_at = _entry_timestamp(entry)
            if published_at is None or published_at < since:
                continue

            link = getattr(entry, "link", None)
            title = getattr(entry, "title", None)
            if not link or not title:
                continue

            articles.append(
                RawArticle(
                    url=link,
                    title=title.strip(),
                    summary=getattr(entry, "summary", None),
                    content=_entry_content(entry),
                    published_at=published_at,
                    source=self.source,
                    source_name=feed_title or feed_url,
                    author=getattr(entry, "author", None),
                )
            )
        return articles

    async def aclose(self) -> None:
        """Release the transport."""
        await self._http.aclose()


class SecFilingsProvider:
    """Recent SEC filings for tracked companies, via EDGAR's submissions API.

    Filings are treated as news because that is how they move prices: an 8-K
    announcing a supply agreement is the same kind of evidence as an article
    reporting it, and often precedes one.
    """

    def __init__(
        self,
        settings: IngestionSettings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the provider.

        SEC requires a descriptive User-Agent with contact details; requests
        without one are refused, so it is set as a default header here.
        """
        self._settings = settings
        self._http = HttpClient(
            settings=settings,
            base_url=settings.sec_base_url,
            rate_limit=settings.sec_rate_limit,
            headers={
                "User-Agent": settings.sec_user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
            provider="sec",
            client=client,
        )

    @property
    def source(self) -> DataSource:
        """Provenance recorded on every filing."""
        return DataSource.SEC_EDGAR

    @property
    def name(self) -> str:
        """Human-readable provider name."""
        return "sec_edgar"

    async def fetch_filings(
        self, cik: str, *, since: datetime, limit: int = 50
    ) -> list[RawArticle]:
        """Return a company's recent filings as articles.

        Args:
            cik: Central Index Key, zero-padded to ten digits by this method.
            since: Lower bound on filing date.
            limit: Maximum filings to return.
        """
        padded = cik.strip().lstrip("CIK").zfill(10)
        payload = await self._http.get_json(f"/submissions/CIK{padded}.json")

        if not isinstance(payload, dict):
            msg = "SEC returned a malformed submissions document"
            raise ExternalServiceError(msg, details={"cik": padded})

        company = payload.get("name", padded)
        recent = payload.get("filings", {}).get("recent", {})
        return self._parse_recent(recent, company=company, since=since, limit=limit)

    async def fetch_articles(
        self, *, since: datetime, query: str | None = None, limit: int = 100
    ) -> list[RawArticle]:
        """Satisfy :class:`NewsProvider` for a single CIK passed as ``query``.

        EDGAR has no cross-company feed for an arbitrary universe, so the
        orchestrating service iterates CIKs and calls :meth:`fetch_filings`.
        This adapter exists so the provider still fits the common protocol.
        """
        if not query:
            return []
        return await self.fetch_filings(query, since=since, limit=limit)

    def _parse_recent(
        self,
        recent: dict[str, Any],
        *,
        company: str,
        since: datetime,
        limit: int,
    ) -> list[RawArticle]:
        """Convert EDGAR's column-oriented payload into articles.

        EDGAR returns parallel arrays -- ``form[i]``, ``filingDate[i]``,
        ``accessionNumber[i]`` -- rather than a list of objects, so the arrays
        are zipped back into records here.
        """
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        documents = recent.get("primaryDocument", [])
        descriptions = recent.get("primaryDocDescription", [])
        ciks = recent.get("cik", [])

        articles: list[RawArticle] = []
        for index, form in enumerate(forms):
            if len(articles) >= limit:
                break
            try:
                filed_at = datetime.fromisoformat(dates[index]).replace(tzinfo=UTC)
            except (IndexError, ValueError):
                continue
            if filed_at < since:
                continue

            accession = accessions[index].replace("-", "") if index < len(accessions) else ""
            document = documents[index] if index < len(documents) else ""
            cik_value = str(ciks[index]).lstrip("0") if index < len(ciks) else ""
            url = f"https://www.sec.gov/Archives/edgar/data/{cik_value}/{accession}/{document}"
            description = descriptions[index] if index < len(descriptions) else ""

            articles.append(
                RawArticle(
                    url=url,
                    title=f"{company} filed {form}",
                    summary=description or f"SEC form {form}",
                    published_at=filed_at,
                    source=self.source,
                    source_name="SEC EDGAR",
                )
            )
        return articles

    async def aclose(self) -> None:
        """Release the transport."""
        await self._http.aclose()


def _entry_timestamp(entry: Any) -> datetime | None:
    """Extract a timezone-aware publication time from a feed entry.

    Feeds disagree about which field carries the date and in what format, so
    several are tried. An entry whose date cannot be resolved is dropped:
    defaulting to "now" would make stale items look like breaking news and
    corrupt every time-window correlation downstream.
    """
    for attribute in ("published", "updated", "created"):
        raw = getattr(entry, attribute, None)
        if not raw:
            continue
        try:
            parsed = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _entry_content(entry: Any) -> str | None:
    """Return an entry's full body when the feed provides one."""
    content = getattr(entry, "content", None)
    if content and isinstance(content, list) and content:
        value = content[0].get("value") if isinstance(content[0], dict) else None
        if value:
            return str(value)
    return None
