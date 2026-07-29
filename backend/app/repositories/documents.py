"""MongoDB-backed repositories.

The repository boundary applies to the document store too: services ask for
"articles mentioning MU last week" and never see a Motor cursor, an aggregation
pipeline or a BSON filter. Without this, Mongo query syntax would leak into
services and the ingestion pipeline would be untestable without a live server.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pymongo import ASCENDING, DESCENDING, UpdateOne
from pymongo.errors import BulkWriteError

from app.core.logging import get_logger
from app.db.mongo import Collection, MongoDatabase, MongoDocument
from app.schemas.documents import NewsArticle, SentimentScore

logger = get_logger(__name__)

#: MongoDB's duplicate-key error code. Expected under concurrent ingestion
#: when two workers race on the same story, so it is not treated as a failure.
DUPLICATE_KEY_ERROR = 11000


class NewsRepository:
    """Reads and idempotent writes over the news collection."""

    def __init__(self, mongo: MongoDatabase) -> None:
        """Bind the repository to the document store."""
        self._mongo = mongo
        self._collection = mongo.collection(Collection.NEWS_ARTICLES)

    async def bulk_upsert(self, articles: Sequence[NewsArticle]) -> tuple[int, int]:
        """Insert articles, skipping ones already stored.

        Args:
            articles: Normalised, tagged articles.

        Returns:
            ``(inserted, matched)`` -- newly stored, and already present.

        Notes:
            Keyed on ``url_hash`` with ``$setOnInsert``, so re-ingesting a story
            that arrived earlier from another feed is a no-op rather than an
            overwrite. That matters because enrichment happens *after* storage:
            an article that already carries a sentiment score must not be reset
            to an unscored copy just because an RSS feed replayed it.

            ``ordered=False`` lets the remaining operations proceed when one
            fails -- a single malformed document must not discard the batch.
        """
        if not articles:
            return (0, 0)

        operations = [
            UpdateOne(
                {"url_hash": article.url_hash},
                {"$setOnInsert": article.to_document()},
                upsert=True,
            )
            for article in articles
        ]

        try:
            result = await self._collection.bulk_write(operations, ordered=False)
        except BulkWriteError as exc:
            # Duplicate-key errors are expected under concurrent ingestion: two
            # workers can race on the same story. They are not failures.
            written = exc.details.get("nUpserted", 0)
            errors = [
                error
                for error in exc.details.get("writeErrors", [])
                if error.get("code") != DUPLICATE_KEY_ERROR
            ]
            if errors:
                logger.warning("news_bulk_write_errors", count=len(errors), sample=errors[:3])
            return (written, len(articles) - written)

        inserted = len(result.upserted_ids or {})
        logger.info("news_upserted", inserted=inserted, submitted=len(articles))
        return (inserted, len(articles) - inserted)

    async def exists(self, url_hash: str) -> bool:
        """Return whether an article with this deduplication key is stored."""
        found = await self._collection.find_one({"url_hash": url_hash}, {"_id": 1})
        return found is not None

    async def list_recent(
        self,
        *,
        since: datetime,
        until: datetime | None = None,
        tickers: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        limit: int = 50,
    ) -> list[NewsArticle]:
        """Return articles in a window, newest first.

        Args:
            since: Inclusive lower bound on publication time.
            until: Inclusive upper bound; open-ended when omitted.
            tickers: Restrict to articles mentioning any of these symbols.
            tags: Restrict to articles carrying any of these segments.
            limit: Maximum articles returned.

        Notes:
            The filter order matches the compound indexes
            ``(tickers, published_at)`` and ``(tags, published_at)``: an equality
            on the array field followed by a range on time, which is the shape
            those indexes serve without a sort stage.
        """
        query: dict[str, Any] = {"published_at": {"$gte": since}}
        if until is not None:
            query["published_at"]["$lte"] = until
        if tickers:
            query["tickers"] = {"$in": [symbol.upper() for symbol in tickers]}
        if tags:
            query["tags"] = {"$in": list(tags)}

        cursor = self._collection.find(query).sort("published_at", DESCENDING).limit(limit)
        return [_to_article(document) async for document in cursor]

    async def list_near(
        self,
        *,
        tickers: Sequence[str],
        window_start: datetime,
        window_end: datetime,
        limit: int = 25,
    ) -> list[NewsArticle]:
        """Return articles about these tickers within a time window.

        The correlation engine's primary read: given an anomaly on a session,
        find the news that could explain it.
        """
        return await self.list_recent(
            since=window_start, until=window_end, tickers=tickers, limit=limit
        )

    async def search_text(self, term: str, *, limit: int = 25) -> list[NewsArticle]:
        """Full-text search over title, summary and body.

        The keyword half of hybrid search. Results are ranked by MongoDB's text
        score, which the vector side will later be fused with.
        """
        cursor = (
            self._collection.find({"$text": {"$search": term}}, {"score": {"$meta": "textScore"}})
            .sort([("score", {"$meta": "textScore"})])
            .limit(limit)
        )
        return [_to_article(document) async for document in cursor]

    async def list_unscored(self, *, since: datetime, limit: int = 200) -> list[NewsArticle]:
        """Return recent articles that carry no sentiment yet.

        The absence of the field *is* the work queue: no cursor, no checkpoint
        table, and a run that dies halfway simply leaves less work for the next
        one. Oldest first, so a backlog drains in publication order rather than
        starving the articles that have waited longest.
        """
        cursor = (
            self._collection.find({"published_at": {"$gte": since}, "sentiment": None})
            .sort("published_at", ASCENDING)
            .limit(limit)
        )
        return [_to_article(document) async for document in cursor]

    async def set_sentiments(self, updates: Sequence[tuple[str, SentimentScore]]) -> int:
        """Attach sentiment scores to stored articles.

        Args:
            updates: ``(url_hash, score)`` pairs.

        Returns:
            The number of articles modified.

        Notes:
            Keyed on ``url_hash`` -- the deduplication key -- rather than
            ``_id``, so a score can be written without the caller having to
            carry MongoDB's identifier around.
        """
        if not updates:
            return 0

        operations = [
            UpdateOne(
                {"url_hash": url_hash},
                {"$set": {"sentiment": score.model_dump()}},
            )
            for url_hash, score in updates
        ]
        result = await self._collection.bulk_write(operations, ordered=False)
        logger.info("sentiment_stored", modified=result.modified_count)
        return int(result.modified_count)

    async def count_since(self, since: datetime) -> int:
        """Return how many articles were published at or after ``since``."""
        return await self._collection.count_documents({"published_at": {"$gte": since}})


def _to_article(document: MongoDocument) -> NewsArticle:
    """Convert a stored document into a validated model.

    ``_id`` is stringified because ``ObjectId`` is not JSON-serialisable and
    every consumer -- the API, the anomaly's ``related_document_ids`` -- needs
    a plain string.
    """
    document = dict(document)
    document["_id"] = str(document["_id"])
    return NewsArticle.model_validate(document)
