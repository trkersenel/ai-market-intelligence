"""Integration tests for the MongoDB repositories.

These exist because of a bug that unit tests could not have caught. The endpoint
tests replace ``NewsRepository`` with a fake, so nothing exercised the real
``$meta`` text-score projection -- which attaches a ``score`` field that the
strict document model rejects. Keyword search was broken from the moment it
shipped, and the suite was green throughout.

The lesson generalises: a fake can only be as correct as the author's mental
model of the real thing. Anything the database itself decides -- projections,
index behaviour, type coercion -- has to be verified against the database.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ServerSelectionTimeoutError

from app.core.config import get_settings
from app.db.mongo import Collection, MongoDatabase
from app.db.mongo_indexes import create_indexes
from app.models.enums import DataSource, Sentiment
from app.repositories.documents import NewsRepository, RagChunkRepository
from app.schemas.documents import NewsArticle, RagChunk, SentimentScore

pytestmark = pytest.mark.integration

NOW = datetime(2026, 7, 29, tzinfo=UTC)

#: A dedicated database, so a careless run can never touch development data.
TEST_DATABASE = "market_intel_integration_test"


@pytest.fixture(scope="session")
async def mongo() -> AsyncIterator[MongoDatabase]:
    """A live MongoDB bound to a throwaway database."""
    settings = get_settings().mongo.model_copy(update={"database": TEST_DATABASE})
    database = MongoDatabase(settings)

    try:
        await database.ping()
    except (ServerSelectionTimeoutError, OSError) as exc:  # pragma: no cover
        pytest.skip(f"MongoDB is not reachable: {exc}")

    await create_indexes(database)
    yield database

    client: AsyncIOMotorClient = database.client
    await client.drop_database(TEST_DATABASE)
    await database.close()


@pytest.fixture(autouse=True)
async def _clean(mongo: MongoDatabase) -> AsyncIterator[None]:
    """Empty the collections between tests."""
    yield
    for collection in (Collection.NEWS_ARTICLES, Collection.RAG_DOCUMENTS):
        await mongo.collection(collection).delete_many({})


def _article(title: str, *, summary: str = "", tickers: list[str] | None = None) -> NewsArticle:
    return NewsArticle(
        url_hash=NewsArticle.hash_url(f"https://news.test/{abs(hash(title))}"),
        url=f"https://news.test/{abs(hash(title))}",
        title=title,
        summary=summary or f"Summary of {title}",
        source=DataSource.RSS,
        published_at=NOW - timedelta(hours=2),
        ingested_at=NOW - timedelta(hours=1),
        tickers=tickers or [],
    )


class TestNewsRepositoryAgainstMongo:
    """Behaviour that only the real server determines."""

    async def test_text_search_returns_validated_articles(self, mongo: MongoDatabase) -> None:
        """The regression test for the projection bug.

        ``search_text`` must project a ``$meta`` score in order to sort by
        relevance. That field is not part of the document, and the model forbids
        extras -- so without stripping it, every text search raises.
        """
        repository = NewsRepository(mongo)
        await repository.bulk_upsert(
            [
                _article("Micron raises HBM guidance on memory demand"),
                _article("Federal Reserve holds rates steady"),
            ]
        )

        results = await repository.search_text("HBM memory")

        assert results
        assert any("HBM" in article.title for article in results)

    async def test_text_search_ranks_by_relevance(self, mongo: MongoDatabase) -> None:
        repository = NewsRepository(mongo)
        await repository.bulk_upsert(
            [
                _article("A passing mention of HBM", summary="Mostly about something else"),
                _article("HBM HBM HBM capacity", summary="HBM supply and HBM pricing"),
            ]
        )

        results = await repository.search_text("HBM")

        assert results[0].title.startswith("HBM HBM HBM")

    async def test_upsert_is_idempotent(self, mongo: MongoDatabase) -> None:
        repository = NewsRepository(mongo)
        article = _article("Micron beats estimates")

        first, _ = await repository.bulk_upsert([article])
        second, duplicates = await repository.bulk_upsert([article])

        assert first == 1
        assert second == 0
        assert duplicates == 1

    async def test_an_existing_sentiment_survives_re_ingestion(self, mongo: MongoDatabase) -> None:
        """``$setOnInsert`` exists for this: enrichment happens after storage."""
        repository = NewsRepository(mongo)
        article = _article("Micron beats estimates")
        await repository.bulk_upsert([article])
        await repository.set_sentiments(
            [
                (
                    article.url_hash,
                    SentimentScore(label=Sentiment.BULLISH, confidence=0.9, polarity=0.8),
                )
            ]
        )

        await repository.bulk_upsert([article])

        stored = await repository.list_recent(since=NOW - timedelta(days=1))
        assert stored[0].sentiment is not None
        assert stored[0].sentiment.label is Sentiment.BULLISH

    async def test_unscored_returns_only_articles_without_sentiment(
        self, mongo: MongoDatabase
    ) -> None:
        repository = NewsRepository(mongo)
        scored = _article("Already scored")
        unscored = _article("Not yet scored")
        await repository.bulk_upsert([scored, unscored])
        await repository.set_sentiments(
            [
                (
                    scored.url_hash,
                    SentimentScore(label=Sentiment.NEUTRAL, confidence=0.5, polarity=0.0),
                )
            ]
        )

        pending = await repository.list_unscored(since=NOW - timedelta(days=1))

        assert [article.title for article in pending] == ["Not yet scored"]

    async def test_ticker_filter_uses_the_array_index(self, mongo: MongoDatabase) -> None:
        repository = NewsRepository(mongo)
        await repository.bulk_upsert(
            [
                _article("About Micron", tickers=["MU"]),
                _article("About Intel", tickers=["INTC"]),
            ]
        )

        results = await repository.list_recent(since=NOW - timedelta(days=1), tickers=["MU"])

        assert [article.title for article in results] == ["About Micron"]


class TestRagChunkRepositoryAgainstMongo:
    """Chunk storage and its re-embedding semantics."""

    def _chunk(self, source_id: str, index: int, model: str = "test-model") -> RagChunk:
        return RagChunk(
            source_id=source_id,
            source_collection="news_articles",
            text=f"Chunk {index} of {source_id}",
            chunk_index=index,
            embedding=[0.1] * 8,
            embedding_model=model,
            embedded_at=NOW,
        )

    async def test_chunks_are_stored_and_counted(self, mongo: MongoDatabase) -> None:
        repository = RagChunkRepository(mongo)

        written = await repository.replace_for_source(
            "doc1", [self._chunk("doc1", 0), self._chunk("doc1", 1)]
        )

        assert written == 2
        assert await repository.count() == 2

    async def test_re_chunking_into_fewer_pieces_leaves_no_orphans(
        self, mongo: MongoDatabase
    ) -> None:
        """The reason for delete-then-insert rather than upsert-by-index.

        A shortened article or a larger chunk size produces fewer pieces. An
        upsert keyed on ``(source_id, chunk_index)`` would leave the surplus
        chunks behind, still answering queries with text the document no longer
        contains.
        """
        repository = RagChunkRepository(mongo)
        await repository.replace_for_source(
            "doc1", [self._chunk("doc1", index) for index in range(5)]
        )

        await repository.replace_for_source("doc1", [self._chunk("doc1", 0)])

        assert await repository.count() == 1

    async def test_indexed_sources_are_scoped_to_the_model(self, mongo: MongoDatabase) -> None:
        """Switching embedders must make the whole corpus pending again."""
        repository = RagChunkRepository(mongo)
        await repository.replace_for_source("doc1", [self._chunk("doc1", 0, model="old")])

        assert await repository.source_ids_for_model("old") == {"doc1"}
        assert await repository.source_ids_for_model("new") == set()

    async def test_retiring_a_model_removes_its_vectors(self, mongo: MongoDatabase) -> None:
        """Retiring a model must remove its vectors.

        Vectors from different models are not comparable, so leaving them
        behind poisons retrieval with results scored in a different space.
        """
        repository = RagChunkRepository(mongo)
        await repository.replace_for_source("doc1", [self._chunk("doc1", 0, model="old")])
        await repository.replace_for_source("doc2", [self._chunk("doc2", 0, model="new")])

        removed = await repository.delete_for_model("old")

        assert removed == 1
        assert await repository.count() == 1
