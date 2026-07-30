"""Declarative MongoDB index definitions and idempotent creation.

MongoDB has no migration tool, so indexes are declared here as data and applied
on startup. ``create_indexes`` is idempotent: creating an index that already
exists with the same definition is a no-op, which makes this safe to run on
every boot and on every replica.

The Atlas Vector Search index is deliberately *not* created here -- it is a
search index, managed by a different Atlas API that local MongoDB does not
implement. :func:`vector_index_definition` emits its JSON so it can be applied
via the Atlas UI, CLI or Terraform, and the platform degrades to keyword search
when it is absent rather than failing to start.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pymongo
from pymongo.errors import OperationFailure

from app.core.logging import get_logger
from app.db.mongo import Collection, MongoDatabase

logger = get_logger(__name__)


@dataclass(frozen=True)
class IndexSpec:
    """One index to guarantee on one collection."""

    collection: Collection
    keys: list[tuple[str, int | str]]
    name: str
    unique: bool = False
    #: Restricts the index to documents matching this filter, keeping it small
    #: when only a subset of documents carry the field.
    partial_filter: dict[str, Any] | None = None
    #: Seconds after which documents expire; ``None`` means never.
    expire_after_seconds: int | None = None
    reason: str = field(default="", compare=False)


#: Every index the platform depends on. Each entry records the query it serves,
#: so an unused index can be retired with confidence rather than kept "just in
#: case" -- every index is write amplification on the ingestion path.
INDEX_SPECS: tuple[IndexSpec, ...] = (
    IndexSpec(
        collection=Collection.NEWS_ARTICLES,
        keys=[("url_hash", pymongo.ASCENDING)],
        name="uq_news_url_hash",
        unique=True,
        reason="Deduplication: the same story arrives from several feeds.",
    ),
    IndexSpec(
        collection=Collection.NEWS_ARTICLES,
        keys=[("published_at", pymongo.DESCENDING)],
        name="ix_news_published_at",
        reason="Reverse-chronological news feed and window queries.",
    ),
    IndexSpec(
        collection=Collection.NEWS_ARTICLES,
        keys=[("tickers", pymongo.ASCENDING), ("published_at", pymongo.DESCENDING)],
        name="ix_news_ticker_published",
        reason="'News for MU this week' -- the correlation engine's hot path.",
    ),
    IndexSpec(
        collection=Collection.NEWS_ARTICLES,
        keys=[("tags", pymongo.ASCENDING), ("published_at", pymongo.DESCENDING)],
        name="ix_news_tags_published",
        reason="Segment feeds: the HBM tracker, AI-infrastructure page.",
    ),
    IndexSpec(
        collection=Collection.NEWS_ARTICLES,
        keys=[("sentiment.label", pymongo.ASCENDING), ("published_at", pymongo.DESCENDING)],
        name="ix_news_sentiment_published",
        partial_filter={"sentiment": {"$exists": True}},
        reason="Sentiment filtering; partial so unscored articles stay out.",
    ),
    IndexSpec(
        collection=Collection.NEWS_ARTICLES,
        keys=[("title", pymongo.TEXT), ("summary", pymongo.TEXT), ("content", pymongo.TEXT)],
        name="ix_news_fulltext",
        reason="Keyword half of hybrid search; vector search covers semantics.",
    ),
    IndexSpec(
        collection=Collection.RAG_DOCUMENTS,
        keys=[("source_id", pymongo.ASCENDING), ("chunk_index", pymongo.ASCENDING)],
        name="uq_rag_source_chunk",
        unique=True,
        reason="Re-embedding a document must replace its chunks, not duplicate them.",
    ),
    IndexSpec(
        collection=Collection.RAG_DOCUMENTS,
        keys=[("tickers", pymongo.ASCENDING), ("published_at", pymongo.DESCENDING)],
        name="ix_rag_ticker_published",
        reason="Pre-filtering retrieval to a company and a time window.",
    ),
    IndexSpec(
        collection=Collection.LLM_SUMMARIES,
        keys=[
            ("scope", pymongo.ASCENDING),
            ("subject", pymongo.ASCENDING),
            ("period_end", pymongo.DESCENDING),
        ],
        name="ix_summary_scope_subject_period",
        reason="Look up the latest summary for a scope before regenerating it.",
    ),
    IndexSpec(
        collection=Collection.CHAT_HISTORY,
        keys=[("conversation_id", pymongo.ASCENDING), ("created_at", pymongo.ASCENDING)],
        name="ix_chat_conversation_created",
        reason="Replaying a conversation in order.",
    ),
    IndexSpec(
        collection=Collection.CHAT_HISTORY,
        keys=[("user_id", pymongo.ASCENDING), ("created_at", pymongo.DESCENDING)],
        name="ix_chat_user_created",
        reason="A user's recent conversations.",
    ),
    IndexSpec(
        collection=Collection.USER_PREFERENCES,
        keys=[("user_id", pymongo.ASCENDING)],
        name="uq_preferences_user",
        unique=True,
        reason="One preferences document per account.",
    ),
    IndexSpec(
        collection=Collection.COMPANY_REPORTS,
        keys=[("company_slug", pymongo.ASCENDING), ("filed_at", pymongo.DESCENDING)],
        name="ix_report_company_filed",
        reason="A company's filings, newest first.",
    ),
    IndexSpec(
        collection=Collection.AI_REPORTS,
        keys=[("symbol", pymongo.ASCENDING), ("model", pymongo.ASCENDING)],
        name="uq_ai_report_symbol_model",
        unique=True,
        reason=(
            "One cached briefing per symbol per model. Unique so a page opened "
            "twice at once cannot store two, and so changing models does not "
            "serve the previous model's prose under the new one's name."
        ),
    ),
    IndexSpec(
        collection=Collection.AI_REPORTS,
        keys=[("generated_at", pymongo.ASCENDING)],
        name="ttl_ai_report_generated",
        # Seven days, not the twelve-hour freshness window. The service decides
        # what is *fresh*; this only stops abandoned symbols accumulating
        # forever. Expiring at the freshness boundary would delete rows the
        # service would otherwise overwrite in place, costing an insert.
        expire_after_seconds=7 * 24 * 3600,
        reason="Reclaim briefings for symbols nobody has opened in a week.",
    ),
    IndexSpec(
        collection=Collection.EARNINGS_CALL_TRANSCRIPTS,
        keys=[("ticker", pymongo.ASCENDING), ("call_date", pymongo.DESCENDING)],
        name="ix_transcript_ticker_date",
        reason="A company's earnings calls, newest first.",
    ),
)


async def create_indexes(mongo: MongoDatabase) -> list[str]:
    """Ensure every declared index exists.

    Args:
        mongo: The document-store adapter.

    Returns:
        Names of the indexes that were created or already present.

    Notes:
        A failure on one index is logged and skipped rather than aborting
        startup. The common cause is an index that exists with different
        options, which needs an operator decision -- not a crash loop that takes
        the API down with it.
    """
    created: list[str] = []
    for spec in INDEX_SPECS:
        collection = mongo.collection(spec.collection)
        options: dict[str, Any] = {"name": spec.name}
        if spec.unique:
            options["unique"] = True
        if spec.partial_filter is not None:
            options["partialFilterExpression"] = spec.partial_filter
        if spec.expire_after_seconds is not None:
            options["expireAfterSeconds"] = spec.expire_after_seconds

        try:
            await collection.create_index(spec.keys, **options)
        except OperationFailure as exc:
            logger.warning(
                "mongo_index_creation_failed",
                collection=spec.collection.value,
                index=spec.name,
                error=str(exc),
            )
            continue
        created.append(f"{spec.collection.value}.{spec.name}")

    logger.info("mongo_indexes_ensured", count=len(created))
    return created


def vector_index_definition(
    *, index_name: str, dimensions: int = 1536, path: str = "embedding"
) -> dict[str, Any]:
    """Return the Atlas Vector Search index definition for the RAG collection.

    Args:
        index_name: Name to register the search index under.
        dimensions: Embedding width. 1536 matches OpenAI
            ``text-embedding-3-small``; changing the model means rebuilding the
            index, since vectors of different widths are not comparable.
        path: Document field holding the embedding.

    Returns:
        The JSON definition, ready to submit to the Atlas Search API.

    Notes:
        ``cosine`` similarity is correct for OpenAI embeddings, which are
        L2-normalised -- cosine and dot product then rank identically, and
        cosine is robust if a future model is not normalised.

        The ``filter`` fields let Atlas apply metadata predicates during the
        vector scan. Filtering afterwards would return the top-k globally and
        *then* discard non-matching hits, often leaving far fewer than k.
    """
    return {
        "name": index_name,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "path": path,
                    "numDimensions": dimensions,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "tickers"},
                {"type": "filter", "path": "tags"},
                {"type": "filter", "path": "source_collection"},
                {"type": "filter", "path": "published_at"},
            ]
        },
    }
