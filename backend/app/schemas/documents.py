"""Pydantic models for the MongoDB document collections.

MongoDB does not enforce a schema, which is exactly why one is defined here.
The flexibility that makes it right for news -- a NewsAPI article, an RSS item
and an SEC filing share almost no fields -- becomes a liability the moment a
producer writes ``published`` where a consumer reads ``published_at``. These
models are the contract every writer validates against.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.enums import DataSource, Sentiment


class MongoModel(BaseModel):
    """Base for documents, mapping MongoDB's ``_id`` to a plain ``id`` field."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    id: str | None = Field(default=None, alias="_id")

    def to_document(self) -> dict[str, Any]:
        """Serialise for insertion, omitting an unset ``_id``.

        Letting MongoDB generate the ``ObjectId`` avoids the client and server
        disagreeing about identity on a retried write.
        """
        document = self.model_dump(by_alias=True, exclude_none=False)
        if document.get("_id") is None:
            document.pop("_id", None)
        return document


class SentimentScore(BaseModel):
    """FinBERT's verdict on one piece of text."""

    model_config = ConfigDict(frozen=True)

    label: Sentiment
    confidence: Annotated[float, Field(ge=0, le=1)]
    #: Signed score in [-1, 1]: negative is bearish, positive bullish. Kept
    #: alongside the label so articles can be ranked, not just filtered.
    polarity: Annotated[float, Field(ge=-1, le=1)]
    model_name: str = "ProsusAI/finbert"


class NewsArticle(MongoModel):
    """A news item from any source, normalised to one shape."""

    #: SHA-256 of the canonical URL. The deduplication key: the same story
    #: arrives from an RSS feed and from NewsAPI minutes apart, and a unique
    #: index on this field makes the second write a no-op rather than a
    #: duplicate the correlation engine would double-count as evidence.
    url_hash: str
    url: str
    title: str
    summary: str | None = None
    content: str | None = None

    source: DataSource
    source_name: str | None = None
    author: str | None = None
    published_at: datetime
    ingested_at: datetime

    #: Ticker symbols the article mentions, resolved during ingestion. Indexed,
    #: so "news for MU this week" never scans the collection.
    tickers: list[str] = Field(default_factory=list)
    company_slugs: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    #: Ecosystem segments the article touches, e.g. ``["hbm", "foundry"]``.
    tags: list[str] = Field(default_factory=list)

    sentiment: SentimentScore | None = None
    language: str = "en"

    @field_validator("tickers", "company_slugs")
    @classmethod
    def _normalise_symbols(cls, value: list[str]) -> list[str]:
        """Upper-case symbols and drop duplicates, preserving order."""
        seen: dict[str, None] = {}
        for item in value:
            seen.setdefault(item.strip().upper(), None)
        return list(seen)

    @staticmethod
    def hash_url(url: str) -> str:
        """Return the deduplication key for a URL.

        The query string is preserved: some publishers use it for the article id
        itself, so stripping it would collapse distinct stories into one.
        """
        return hashlib.sha256(url.strip().lower().encode()).hexdigest()


class RagChunk(MongoModel):
    """One embedded passage, the unit Atlas Vector Search retrieves.

    Chunks are stored in their own collection rather than as an array on the
    parent document: Atlas indexes one vector per document, and keeping the
    source text next to its embedding means retrieval returns everything the
    prompt builder needs in a single query.
    """

    #: Identifier of the parent document in its own collection.
    source_id: str
    source_collection: Literal[
        "news_articles", "company_reports", "earnings_call_transcripts", "llm_summaries"
    ]
    source_url: str | None = None
    title: str | None = None

    text: str
    #: Position within the parent, so adjacent context can be re-assembled.
    chunk_index: int = 0
    token_count: int | None = None

    embedding: list[float] = Field(repr=False)
    embedding_model: str
    embedded_at: datetime

    #: Filter fields duplicated from the parent. Atlas applies filters *inside*
    #: the vector search stage, so denormalising them here is what allows
    #: "semantically similar AND about MU AND from last week" to stay one query.
    tickers: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    published_at: datetime | None = None


class LlmSummary(MongoModel):
    """A generated summary or analysis, with the provenance to reproduce it."""

    scope: Literal["daily", "weekly", "company", "sector", "anomaly"]
    subject: str = Field(description="Ticker, sector, or anomaly id the summary covers.")
    period_start: datetime
    period_end: datetime

    headline: str
    body: str
    key_points: list[str] = Field(default_factory=list)

    #: Documents the model was given. Every claim must be traceable to one of
    #: these -- an answer without sources is a bug, not a stylistic choice.
    source_document_ids: list[str] = Field(default_factory=list)

    model_name: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    generated_at: datetime


class ChatMessage(MongoModel):
    """One turn in a conversation with the assistant."""

    conversation_id: str
    user_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: datetime

    #: Populated on assistant turns: what was retrieved, and how confident the
    #: answer is. Stored so a past answer can be audited against its evidence.
    retrieved_document_ids: list[str] = Field(default_factory=list)
    confidence: Annotated[float, Field(ge=0, le=1)] | None = None
    model_name: str | None = None
    latency_ms: float | None = None
