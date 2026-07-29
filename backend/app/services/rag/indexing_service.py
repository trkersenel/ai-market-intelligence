"""Chunk, embed and store news articles for retrieval.

Runs after sentiment scoring so an embedded chunk carries its article's
sentiment as filterable metadata. Ordering the pipeline this way costs nothing
and means "bearish news about Micron last week" is one query rather than a join.

Idempotent on ``(source_id, chunk_index)``: re-embedding a document replaces its
chunks rather than duplicating them, which is what makes a model change a
re-run rather than a migration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.core.logging import get_logger
from app.repositories.documents import NewsRepository, RagChunkRepository
from app.schemas.documents import NewsArticle, RagChunk
from app.services.rag.chunking import chunk_text
from app.services.rag.embeddings import EmbeddingProvider

logger = get_logger(__name__)

#: How far back an unembedded article is worth indexing.
DEFAULT_MAX_AGE_DAYS = 30


@dataclass(frozen=True)
class IndexingReport:
    """Outcome of one indexing pass."""

    started_at: datetime
    finished_at: datetime
    documents: int
    chunks: int
    model_name: str
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the pass completed without an error."""
        return self.error is None

    @property
    def duration_seconds(self) -> float:
        """Wall-clock duration of the pass."""
        return (self.finished_at - self.started_at).total_seconds()


class DocumentIndexingService:
    """Turns stored articles into embedded, retrievable chunks."""

    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        news: NewsRepository,
        chunks: RagChunkRepository,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        documents_per_run: int = 200,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    ) -> None:
        """Wire the service to its collaborators.

        Args:
            embedder: Anything satisfying :class:`EmbeddingProvider`.
            news: Source of the articles.
            chunks: Destination repository.
            chunk_size: Target characters per passage.
            chunk_overlap: Characters repeated between consecutive passages.
            documents_per_run: Articles indexed per pass.
            max_age_days: Oldest article worth indexing.
        """
        self._embedder = embedder
        self._news = news
        self._chunks = chunks
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._documents_per_run = documents_per_run
        self._max_age = timedelta(days=max_age_days)

    async def index_pending(self, *, limit: int | None = None) -> IndexingReport:
        """Embed every recent article that has no chunks for this model.

        Notes:
            The work queue is derived, not stored: an article with no chunks
            *carrying this model's name* is pending. That definition makes a
            model switch self-healing -- point the config at a new model and the
            same job re-embeds the corpus, with no migration and no flag.
        """
        started = datetime.now(UTC)
        since = started - self._max_age
        model = self._embedder.model_name

        try:
            candidates = await self._news.list_recent(
                since=since, limit=limit or self._documents_per_run
            )
            indexed = await self._chunks.source_ids_for_model(model)
            pending = [
                article
                for article in candidates
                if article.id is not None and article.id not in indexed
            ]

            if not pending:
                return IndexingReport(
                    started_at=started,
                    finished_at=datetime.now(UTC),
                    documents=0,
                    chunks=0,
                    model_name=model,
                )

            written = 0
            for article in pending:
                written += await self._index_one(article)

            report = IndexingReport(
                started_at=started,
                finished_at=datetime.now(UTC),
                documents=len(pending),
                chunks=written,
                model_name=model,
            )
            logger.info(
                "indexing_run_complete",
                documents=report.documents,
                chunks=report.chunks,
                model=model,
                duration_seconds=round(report.duration_seconds, 2),
            )
            return report

        except Exception as exc:  # reported in the run summary, never propagated
            logger.exception("indexing_run_failed")
            return IndexingReport(
                started_at=started,
                finished_at=datetime.now(UTC),
                documents=0,
                chunks=0,
                model_name=model,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _index_one(self, article: NewsArticle) -> int:
        """Chunk, embed and store one article. Returns chunks written."""
        body = _indexable_text(article)
        pieces = chunk_text(body, chunk_size=self._chunk_size, overlap=self._chunk_overlap)
        if not pieces:
            return 0

        vectors = await self._embedder.embed([piece.text for piece in pieces])
        embedded_at = datetime.now(UTC)

        chunks = [
            RagChunk(
                source_id=str(article.id),
                source_collection="news_articles",
                source_url=article.url,
                title=article.title,
                text=piece.text,
                chunk_index=piece.index,
                token_count=piece.char_count // 4,  # rough, for cost estimation only
                embedding=vector,
                embedding_model=self._embedder.model_name,
                embedded_at=embedded_at,
                # Denormalised from the parent so Atlas can filter *inside* the
                # vector scan. Joining afterwards would return the global top-k
                # and then discard non-matching hits, often leaving far fewer.
                tickers=list(article.tickers),
                tags=list(article.tags),
                published_at=article.published_at,
            )
            for piece, vector in zip(pieces, vectors, strict=True)
        ]
        return await self._chunks.replace_for_source(str(article.id), chunks)


def _indexable_text(article: NewsArticle) -> str:
    """Assemble the text to embed.

    The title is prepended to the body rather than embedded separately: it is
    the densest statement of what the article is about, and repeating it at the
    head of the document means the first chunk -- usually the most retrieved --
    carries it.
    """
    parts = [article.title]
    if article.summary:
        parts.append(article.summary)
    if article.content:
        parts.append(article.content)
    return "\n\n".join(part for part in parts if part)
