"""Hybrid search: keyword and vector retrieval fused by reciprocal rank.

The two halves fail in opposite directions, which is the entire argument for
running both. Keyword search cannot match "memory prices firmed" to "DRAM ASPs
rose" -- no shared terms. Vector search will happily return something
semantically adjacent when the user typed an exact ticker or a product name it
has never seen, because approximate similarity has no notion of "must contain".

**Reciprocal Rank Fusion** combines them, rather than normalising and adding the
scores. That choice is deliberate and load-bearing:

- MongoDB's text score is an unbounded TF-IDF-ish number; cosine similarity is
  bounded in [-1, 1]. They are not on comparable scales, and no fixed weighting
  makes them so across different queries.
- Min-max normalising per query makes the top result's score depend on how bad
  the *worst* returned result happened to be, which is not a property of
  relevance.
- RRF uses only *rank*, which both systems produce meaningfully. A document
  ranked 1st by keyword and 40th by vector still scores well; one ranked 40th by
  both does not.

The constant k=60 comes from the original TREC work. It damps the influence of
the very top ranks enough that one confident-but-wrong system cannot dominate
the fused ordering.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from app.core.logging import get_logger
from app.repositories.documents import NewsRepository
from app.services.rag.embeddings import EmbeddingProvider
from app.services.rag.vector_store import VectorFilter, VectorHit, VectorStore

logger = get_logger(__name__)


class SearchMode(StrEnum):
    """Which retrieval strategies to run."""

    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class SearchResult:
    """One fused result, with the evidence for why it ranked where it did."""

    text: str
    score: float
    source_id: str
    source_url: str | None = None
    title: str | None = None
    tickers: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    published_at: datetime | None = None
    #: Which retrievers surfaced this result, and at what rank. Kept because a
    #: RAG answer must be auditable: "why was this cited?" has to be answerable
    #: after the fact, not re-derived by re-running the query.
    matched_by: tuple[str, ...] = ()
    ranks: dict[str, int] = field(default_factory=dict)
    #: Raw cosine similarity from the vector store, preserved through fusion.
    #: RRF replaces `score` with a rank-derived value, and rank is relative:
    #: the top result among entirely irrelevant documents still ranks first and
    #: still scores 1/(k+1). Only an absolute similarity can answer "did we
    #: actually find anything?", which is what refusal depends on.
    vector_similarity: float | None = None


@dataclass(frozen=True)
class SearchResponse:
    """A complete search, including how it was served."""

    query: str
    mode: SearchMode
    backend: str
    results: tuple[SearchResult, ...]

    @property
    def count(self) -> int:
        """Number of results returned."""
        return len(self.results)


class HybridSearchService:
    """Runs keyword and vector retrieval and fuses the rankings."""

    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        news: NewsRepository,
        candidates: int = 100,
        rrf_k: int = 60,
    ) -> None:
        """Wire the service to its collaborators.

        Args:
            embedder: Produces the query vector.
            vector_store: Semantic half of retrieval.
            news: Keyword half, via the collection's text index.
            candidates: Depth each retriever is asked for before fusion. Deeper
                than the final result count, because fusion needs candidates to
                fuse -- truncating each list to k first would discard exactly the
                cross-system agreement RRF exists to reward.
            rrf_k: Fusion damping constant.
        """
        self._embedder = embedder
        self._vector_store = vector_store
        self._news = news
        self._candidates = candidates
        self._rrf_k = rrf_k

    async def search(
        self,
        query: str,
        *,
        mode: SearchMode = SearchMode.HYBRID,
        limit: int = 10,
        tickers: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        since: datetime | None = None,
    ) -> SearchResponse:
        """Retrieve passages matching ``query``.

        Args:
            query: Natural-language or keyword query.
            mode: Which retrievers to run.
            limit: Results returned after fusion.
            tickers: Restrict to passages about these symbols.
            tags: Restrict to these ecosystem segments.
            since: Restrict to passages published on or after this time.

        Returns:
            Fused results, best first.
        """
        filters = VectorFilter(
            tickers=tuple(symbol.upper() for symbol in tickers or ()),
            tags=tuple(tags or ()),
            since=since,
        )

        keyword_hits: list[SearchResult] = []
        vector_hits: list[SearchResult] = []

        if mode in {SearchMode.KEYWORD, SearchMode.HYBRID}:
            keyword_hits = await self._keyword_search(query, filters)
        if mode in {SearchMode.VECTOR, SearchMode.HYBRID}:
            vector_hits = await self._vector_search(query, filters)

        rankings: dict[str, list[SearchResult]] = {}
        if mode in {SearchMode.KEYWORD, SearchMode.HYBRID}:
            rankings["keyword"] = keyword_hits
        if mode in {SearchMode.VECTOR, SearchMode.HYBRID}:
            rankings["vector"] = vector_hits
        fused = self._fuse(rankings)

        logger.info(
            "search_complete",
            mode=mode.value,
            keyword_hits=len(keyword_hits),
            vector_hits=len(vector_hits),
            returned=min(limit, len(fused)),
        )
        return SearchResponse(
            query=query,
            mode=mode,
            backend=self._vector_store.backend,
            results=tuple(fused[:limit]),
        )

    async def _keyword_search(self, query: str, filters: VectorFilter) -> list[SearchResult]:
        """Run the text-index half against whole articles."""
        articles = await self._news.search_text(query, limit=self._candidates)

        results: list[SearchResult] = []
        for article in articles:
            if not _passes(filters, tickers=article.tickers, published_at=article.published_at):
                continue
            results.append(
                SearchResult(
                    text=article.summary or article.title,
                    score=0.0,  # replaced by the fused score
                    source_id=str(article.id),
                    source_url=article.url,
                    title=article.title,
                    tickers=tuple(article.tickers),
                    tags=tuple(article.tags),
                    published_at=article.published_at,
                )
            )
        return results

    async def _vector_search(self, query: str, filters: VectorFilter) -> list[SearchResult]:
        """Run the semantic half against embedded chunks."""
        vectors = await self._embedder.embed([query])
        if not vectors:
            return []

        hits = await self._vector_store.search(
            vectors[0],
            limit=self._candidates,
            candidates=self._candidates * 2,
            filters=filters,
        )
        return [_from_vector_hit(hit) for hit in hits]

    def _fuse(self, rankings: dict[str, list[SearchResult]]) -> list[SearchResult]:
        """Combine ranked lists by reciprocal rank fusion.

        Each list contributes ``1 / (k + rank)`` to every document it returned.
        Documents both retrievers found accumulate from both, which is precisely
        the cross-system agreement worth rewarding.

        Deduplication is by ``source_id``, not chunk: three chunks of the same
        article are one piece of evidence, and letting them occupy three result
        slots would crowd out genuinely distinct sources.
        """
        fused: dict[str, SearchResult] = {}
        scores: dict[str, float] = {}
        seen: dict[str, set[str]] = {}

        for retriever, results in rankings.items():
            for rank, result in enumerate(results, start=1):
                key = result.source_id

                # Only a retriever's *best* rank for a document counts. Several
                # chunks of one article arrive as several hits; scoring each
                # would let a long article outrank a short one purely by being
                # chunked more finely, which is a property of the text length,
                # not of its relevance.
                already = seen.setdefault(key, set())
                if retriever in already:
                    continue
                already.add(retriever)

                scores[key] = scores.get(key, 0.0) + 1.0 / (self._rrf_k + rank)

                existing = fused.get(key)
                if existing is None:
                    fused[key] = SearchResult(
                        text=result.text,
                        score=0.0,
                        source_id=result.source_id,
                        source_url=result.source_url,
                        title=result.title,
                        tickers=result.tickers,
                        tags=result.tags,
                        published_at=result.published_at,
                        matched_by=(retriever,),
                        ranks={retriever: rank},
                        vector_similarity=result.vector_similarity,
                    )
                else:
                    fused[key] = SearchResult(
                        text=existing.text or result.text,
                        score=0.0,
                        source_id=existing.source_id,
                        source_url=existing.source_url or result.source_url,
                        title=existing.title or result.title,
                        tickers=existing.tickers or result.tickers,
                        tags=existing.tags or result.tags,
                        published_at=existing.published_at or result.published_at,
                        matched_by=(*existing.matched_by, retriever),
                        ranks={**existing.ranks, retriever: rank},
                        vector_similarity=(
                            existing.vector_similarity
                            if existing.vector_similarity is not None
                            else result.vector_similarity
                        ),
                    )

        ordered = sorted(fused.values(), key=lambda item: scores[item.source_id], reverse=True)
        return [
            SearchResult(
                text=item.text,
                score=round(scores[item.source_id], 6),
                source_id=item.source_id,
                source_url=item.source_url,
                title=item.title,
                tickers=item.tickers,
                tags=item.tags,
                published_at=item.published_at,
                matched_by=item.matched_by,
                ranks=item.ranks,
                vector_similarity=item.vector_similarity,
            )
            for item in ordered
        ]


def _from_vector_hit(hit: VectorHit) -> SearchResult:
    """Map a vector hit onto the common result shape."""
    return SearchResult(
        text=hit.text,
        score=hit.score,
        vector_similarity=hit.score,
        source_id=hit.source_id,
        source_url=hit.source_url,
        title=hit.title,
        tickers=hit.tickers,
        tags=hit.tags,
        published_at=hit.published_at,
    )


def _passes(
    filters: VectorFilter, *, tickers: Sequence[str], published_at: datetime | None
) -> bool:
    """Apply the vector filters to a keyword hit.

    The text index cannot express these predicates, so they are applied after
    retrieval on this half only. That asymmetry is acceptable because the
    keyword stage is bounded by ``candidates``; the vector stage, which could
    scan an entire corpus, filters during retrieval instead.
    """
    if filters.tickers and not set(filters.tickers) & {t.upper() for t in tickers}:
        return False
    return not (
        filters.since is not None and (published_at is None or published_at < filters.since)
    )
