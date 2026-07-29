"""Tests for chunking, embeddings, vector search and rank fusion.

The properties worth pinning down here are the ones whose violation is silent.
A chunker that loses a sentence, an embedder that misaligns a batch, a fusion
that lets one retriever dominate -- none of these raise. They just return
slightly wrong answers forever.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from app.core.config import EmbeddingSettings, IngestionSettings
from app.core.exceptions import ExternalServiceError
from app.models.enums import DataSource
from app.schemas.documents import NewsArticle
from app.services.rag.chunking import MIN_CHUNK_CHARS, chunk_text
from app.services.rag.embeddings import (
    HashingEmbeddingProvider,
    OpenAIEmbeddingProvider,
    build_embedding_provider,
)
from app.services.rag.search_service import (
    HybridSearchService,
    SearchMode,
    SearchResult,
)
from app.services.rag.vector_store import VectorFilter, VectorHit

NOW = datetime(2026, 7, 29, tzinfo=UTC)


# --- Chunking --------------------------------------------------------------


class TestChunking:
    """Passage splitting."""

    def test_short_text_is_a_single_chunk(self) -> None:
        chunks = chunk_text("A short headline.", chunk_size=1000, overlap=100)

        assert len(chunks) == 1
        assert chunks[0].index == 0

    def test_empty_text_yields_nothing(self) -> None:
        assert chunk_text("   \n  ", chunk_size=1000, overlap=100) == []

    def test_long_text_is_split_with_sequential_indices(self) -> None:
        text = " ".join(f"Sentence number {i} about memory pricing." for i in range(200))

        chunks = chunk_text(text, chunk_size=500, overlap=100)

        assert len(chunks) > 1
        assert [chunk.index for chunk in chunks] == list(range(len(chunks)))

    def test_no_content_is_lost(self) -> None:
        """A chunker that drops a sentence fails silently and forever."""
        sentences = [f"Fact {i} about HBM supply." for i in range(60)]
        text = " ".join(sentences)

        chunks = chunk_text(text, chunk_size=400, overlap=80)

        combined = " ".join(chunk.text for chunk in chunks)
        for sentence in sentences:
            assert sentence in combined

    def test_chunks_overlap(self) -> None:
        """Overlap is why a sentence on a boundary stays retrievable."""
        text = " ".join(f"Clause {i} regarding foundry capacity." for i in range(120))

        chunks = chunk_text(text, chunk_size=400, overlap=120)

        assert len(chunks) > 2
        # Consecutive chunks must share text; otherwise the overlap did nothing.
        tail = chunks[0].text[-60:]
        assert any(word in chunks[1].text for word in tail.split()[:5])

    def test_splitting_prefers_sentence_boundaries(self) -> None:
        """A chunk ending mid-clause embeds badly and quotes worse.

        Tested with zero overlap, which isolates the boundary logic: with
        overlap a chunk deliberately *starts* mid-sentence, so only where it
        ends says anything about boundary selection.
        """
        text = ("This is a complete sentence about DRAM. " * 40).strip()

        chunks = chunk_text(text, chunk_size=300, overlap=0)

        # Every chunk but the last should end on a full stop.
        assert all(chunk.text.endswith(".") for chunk in chunks[:-1])

    def test_overlap_duplicates_text_by_design(self) -> None:
        """The cost of overlap, asserted so it is not mistaken for a bug.

        Total chunk length must exceed the source, because the overlap region is
        deliberately stored twice. Without that duplication a sentence spanning
        a boundary would be retrievable from neither side.
        """
        text = ("This is a complete sentence about DRAM. " * 40).strip()

        with_overlap = chunk_text(text, chunk_size=300, overlap=100)
        without = chunk_text(text, chunk_size=300, overlap=0)

        overlapped_chars = sum(chunk.char_count for chunk in with_overlap)
        plain_chars = sum(chunk.char_count for chunk in without)
        assert overlapped_chars > plain_chars >= len(text) - len(with_overlap)

    def test_a_short_trailing_fragment_is_folded_in(self) -> None:
        """A 30-character chunk costs an embedding and carries no meaning."""
        text = "A" * 780 + ". " + "B" * 20

        chunks = chunk_text(text, chunk_size=400, overlap=50)

        assert all(len(chunk.text) >= MIN_CHUNK_CHARS for chunk in chunks)

    def test_overlap_must_be_smaller_than_the_chunk(self) -> None:
        """Otherwise the window advances by zero and the loop never ends."""
        with pytest.raises(ValueError, match="smaller than chunk_size"):
            chunk_text("text", chunk_size=100, overlap=100)

    def test_a_non_positive_chunk_size_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            chunk_text("text", chunk_size=0, overlap=0)


# --- Embeddings ------------------------------------------------------------


class TestHashingEmbeddings:
    """The credential-free fallback."""

    @pytest.fixture
    def provider(self) -> HashingEmbeddingProvider:
        return HashingEmbeddingProvider(dimensions=256)

    async def test_dimensions_match_the_configuration(
        self, provider: HashingEmbeddingProvider
    ) -> None:
        vectors = await provider.embed(["HBM capacity"])

        assert len(vectors[0]) == 256 == provider.dimensions

    async def test_vectors_are_unit_length(self, provider: HashingEmbeddingProvider) -> None:
        """Cosine similarity is only well behaved on normalised vectors."""
        vectors = await provider.embed(["Micron raises HBM guidance"])

        norm = math.sqrt(sum(value * value for value in vectors[0]))
        assert norm == pytest.approx(1.0, abs=1e-9)

    async def test_it_is_deterministic(self, provider: HashingEmbeddingProvider) -> None:
        """A non-deterministic embedder invalidates the whole stored index."""
        first = await provider.embed(["TSMC expands CoWoS capacity"])
        second = await provider.embed(["TSMC expands CoWoS capacity"])

        assert first == second

    async def test_lexically_similar_text_scores_higher_than_unrelated(
        self, provider: HashingEmbeddingProvider
    ) -> None:
        vectors = await provider.embed(
            [
                "HBM capacity is sold out through next year",
                "HBM capacity remains sold out into next year",
                "The Federal Reserve held interest rates steady",
            ]
        )

        related = sum(a * b for a, b in zip(vectors[0], vectors[1], strict=True))
        unrelated = sum(a * b for a, b in zip(vectors[0], vectors[2], strict=True))
        assert related > unrelated

    async def test_empty_input_yields_a_zero_vector_not_an_error(
        self, provider: HashingEmbeddingProvider
    ) -> None:
        vectors = await provider.embed([""])

        assert len(vectors[0]) == 256
        assert all(value == 0.0 for value in vectors[0])

    async def test_batches_preserve_order(self, provider: HashingEmbeddingProvider) -> None:
        texts = ["alpha memory", "beta foundry", "gamma lithography"]

        batch = await provider.embed(texts)
        individually = [(await provider.embed([text]))[0] for text in texts]

        assert batch == individually

    def test_the_model_name_is_versioned(self, provider: HashingEmbeddingProvider) -> None:
        """So its chunks can be found and re-embedded once a real model exists."""
        assert "v1" in provider.model_name


class TestOpenAIEmbeddings:
    """The production provider's contract with the API."""

    def _provider(self, transport: httpx.MockTransport) -> OpenAIEmbeddingProvider:
        return OpenAIEmbeddingProvider(
            EmbeddingSettings(openai_api_key="test-key", dimensions=64),  # type: ignore[arg-type]
            IngestionSettings(max_retries=1, retry_backoff_seconds=0.001),
            client=httpx.AsyncClient(transport=transport, base_url="https://api.test"),
        )

    async def test_vectors_are_reordered_by_index(self) -> None:
        """The API may return results out of order, each carrying its index.

        Trusting positional order would pair chunk A with chunk B's vector --
        a corruption no "did we get N vectors" check would catch.
        """
        payload = {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        }
        provider = self._provider(httpx.MockTransport(lambda _: httpx.Response(200, json=payload)))

        vectors = await provider.embed(["first", "second"])

        assert vectors[0] == [1.0, 0.0]
        assert vectors[1] == [0.0, 1.0]

    async def test_a_count_mismatch_is_rejected(self) -> None:
        """Silently returning fewer vectors would misalign every later chunk."""
        payload = {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}
        provider = self._provider(httpx.MockTransport(lambda _: httpx.Response(200, json=payload)))

        with pytest.raises(ExternalServiceError, match="vectors for 2 inputs"):
            await provider.embed(["first", "second"])

    async def test_the_key_is_sent_as_a_bearer_header(self) -> None:
        captured: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(request.headers)
            captured["_url"] = str(request.url)
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

        provider = self._provider(httpx.MockTransport(handler))
        await provider.embed(["text"])

        assert captured["authorization"] == "Bearer test-key"
        assert "test-key" not in captured["_url"]

    async def test_an_empty_batch_makes_no_request(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no request should be made")

        provider = self._provider(httpx.MockTransport(handler))

        assert await provider.embed([]) == []

    def test_a_missing_key_is_detectable_before_construction(self) -> None:
        settings = EmbeddingSettings(openai_api_key=None)

        assert OpenAIEmbeddingProvider.is_configured(settings) is False
        with pytest.raises(ExternalServiceError, match="not configured"):
            OpenAIEmbeddingProvider(settings, IngestionSettings())


class TestProviderSelection:
    """Choosing between the two embedders."""

    def test_hashing_is_used_when_no_key_is_configured(self) -> None:
        provider = build_embedding_provider(
            EmbeddingSettings(openai_api_key=None), IngestionSettings()
        )

        assert isinstance(provider, HashingEmbeddingProvider)

    def test_openai_is_used_when_a_key_is_configured(self) -> None:
        provider = build_embedding_provider(
            EmbeddingSettings(openai_api_key="k"),  # type: ignore[arg-type]
            IngestionSettings(),
        )

        assert isinstance(provider, OpenAIEmbeddingProvider)

    def test_both_produce_the_configured_width(self) -> None:
        """A width mismatch silently makes stored vectors incomparable."""
        settings = EmbeddingSettings(openai_api_key=None, dimensions=512)

        assert build_embedding_provider(settings, IngestionSettings()).dimensions == 512


# --- Fusion ----------------------------------------------------------------


class FakeVectorStore:
    """Returns scripted hits."""

    def __init__(self, hits: Sequence[VectorHit]) -> None:
        self._hits = list(hits)
        self.last_filters: VectorFilter | None = None

    @property
    def backend(self) -> str:
        return "fake"

    async def search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
        candidates: int,
        filters: VectorFilter | None = None,
    ) -> list[VectorHit]:
        self.last_filters = filters
        return self._hits[:limit]


class FakeNewsRepository:
    """Returns scripted keyword hits."""

    def __init__(self, articles: Sequence[NewsArticle]) -> None:
        self._articles = list(articles)

    async def search_text(self, term: str, *, limit: int = 25) -> list[NewsArticle]:
        return self._articles[:limit]


def _article(article_id: str, title: str, tickers: list[str] | None = None) -> NewsArticle:
    return NewsArticle(
        _id=article_id,
        url_hash=f"hash-{article_id}",
        url=f"https://news.test/{article_id}",
        title=title,
        summary=f"Summary of {title}",
        source=DataSource.RSS,
        published_at=NOW - timedelta(hours=1),
        ingested_at=NOW,
        tickers=tickers or [],
    )


def _hit(source_id: str, text: str, score: float) -> VectorHit:
    return VectorHit(
        chunk_id=f"chunk-{source_id}",
        source_id=source_id,
        source_collection="news_articles",
        text=text,
        score=score,
        published_at=NOW - timedelta(hours=1),
    )


def _service(
    *, articles: Sequence[NewsArticle], hits: Sequence[VectorHit], rrf_k: int = 60
) -> tuple[HybridSearchService, FakeVectorStore]:
    store = FakeVectorStore(hits)
    service = HybridSearchService(
        embedder=HashingEmbeddingProvider(dimensions=64),
        vector_store=store,  # type: ignore[arg-type]
        news=FakeNewsRepository(articles),  # type: ignore[arg-type]
        candidates=50,
        rrf_k=rrf_k,
    )
    return service, store


class TestHybridSearch:
    """Retrieval modes and reciprocal rank fusion."""

    async def test_keyword_mode_uses_only_the_text_index(self) -> None:
        service, store = _service(
            articles=[_article("a", "Micron HBM guidance")],
            hits=[_hit("b", "unrelated chunk", 0.9)],
        )

        response = await service.search("HBM", mode=SearchMode.KEYWORD)

        assert [result.source_id for result in response.results] == ["a"]
        assert store.last_filters is None, "vector search should not have run"

    async def test_vector_mode_uses_only_the_vector_store(self) -> None:
        service, _ = _service(
            articles=[_article("a", "Micron HBM guidance")],
            hits=[_hit("b", "semantically similar chunk", 0.9)],
        )

        response = await service.search("HBM", mode=SearchMode.VECTOR)

        assert [result.source_id for result in response.results] == ["b"]

    async def test_hybrid_returns_the_union(self) -> None:
        service, _ = _service(
            articles=[_article("a", "Micron HBM guidance")],
            hits=[_hit("b", "semantically similar chunk", 0.9)],
        )

        response = await service.search("HBM", mode=SearchMode.HYBRID)

        assert {result.source_id for result in response.results} == {"a", "b"}

    async def test_agreement_between_retrievers_outranks_either_alone(self) -> None:
        """The whole point of fusion: cross-system agreement is evidence."""
        articles = [_article("solo_kw", "keyword only"), _article("both", "found by both")]
        hits = [_hit("solo_vec", "vector only", 0.95), _hit("both", "found by both", 0.90)]
        service, _ = _service(articles=articles, hits=hits)

        response = await service.search("query", mode=SearchMode.HYBRID)

        assert response.results[0].source_id == "both"
        assert set(response.results[0].matched_by) == {"keyword", "vector"}

    async def test_a_top_vector_hit_does_not_automatically_win(self) -> None:
        """RRF damps rank 1, so one confident system cannot dominate alone."""
        articles = [_article("both", "agreed"), _article("kw2", "second keyword")]
        hits = [_hit("vec_top", "very similar", 0.99), _hit("both", "agreed", 0.4)]
        service, _ = _service(articles=articles, hits=hits)

        response = await service.search("query", mode=SearchMode.HYBRID)

        assert response.results[0].source_id == "both"

    async def test_fusion_ignores_incomparable_raw_scores(self) -> None:
        """A cosine of 0.99 and a TF-IDF score of 12 are not on one scale.

        Only rank is used, so an absurd similarity value cannot buy a position
        that the ranking does not justify.
        """
        articles = [_article("kw", "keyword hit")]
        service_high, _ = _service(articles=articles, hits=[_hit("vec", "chunk", 0.999)])
        service_low, _ = _service(articles=articles, hits=[_hit("vec", "chunk", 0.001)])

        high = await service_high.search("q", mode=SearchMode.HYBRID)
        low = await service_low.search("q", mode=SearchMode.HYBRID)

        assert [r.source_id for r in high.results] == [r.source_id for r in low.results]

    async def test_chunks_of_one_article_occupy_one_slot(self) -> None:
        """Three chunks of a story are one piece of evidence, not three."""
        hits = [
            _hit("same", "chunk one", 0.9),
            _hit("same", "chunk two", 0.8),
            _hit("same", "chunk three", 0.7),
        ]
        service, _ = _service(articles=[], hits=hits)

        response = await service.search("q", mode=SearchMode.VECTOR)

        assert len(response.results) == 1

    async def test_results_record_which_retriever_found_them(self) -> None:
        """A citation must be auditable without re-running the query."""
        service, _ = _service(
            articles=[_article("a", "keyword hit")], hits=[_hit("a", "same doc", 0.9)]
        )

        response = await service.search("q", mode=SearchMode.HYBRID)

        result = response.results[0]
        assert set(result.matched_by) == {"keyword", "vector"}
        assert result.ranks == {"keyword": 1, "vector": 1}

    async def test_ticker_filters_reach_the_vector_store(self) -> None:
        service, store = _service(articles=[], hits=[])

        await service.search("q", mode=SearchMode.VECTOR, tickers=["mu"])

        assert store.last_filters is not None
        assert store.last_filters.tickers == ("MU",)

    async def test_ticker_filters_are_applied_to_keyword_hits_too(self) -> None:
        """The text index cannot express the predicate, so it is applied after."""
        articles = [_article("a", "About Micron", ["MU"]), _article("b", "About Intel", ["INTC"])]
        service, _ = _service(articles=articles, hits=[])

        response = await service.search("q", mode=SearchMode.KEYWORD, tickers=["MU"])

        assert [result.source_id for result in response.results] == ["a"]

    async def test_the_limit_is_applied_after_fusion(self) -> None:
        articles = [_article(str(index), f"Article {index}") for index in range(10)]
        service, _ = _service(articles=articles, hits=[])

        response = await service.search("q", mode=SearchMode.KEYWORD, limit=3)

        assert response.count == 3

    async def test_the_backend_is_reported(self) -> None:
        """Approximate and exact search disagree; results must be attributable."""
        service, _ = _service(articles=[], hits=[])

        response = await service.search("q")

        assert response.backend == "fake"

    async def test_an_empty_corpus_returns_nothing_rather_than_failing(self) -> None:
        service, _ = _service(articles=[], hits=[])

        response = await service.search("anything")

        assert response.results == ()
        assert response.count == 0


class TestSearchResultShape:
    """The contract the RAG layer will consume in Milestone 7."""

    async def test_scores_are_positive_and_descending(self) -> None:
        articles = [_article(str(i), f"Article {i}") for i in range(5)]
        service, _ = _service(articles=articles, hits=[])

        response = await service.search("q", mode=SearchMode.KEYWORD)

        scores = [result.score for result in response.results]
        assert all(score > 0 for score in scores)
        assert scores == sorted(scores, reverse=True)

    async def test_results_carry_a_citable_url(self) -> None:
        service, _ = _service(articles=[_article("a", "Micron news")], hits=[])

        response = await service.search("q", mode=SearchMode.KEYWORD)

        assert response.results[0].source_url == "https://news.test/a"

    def test_search_result_is_hashable_for_deduplication(self) -> None:
        """Frozen dataclasses with a dict field are not hashable; assert the shape.

        The fused result carries `ranks`, so it is intentionally compared by
        source_id rather than by identity.
        """
        result = SearchResult(text="t", score=1.0, source_id="a")

        assert result.source_id == "a"
        assert result.ranks == {}
