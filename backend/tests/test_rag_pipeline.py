"""Tests for answer generation, correlation and chat.

The properties under test are the platform's promises, not its plumbing: that it
never answers without evidence, that confidence tracks the evidence rather than
the prose, that citations resolve to real sources, and that correlation is never
presented as causation.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta

import httpx
import pytest

from app.core.config import EmbeddingSettings, IngestionSettings, LlmSettings
from app.core.exceptions import ExternalServiceError
from app.models.enums import (
    AnomalyType,
    DataSource,
    DetectionMethod,
    Direction,
    Sentiment,
    Severity,
)
from app.schemas.documents import ChatMessage, NewsArticle, SentimentScore
from app.services.rag.chat_service import ChatService
from app.services.rag.correlation import CorrelationEngine
from app.services.rag.embeddings import HashingEmbeddingProvider, OpenAIEmbeddingProvider
from app.services.rag.llm import (
    SYSTEM_PROMPT,
    ExtractiveAnswerer,
    LlmResponse,
    OpenAIChatClient,
    build_llm_client,
)
from app.services.rag.rag_service import MIN_EVIDENCE_SCORE, RagService
from app.services.rag.search_service import SearchMode, SearchResponse, SearchResult

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


# --- LLM clients -----------------------------------------------------------


class TestExtractiveAnswerer:
    """The fallback that cannot fabricate."""

    @pytest.fixture
    def answerer(self) -> ExtractiveAnswerer:
        return ExtractiveAnswerer()

    async def test_it_quotes_sentences_from_the_context(self, answerer: ExtractiveAnswerer) -> None:
        context = (
            "[1] Micron Q3\nMicron raised its HBM revenue outlook for the year. "
            "The company said capacity is sold out through next year.\n\n"
            "[2] Fed\nThe Federal Reserve held interest rates steady this month."
        )

        response = await answerer.complete(
            question="What did Micron say about HBM?", context=context
        )

        assert "HBM revenue outlook" in response.text
        assert response.extractive is True

    async def test_every_selected_sentence_exists_in_the_context(
        self, answerer: ExtractiveAnswerer
    ) -> None:
        """The structural guarantee: there is no generation step to invent in."""
        context = "[1] Title\nMicron raised its HBM revenue outlook for the year."

        response = await answerer.complete(question="HBM outlook?", context=context)

        quoted = [
            line.lstrip("- ").rsplit(" [", 1)[0]
            for line in response.text.splitlines()
            if line.startswith("- ")
        ]
        for sentence in quoted:
            assert sentence in context

    async def test_selected_sentences_carry_their_citation(
        self, answerer: ExtractiveAnswerer
    ) -> None:
        context = "[1] Title\nMicron raised its HBM revenue outlook substantially this year."

        response = await answerer.complete(question="HBM outlook?", context=context)

        assert "[1]" in response.text

    async def test_it_says_so_when_the_context_does_not_answer(
        self, answerer: ExtractiveAnswerer
    ) -> None:
        context = "[1] Weather\nIt rained in Boise on Tuesday and again on Wednesday."

        response = await answerer.complete(
            question="What did Micron guide for HBM revenue?", context=context
        )

        assert "do not contain" in response.text

    async def test_it_is_not_generative(self, answerer: ExtractiveAnswerer) -> None:
        assert answerer.is_generative is False
        assert answerer.model_name == "extractive-v1"


class TestOpenAIChatClient:
    """The generative path's contract with the API."""

    def _client(self, transport: httpx.MockTransport) -> OpenAIChatClient:
        return OpenAIChatClient(
            LlmSettings(openai_api_key="test-key"),  # type: ignore[arg-type]
            IngestionSettings(max_retries=1, retry_backoff_seconds=0.001),
            client=httpx.AsyncClient(transport=transport, base_url="https://api.test"),
        )

    async def test_the_grounding_prompt_is_sent(self) -> None:
        """The instruction is what makes the answer groundable; assert it ships."""
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json  # noqa: PLC0415

            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Answer [1]."}}], "usage": {}},
            )

        client = self._client(httpx.MockTransport(handler))
        await client.complete(question="Why did MU move?", context="[1] passage")

        messages = captured["messages"]
        assert isinstance(messages, list)
        assert messages[0]["content"] == SYSTEM_PROMPT
        assert "ONLY from the numbered context" in messages[0]["content"]

    async def test_token_usage_is_captured(self) -> None:
        payload = {
            "choices": [{"message": {"content": "Micron raised guidance [1]."}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30},
        }
        client = self._client(httpx.MockTransport(lambda _: httpx.Response(200, json=payload)))

        response = await client.complete(question="q", context="[1] c")

        assert response.prompt_tokens == 120
        assert response.completion_tokens == 30
        assert response.extractive is False

    async def test_an_empty_completion_is_rejected(self) -> None:
        """Returning "" as an answer would look like a confident non-answer."""
        payload = {"choices": [{"message": {"content": "   "}}]}
        client = self._client(httpx.MockTransport(lambda _: httpx.Response(200, json=payload)))

        with pytest.raises(ExternalServiceError, match="empty completion"):
            await client.complete(question="q", context="[1] c")

    def test_a_missing_key_is_detectable_before_construction(self) -> None:
        settings = LlmSettings(openai_api_key=None)

        assert OpenAIChatClient.is_configured(settings) is False
        with pytest.raises(ExternalServiceError, match="not configured"):
            OpenAIChatClient(settings, IngestionSettings())

    async def test_a_missing_key_degrades_to_extraction(self) -> None:
        client = await build_llm_client(LlmSettings(openai_api_key=None), IngestionSettings())

        assert isinstance(client, ExtractiveAnswerer)


# --- RAG service -----------------------------------------------------------


class FakeSearchService:
    """Returns scripted search responses."""

    def __init__(self, results: Sequence[SearchResult]) -> None:
        self._results = list(results)
        self.last_mode: SearchMode | None = None

    async def search(self, query: str, **kwargs: object) -> SearchResponse:
        self.last_mode = kwargs.get("mode")  # type: ignore[assignment]
        return SearchResponse(
            query=query,
            mode=SearchMode.HYBRID,
            backend="fake",
            results=tuple(self._results),
        )


class RecordingLlm:
    """Captures the context it was given."""

    def __init__(self) -> None:
        self.context: str | None = None

    @property
    def model_name(self) -> str:
        return "recording"

    @property
    def is_generative(self) -> bool:
        return True

    async def complete(self, *, question: str, context: str) -> LlmResponse:
        self.context = context
        return LlmResponse(text=f"Answer to {question} [1]", model_name=self.model_name)


def _result(
    source_id: str,
    score: float,
    *,
    matched_by: tuple[str, ...] = ("keyword",),
    title: str = "A headline",
    similarity: float | None = 0.4,
) -> SearchResult:
    return SearchResult(
        text=f"Passage body for {source_id}. It discusses memory pricing at length.",
        score=score,
        source_id=source_id,
        source_url=f"https://news.test/{source_id}",
        title=title,
        published_at=NOW - timedelta(hours=3),
        matched_by=matched_by,
        vector_similarity=similarity,
    )


def _rag(results: Sequence[SearchResult]) -> tuple[RagService, RecordingLlm]:
    llm = RecordingLlm()
    service = RagService(
        search=FakeSearchService(results),  # type: ignore[arg-type]
        llm=llm,
        relevance_floor=0.15,
        context_passages=8,
    )
    return service, llm


class TestRagService:
    """Grounding, refusal and confidence."""

    async def test_an_answer_carries_numbered_citations(self) -> None:
        service, _ = _rag([_result("a", 0.03), _result("b", 0.02)])

        answer = await service.answer("Why did memory move?")

        assert [c.number for c in answer.citations] == [1, 2]
        assert answer.citations[0].url == "https://news.test/a"

    async def test_the_context_is_numbered_to_match_the_citations(self) -> None:
        """Numbering is what makes a cited claim traceable to a URL."""
        service, llm = _rag([_result("a", 0.03), _result("b", 0.02)])

        await service.answer("q")

        assert llm.context is not None
        assert llm.context.startswith("[1]")
        assert "[2]" in llm.context

    async def test_it_refuses_when_nothing_is_retrieved(self) -> None:
        """The central promise: no evidence means no answer, not a guess."""
        service, llm = _rag([])

        answer = await service.answer("Why did an untracked company move?")

        assert answer.refused is True
        assert answer.citations == ()
        assert answer.confidence == 0.0
        assert llm.context is None, "the model must not be called without evidence"

    async def test_it_refuses_when_every_result_is_below_the_evidence_floor(self) -> None:
        service, llm = _rag([_result("a", MIN_EVIDENCE_SCORE / 2)])

        answer = await service.answer("q")

        assert answer.refused is True
        assert llm.context is None

    async def test_confidence_rises_with_the_amount_of_evidence(self) -> None:
        thin, _ = _rag([_result("a", 0.02)])
        thick, _ = _rag([_result(str(i), 0.02) for i in range(8)])

        assert (await thick.answer("q")).confidence > (await thin.answer("q")).confidence

    async def test_confidence_rises_when_both_retrievers_agree(self) -> None:
        """Cross-retriever agreement is the strongest signal short of a human."""
        one_sided, _ = _rag([_result("a", 0.03, matched_by=("keyword",))])
        agreed, _ = _rag([_result("a", 0.03, matched_by=("keyword", "vector"))])

        assert (await agreed.answer("q")).confidence > (await one_sided.answer("q")).confidence

    async def test_confidence_never_reaches_certainty(self) -> None:
        """A confidence of 1.0 would invite a user to stop checking."""
        service, _ = _rag(
            [_result(str(i), 0.9, matched_by=("keyword", "vector")) for i in range(20)]
        )

        assert (await service.answer("q")).confidence <= 0.95

    async def test_the_model_is_never_asked_for_its_own_confidence(self) -> None:
        """Confidence is computed pre-generation, from observable retrieval."""
        service, llm = _rag([_result("a", 0.03)])

        answer = await service.answer("q")

        assert llm.context is not None
        assert "confidence" not in llm.context.lower()
        assert answer.confidence > 0

    async def test_retrieval_always_runs_in_hybrid_mode(self) -> None:
        results = [_result("a", 0.03)]
        llm = RecordingLlm()
        search = FakeSearchService(results)
        service = RagService(search=search, llm=llm, relevance_floor=0.15)  # type: ignore[arg-type]

        await service.answer("q")

        assert search.last_mode is SearchMode.HYBRID


# --- Correlation -----------------------------------------------------------


class FakeAnomaly:
    """Stands in for the Anomaly ORM model."""

    def __init__(
        self,
        anomaly_id: int = 1,
        *,
        direction: Direction = Direction.DOWN,
        anomaly_type: AnomalyType = AnomalyType.RETURN,
        observed: float = -0.08,
    ) -> None:
        self.id = anomaly_id
        self.ticker_id = 1
        self.trade_date = date(2026, 7, 28)
        self.direction = direction
        self.anomaly_type = anomaly_type
        self.method = DetectionMethod.Z_SCORE
        self.severity = Severity.HIGH
        self.observed_value = observed
        self.explanation: str | None = None


class FakeTickerRepo:
    """Resolves an anomaly's ticker id to a symbol."""

    class _Ticker:
        id = 1
        symbol = "MU"
        display_name = "Micron Technology, Inc."

    async def get(self, ticker_id: int) -> _Ticker | None:
        return self._Ticker() if ticker_id == 1 else None


class FakeAnomalyRepo:
    """Tracks which anomalies were explained and with what evidence."""

    def __init__(self, anomalies: Sequence[FakeAnomaly]) -> None:
        self.anomalies = list(anomalies)
        self.explained: dict[int, tuple[str, list[str]]] = {}

    async def list_unexplained(self, *, limit: int = 50) -> list[FakeAnomaly]:
        return [a for a in self.anomalies if a.explanation is None][:limit]

    async def attach_explanation(
        self, anomaly_id: int, *, explanation: str, document_ids: Sequence[str]
    ) -> None:
        self.explained[anomaly_id] = (explanation, list(document_ids))


class FakeNewsRepo:
    """Serves scripted articles and records the window it was asked for."""

    def __init__(self, articles: Sequence[NewsArticle]) -> None:
        self._articles = list(articles)
        self.window: tuple[datetime, datetime] | None = None

    async def list_recent(self, **kwargs: object) -> list[NewsArticle]:
        self.window = (kwargs["since"], kwargs["until"])  # type: ignore[assignment]
        since, until = self.window
        return [a for a in self._articles if since <= a.published_at <= until]


def _news(
    article_id: str,
    title: str,
    *,
    hours_before_close: float,
    tickers: list[str] | None = None,
    sentiment: Sentiment | None = None,
    tags: list[str] | None = None,
) -> NewsArticle:
    close = datetime(2026, 7, 28, 21, 0, tzinfo=UTC)
    return NewsArticle(
        _id=article_id,
        url_hash=f"hash-{article_id}",
        url=f"https://news.test/{article_id}",
        title=title,
        source=DataSource.RSS,
        published_at=close - timedelta(hours=hours_before_close),
        ingested_at=close,
        tickers=tickers or [],
        tags=["dram"] if tags is None else tags,
        sentiment=(
            None
            if sentiment is None
            else SentimentScore(label=sentiment, confidence=0.8, polarity=-0.5)
        ),
    )


def _engine(articles: Sequence[NewsArticle], anomalies: Sequence[FakeAnomaly]):  # noqa: ANN202
    news = FakeNewsRepo(articles)
    anomaly_repo = FakeAnomalyRepo(anomalies)
    engine = CorrelationEngine(
        news=news,  # type: ignore[arg-type]
        anomalies=anomaly_repo,  # type: ignore[arg-type]
        tickers=FakeTickerRepo(),  # type: ignore[arg-type]
    )
    return engine, news, anomaly_repo


class TestCorrelationEngine:
    """Ranking, hedging and the asymmetric window."""

    async def test_the_window_is_asymmetric(self) -> None:
        """News after the close mostly reports the move rather than causing it."""
        engine, news, _ = _engine([], [FakeAnomaly()])

        await engine.explain(FakeAnomaly())  # type: ignore[arg-type]

        assert news.window is not None
        start, end = news.window
        close = datetime(2026, 7, 28, 21, 0, tzinfo=UTC)
        assert (close - start) > (end - close)

    async def test_an_article_naming_the_company_outranks_one_that_does_not(self) -> None:
        articles = [
            _news("generic", "Memory sector under pressure", hours_before_close=2),
            _news("named", "MU cuts guidance", hours_before_close=2, tickers=["MU"]),
        ]
        engine, _, _ = _engine(articles, [])

        result = await engine.explain(FakeAnomaly())  # type: ignore[arg-type]

        assert result is not None
        assert result.candidates[0].article.id == "named"
        assert result.candidates[0].named_in_text is True

    async def test_news_published_after_the_close_ranks_below_news_before(self) -> None:
        """The reaction article must not be cited as the cause."""
        articles = [
            _news("reaction", "Micron falls 8%", hours_before_close=-3, tickers=["MU"]),
            _news("cause", "Micron cuts guidance", hours_before_close=4, tickers=["MU"]),
        ]
        engine, _, _ = _engine(articles, [])

        result = await engine.explain(FakeAnomaly())  # type: ignore[arg-type]

        assert result is not None
        assert result.candidates[0].article.id == "cause"

    async def test_agreeing_sentiment_is_corroborating(self) -> None:
        articles = [
            _news("neutral", "Micron update", hours_before_close=2, tickers=["MU"]),
            _news(
                "bearish",
                "Micron warns on demand",
                hours_before_close=2,
                tickers=["MU"],
                sentiment=Sentiment.BEARISH,
            ),
        ]
        engine, _, _ = _engine(articles, [])

        result = await engine.explain(FakeAnomaly(direction=Direction.DOWN))  # type: ignore[arg-type]

        assert result is not None
        assert result.candidates[0].article.id == "bearish"
        assert result.candidates[0].sentiment_agrees is True

    async def test_disagreeing_sentiment_is_not_disqualifying(self) -> None:
        """A stock can fall on good news; excluding those loses the best stories."""
        articles = [
            _news(
                "bullish",
                "Micron beats estimates",
                hours_before_close=2,
                tickers=["MU"],
                sentiment=Sentiment.BULLISH,
            )
        ]
        engine, _, _ = _engine(articles, [])

        result = await engine.explain(FakeAnomaly(direction=Direction.DOWN))  # type: ignore[arg-type]

        assert result is not None
        assert len(result.candidates) == 1

    async def test_irrelevant_articles_are_excluded_entirely(self) -> None:
        """Not about this company and not about the sector is not evidence."""
        articles = [_news("noise", "Local weather report", hours_before_close=2, tags=[])]
        engine, _, _ = _engine(articles, [])

        assert await engine.explain(FakeAnomaly()) is None  # type: ignore[arg-type]

    async def test_the_explanation_never_claims_causation(self) -> None:
        """The most damaging kind of wrong is confident, plausible and unfounded."""
        articles = [_news("a", "Micron cuts guidance", hours_before_close=2, tickers=["MU"])]
        engine, _, _ = _engine(articles, [])

        result = await engine.explain(FakeAnomaly())  # type: ignore[arg-type]

        assert result is not None
        assert "not established causes" in result.explanation
        assert "Possible contributing factors" in result.explanation
        assert "caused by" not in result.explanation.lower()

    async def test_the_explanation_is_stored_with_its_sources(self) -> None:
        articles = [_news("a", "Micron cuts guidance", hours_before_close=2, tickers=["MU"])]
        engine, _, anomaly_repo = _engine(articles, [])

        await engine.explain(FakeAnomaly(anomaly_id=7))  # type: ignore[arg-type]

        explanation, document_ids = anomaly_repo.explained[7]
        assert "MU" in explanation
        assert document_ids == ["a"]

    async def test_no_explanation_is_written_when_nothing_is_plausible(self) -> None:
        """A placeholder would keep the anomaly out of the queue forever."""
        engine, _, anomaly_repo = _engine([], [])

        result = await engine.explain(FakeAnomaly(anomaly_id=9))  # type: ignore[arg-type]

        assert result is None
        assert 9 not in anomaly_repo.explained

    async def test_pending_anomalies_are_processed(self) -> None:
        articles = [_news("a", "Micron cuts guidance", hours_before_close=2, tickers=["MU"])]
        engine, _, _ = _engine(articles, [FakeAnomaly(1), FakeAnomaly(2)])

        results = await engine.explain_pending()

        assert len(results) == 2


# --- Chat ------------------------------------------------------------------


class FakeChatRepo:
    """In-memory conversation store."""

    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    async def append(self, message: ChatMessage) -> str:
        self.messages.append(message)
        return str(len(self.messages))

    async def list_conversation(
        self, conversation_id: str, *, limit: int = 20
    ) -> list[ChatMessage]:
        return [m for m in self.messages if m.conversation_id == conversation_id][:limit]

    async def last_user_message(self, conversation_id: str) -> ChatMessage | None:
        turns = [
            m for m in self.messages if m.conversation_id == conversation_id and m.role == "user"
        ]
        return turns[-1] if turns else None


class TestChatService:
    """Conversation persistence and follow-up handling."""

    def _service(self) -> tuple[ChatService, FakeChatRepo]:
        rag, _ = _rag([_result("a", 0.03)])
        chat = FakeChatRepo()
        return ChatService(rag=rag, chat=chat), chat

    async def test_both_sides_of_the_exchange_are_stored(self) -> None:
        """An audit trail: what was said, and on what basis, must be recoverable."""
        service, chat = self._service()

        await service.ask("Why did Micron fall?", user_id="u1")

        assert [m.role for m in chat.messages] == ["user", "assistant"]

    async def test_the_assistant_turn_records_its_evidence(self) -> None:
        service, chat = self._service()

        await service.ask("Why did Micron fall?", user_id="u1")

        assistant = chat.messages[1]
        assert assistant.retrieved_document_ids == ["a"]
        assert assistant.confidence is not None
        assert assistant.latency_ms is not None

    async def test_a_new_conversation_gets_an_id(self) -> None:
        service, _ = self._service()

        turn = await service.ask("A question", user_id="u1")

        assert turn.conversation_id

    async def test_an_existing_conversation_is_continued(self) -> None:
        service, chat = self._service()
        first = await service.ask("First question", user_id="u1")

        await service.ask("Another question", user_id="u1", conversation_id=first.conversation_id)

        assert len({m.conversation_id for m in chat.messages}) == 1

    async def test_a_follow_up_is_resolved_against_the_previous_turn(self) -> None:
        service, _ = self._service()
        first = await service.ask("How did Micron perform?", user_id="u1")

        second = await service.ask(
            "What about its volume?", user_id="u1", conversation_id=first.conversation_id
        )

        assert second.was_resolved
        assert "Micron" in second.resolved_question

    async def test_a_standalone_question_is_not_rewritten(self) -> None:
        """Blending two topics into one query answers neither well."""
        service, _ = self._service()
        first = await service.ask("How did Micron perform?", user_id="u1")

        second = await service.ask(
            "How did NVIDIA perform?", user_id="u1", conversation_id=first.conversation_id
        )

        assert not second.was_resolved
        assert "Micron" not in second.resolved_question

    async def test_a_follow_up_in_a_fresh_conversation_is_left_alone(self) -> None:
        service, _ = self._service()

        turn = await service.ask("What about it?", user_id="u1")

        assert not turn.was_resolved

    async def test_history_replays_in_order(self) -> None:
        service, _ = self._service()
        turn = await service.ask("A question", user_id="u1")

        history = await service.history(turn.conversation_id)

        assert [m.role for m in history] == ["user", "assistant"]


class TestAttributionStrength:
    """Source attribution and text naming are different strengths of evidence.

    A live run made the distinction necessary: Yahoo's per-ticker feed also
    carries general market news, so an article can arrive stamped "000660.KS"
    while being about US crypto regulation. Reporting that as "names this
    company" was simply false.
    """

    async def test_a_feed_stamp_alone_is_not_reported_as_naming_the_company(self) -> None:
        articles = [
            _news(
                "generic",
                "The CLARITY Act divides finance giants ahead of recess",
                hours_before_close=4,
                tickers=["MU"],
            )
        ]
        engine, _, _ = _engine(articles, [])

        result = await engine.explain(FakeAnomaly())  # type: ignore[arg-type]

        assert result is not None
        candidate = result.candidates[0]
        assert candidate.source_attributed is True
        assert candidate.named_in_text is False
        assert "filed under this ticker by the source" in result.explanation
        assert "names this company" not in result.explanation

    async def test_text_naming_outranks_a_bare_feed_stamp(self) -> None:
        articles = [
            _news("stamped", "Broad market wrap", hours_before_close=2, tickers=["MU"]),
            _news("named", "MU cuts guidance", hours_before_close=2),
        ]
        engine, _, _ = _engine(articles, [])

        result = await engine.explain(FakeAnomaly())  # type: ignore[arg-type]

        assert result is not None
        assert result.candidates[0].article.id == "named"

    async def test_both_signals_together_rank_highest(self) -> None:
        articles = [
            _news("stamped", "Broad market wrap", hours_before_close=2, tickers=["MU"]),
            _news("named", "MU cuts guidance", hours_before_close=2),
            _news("both", "MU warns on demand", hours_before_close=2, tickers=["MU"]),
        ]
        engine, _, _ = _engine(articles, [])

        result = await engine.explain(FakeAnomaly())  # type: ignore[arg-type]

        assert result is not None
        assert result.candidates[0].article.id == "both"


class TestReactionVersusCause:
    """Coverage published after the close cannot be offered as explanation."""

    async def test_a_corpus_with_only_later_coverage_says_so(self) -> None:
        """The specific failure the engine exists to avoid.

        With no pre-close news, ranking still produces candidates -- but calling
        them "contributing factors" would present an article reporting the move
        as a reason for it.
        """
        articles = [
            _news("after", "MU falls 8% on the day", hours_before_close=-19, tickers=["MU"])
        ]
        engine, _, _ = _engine(articles, [])

        result = await engine.explain(FakeAnomaly())  # type: ignore[arg-type]

        assert result is not None
        assert "No news published before the close" in result.explanation
        assert "cannot explain" in result.explanation
        assert "Possible contributing factors" not in result.explanation
        # The softer hedge is omitted here: "cannot explain" is the stronger
        # statement, and following it with "not established causes" would read
        # as a partial retraction.
        assert "not established causes" not in result.explanation

    async def test_pre_close_news_keeps_the_causal_framing(self) -> None:
        articles = [_news("before", "MU cuts guidance", hours_before_close=5, tickers=["MU"])]
        engine, _, _ = _engine(articles, [])

        result = await engine.explain(FakeAnomaly())  # type: ignore[arg-type]

        assert result is not None
        assert "Possible contributing factors" in result.explanation
        assert "not established causes" in result.explanation


class TestRelevanceGate:
    """Refusal must depend on absolute similarity, not on fused rank.

    Found by a live test: asked about the 1987 Brazilian coffee harvest, the
    platform returned seven semiconductor articles with confidence 0.79. Every
    one cleared the fused-score floor, because rank 1 of anything scores
    1/(k+1) whether or not it is remotely related.
    """

    async def test_it_refuses_when_nothing_is_semantically_close(self) -> None:
        service, llm = _rag([_result(str(i), 0.03, similarity=0.001) for i in range(7)])

        answer = await service.answer("What did the Brazilian coffee harvest yield in 1987?")

        assert answer.refused is True
        assert answer.confidence == 0.0
        assert answer.citations == ()
        assert llm.context is None, "the model must not be called without relevant evidence"

    async def test_keyword_only_hits_cannot_vouch_for_relevance(self) -> None:
        """A text index matches a shared word, which is how noise looks plausible."""
        service, _ = _rag([_result("a", 0.03, matched_by=("keyword",), similarity=None)])

        assert (await service.answer("q")).refused is True

    async def test_one_semantically_close_result_is_enough(self) -> None:
        service, _ = _rag(
            [
                _result("far", 0.03, similarity=0.09),
                _result("near", 0.02, similarity=0.4),
            ]
        )

        answer = await service.answer("q")

        assert answer.refused is False
        assert answer.retrieved == 2


class TestProviderRelevanceFloors:
    """The bar belongs to the embedder, because similarity scales differ."""

    def test_the_hashing_floor_separates_the_measured_corpus_scores(self) -> None:
        """Measured live: off-topic peaked at 0.091, on-topic at 0.247."""
        floor = HashingEmbeddingProvider(dimensions=64).relevance_floor

        assert 0.091 < floor < 0.247

    def test_the_openai_floor_is_higher(self) -> None:
        """A semantic model spreads related and unrelated text much further."""
        openai_floor = OpenAIEmbeddingProvider(
            EmbeddingSettings(openai_api_key="k"),  # type: ignore[arg-type]
            IngestionSettings(),
        ).relevance_floor

        assert openai_floor > HashingEmbeddingProvider().relevance_floor
