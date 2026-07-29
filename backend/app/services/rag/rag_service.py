"""The question-answering pipeline.

Retrieve, assemble, generate, cite. The ordering is the product: the platform
answers *only* from what it retrieved, so an answer with no retrieved evidence is
a refusal rather than a guess.

Confidence is computed from the evidence, never asked of the model. A language
model's stated confidence is a property of its prose style, not of whether the
sources support the claim -- it will say "certainly" about a fabrication as
readily as about a fact. Here confidence rises with how much was retrieved, how
well it scored, and whether the retrievers agreed, all of which are observable
before generation happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from app.core.logging import get_logger
from app.services.rag.llm import LlmClient
from app.services.rag.search_service import HybridSearchService, SearchMode, SearchResult

logger = get_logger(__name__)

#: Below this fused score, a result did not rank anywhere useful.
#: RRF scores are small by construction: a single top-ranked hit contributes
#: 1/(60+1) = 0.0164, so this admits roughly that and better.
MIN_EVIDENCE_SCORE = 0.015

#: The absolute similarity bar comes from the embedding provider, because
#: similarity scales differ between embedders and a platform-wide constant
#: would be right for at most one of them.
#:
#: A bar is needed at all because the fused score cannot support refusal, as a
#: live test made concrete: asked about the 1987 Brazilian coffee harvest,
#: retrieval returned seven semiconductor articles, every one clearing the fused
#: floor because rank 1 of anything scores 1/(k+1). The platform reported
#: confidence 0.79 on a question it could not answer. Rank is relative; cosine
#: is not.

#: Retrieved passages beyond which the platform is confident it has enough.
#: Used only to scale confidence, not to truncate.
CONFIDENCE_SATURATION = 6


@dataclass(frozen=True)
class Citation:
    """One source backing an answer."""

    number: int
    title: str | None
    url: str | None
    source_id: str
    published_at: datetime | None = None
    matched_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class RagAnswer:
    """A grounded answer with everything needed to audit it."""

    question: str
    answer: str
    citations: tuple[Citation, ...]
    confidence: float
    model_name: str
    extractive: bool
    retrieved: int
    #: True when retrieval found nothing usable and the platform declined to
    #: answer. Distinguished from a low-confidence answer because the two need
    #: different things from the user: rephrasing versus scepticism.
    refused: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    context_preview: tuple[str, ...] = field(default_factory=tuple)


class RagService:
    """Answers natural-language questions from the retrieved corpus."""

    def __init__(
        self,
        *,
        search: HybridSearchService,
        llm: LlmClient,
        relevance_floor: float,
        context_passages: int = 8,
        passage_chars: int = 900,
    ) -> None:
        """Wire the service to its collaborators.

        Args:
            search: Hybrid retrieval.
            llm: Answer generator.
            relevance_floor: Minimum cosine for a retrieval to count as
                evidence, taken from the embedding provider's own scale.
            context_passages: Passages placed in the prompt. Beyond roughly
                this many, retrieval noise crowds out signal and answers get
                vaguer rather than better.
            passage_chars: Characters kept per passage.
        """
        self._search = search
        self._llm = llm
        self._relevance_floor = relevance_floor
        self._context_passages = context_passages
        self._passage_chars = passage_chars

    async def answer(
        self,
        question: str,
        *,
        tickers: list[str] | None = None,
        days: int | None = None,
    ) -> RagAnswer:
        """Answer ``question`` from retrieved evidence.

        Args:
            question: The user's question.
            tickers: Optional symbol filter.
            days: Optional recency filter.

        Returns:
            An answer with citations and a computed confidence, or a refusal
            when nothing usable was retrieved.
        """
        since = datetime.now(UTC) - timedelta(days=days) if days else None
        response = await self._search.search(
            question,
            mode=SearchMode.HYBRID,
            limit=self._context_passages,
            tickers=tickers,
            since=since,
        )

        evidence = [result for result in response.results if result.score >= MIN_EVIDENCE_SCORE]
        if not evidence or not self._is_relevant(evidence):
            return self._refuse(question, retrieved=len(response.results))

        context = self._build_context(evidence)
        generated = await self._llm.complete(question=question, context=context)

        answer = RagAnswer(
            question=question,
            answer=generated.text,
            citations=tuple(
                Citation(
                    number=index,
                    title=result.title,
                    url=result.source_url,
                    source_id=result.source_id,
                    published_at=result.published_at,
                    matched_by=result.matched_by,
                )
                for index, result in enumerate(evidence, start=1)
            ),
            confidence=self._confidence(evidence),
            model_name=generated.model_name,
            extractive=generated.extractive,
            retrieved=len(evidence),
            prompt_tokens=generated.prompt_tokens,
            completion_tokens=generated.completion_tokens,
            context_preview=tuple(result.text[:200] for result in evidence[:3]),
        )
        logger.info(
            "rag_answer",
            retrieved=answer.retrieved,
            confidence=answer.confidence,
            model=answer.model_name,
            extractive=answer.extractive,
        )
        return answer

    def _is_relevant(self, evidence: list[SearchResult]) -> bool:
        """Whether anything retrieved is semantically close to the question.

        Keyword-only hits carry no similarity and cannot vouch for relevance on
        their own -- a text index matches on a shared word, which is exactly how
        an unrelated question retrieves plausible-looking noise. At least one
        result must clear the absolute bar.
        """
        return any(
            result.vector_similarity is not None
            and result.vector_similarity >= self._relevance_floor
            for result in evidence
        )

    def _build_context(self, evidence: list[SearchResult]) -> str:
        """Assemble the numbered passage block handed to the model.

        Numbering is what makes citation checkable: the model is asked to cite
        [1], [2], and every number maps to a stored source id, so a claim can be
        traced to a URL after the fact.
        """
        blocks: list[str] = []
        for index, result in enumerate(evidence, start=1):
            heading = result.title or "Untitled"
            published = result.published_at.date().isoformat() if result.published_at else "undated"
            body = result.text[: self._passage_chars]
            blocks.append(f"[{index}] {heading} ({published})\n{body}")
        return "\n\n".join(blocks)

    def _confidence(self, evidence: list[SearchResult]) -> float:
        """Derive confidence from the evidence, not from the model.

        Three observable signals, deliberately all pre-generation:

        - **Volume**: more corroborating passages is better, saturating quickly.
        - **Strength**: the top fused score, which reflects how well the best
          result matched.
        - **Agreement**: the share of evidence both retrievers found. When
          keyword and vector search independently surface the same document,
          that is the strongest signal available without a human.
        """
        volume = min(len(evidence) / CONFIDENCE_SATURATION, 1.0)
        strength = min(evidence[0].score / (2 * MIN_EVIDENCE_SCORE), 1.0)
        agreement = sum(1 for item in evidence if len(set(item.matched_by)) > 1) / len(evidence)

        score = 0.4 * volume + 0.35 * strength + 0.25 * agreement
        # Capped below 1.0: the platform is never certain, and a confidence of
        # exactly 1 would invite a user to stop checking.
        return round(min(score, 0.95), 3)

    def _refuse(self, question: str, *, retrieved: int) -> RagAnswer:
        """Decline to answer when nothing usable was retrieved."""
        logger.info("rag_refused", retrieved=retrieved)
        return RagAnswer(
            question=question,
            answer=(
                "I could not find sources in the indexed corpus that answer this. "
                "The platform only answers from retrieved documents, so rather "
                "than guess: try naming a specific ticker, widening the date "
                "range, or rephrasing in the language a news headline would use."
            ),
            citations=(),
            confidence=0.0,
            model_name=self._llm.model_name,
            extractive=not self._llm.is_generative,
            retrieved=0,
            refused=True,
        )
