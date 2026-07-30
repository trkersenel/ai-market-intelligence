"""Language model clients for grounded answer generation.

The platform's central promise is that it never answers from parametric memory:
every claim must trace to a retrieved document. That constraint shapes both
implementations here.

**OpenAI** generates fluent synthesis from retrieved context, under a system
prompt that forbids using outside knowledge and requires inline citations.

**The extractive answerer** is the fallback, and it is not a placeholder. It
composes an answer purely by selecting and ordering sentences from the retrieved
passages. That makes it *structurally incapable* of fabrication -- there is no
generation step in which a fact could be invented. It reads worse than a
generated answer and it cannot synthesise across documents, but it is never
wrong about what the sources said.

Choosing that as the degradation, rather than "return an error", follows from
what the product is for. An analyst reading three relevant excerpts with sources
is served. An analyst reading a confident paragraph containing an invented
number is worse than unserved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from app.clients.http import HttpClient
from app.core.config import IngestionSettings, LlmSettings, OllamaSettings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger

logger = get_logger(__name__)

#: The instruction that makes the answer groundable. Stated as prohibitions
#: because a model asked to "prefer the context" will still fall back on what it
#: remembers; one told it has no other knowledge generally will not.
SYSTEM_PROMPT = """\
You are a market intelligence analyst covering the AI infrastructure and \
semiconductor sector.

Rules you must follow without exception:
1. Answer ONLY from the numbered context passages provided. You have no other \
knowledge of these companies or events.
2. Cite the passages you used inline, as [1], [2]. Every factual claim needs a \
citation.
3. If the context does not answer the question, say exactly what is missing. Do \
not speculate, and do not fill gaps from memory.
4. Never invent a number, date, percentage or company name that is not in the \
context.
5. Be concise and specific. An analyst wants the mechanism, not a summary of \
the summary.
"""


@dataclass(frozen=True)
class LlmResponse:
    """A generated answer with its cost and provenance."""

    text: str
    model_name: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    #: True when the answer was assembled rather than generated. Surfaced so the
    #: API can tell a client which it received -- the two have very different
    #: fluency and very different failure modes.
    extractive: bool = False


@runtime_checkable
class LlmClient(Protocol):
    """Generates an answer from a question and retrieved context."""

    @property
    def model_name(self) -> str:
        """Identifier stored with every answer, for provenance."""
        ...

    @property
    def is_generative(self) -> bool:
        """Whether this client synthesises text or only selects it."""
        ...

    async def complete(self, *, question: str, context: str) -> LlmResponse:
        """Answer ``question`` using only ``context``."""
        ...


class OpenAIChatClient:
    """Answer generation via the OpenAI chat completions API."""

    def __init__(
        self,
        settings: LlmSettings,
        ingestion: IngestionSettings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the client.

        Raises:
            ExternalServiceError: If no API key is configured.
        """
        if settings.openai_api_key is None:
            msg = "OpenAI API key is not configured"
            raise ExternalServiceError(msg)

        self._settings = settings
        self._http = HttpClient(
            settings=ingestion,
            base_url=settings.openai_base_url,
            rate_limit=4.0,
            headers={
                "Authorization": f"Bearer {settings.openai_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            provider="openai_chat",
            client=client,
        )

    @staticmethod
    def is_configured(settings: LlmSettings) -> bool:
        """Return whether a credential is available."""
        return settings.openai_api_key is not None

    @property
    def model_name(self) -> str:
        """Identifier stored with every answer."""
        return self._settings.chat_model

    @property
    def is_generative(self) -> bool:
        """This client synthesises text."""
        return True

    async def complete(self, *, question: str, context: str) -> LlmResponse:
        """Generate a grounded answer.

        Raises:
            ExternalServiceError: On a malformed or empty response.
        """
        payload = await self._http.post_json(
            "/chat/completions",
            json={
                "model": self._settings.chat_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context passages:\n{context}\n\nQuestion: {question}",
                    },
                ],
                # Low but not zero. Deterministic decoding on a grounded task
                # produces stilted, repetitive prose without improving accuracy,
                # since the grounding comes from the prompt, not the sampling.
                "temperature": self._settings.temperature,
                "max_tokens": self._settings.max_output_tokens,
            },
        )
        return self._parse(payload)

    def _parse(self, payload: Any) -> LlmResponse:
        """Extract the answer and token usage from a completion response."""
        if not isinstance(payload, dict) or not payload.get("choices"):
            msg = "OpenAI returned a malformed completion"
            raise ExternalServiceError(msg)

        content = payload["choices"][0].get("message", {}).get("content", "")
        if not content or not content.strip():
            msg = "OpenAI returned an empty completion"
            raise ExternalServiceError(msg)

        usage = payload.get("usage") or {}
        return LlmResponse(
            text=content.strip(),
            model_name=self.model_name,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    async def aclose(self) -> None:
        """Release the transport."""
        await self._http.aclose()


#: Words carrying no topical signal. Scoring against them would rank every
#: sentence containing "the" as relevant to every question.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would",
})  # fmt: skip


class ExtractiveAnswerer:
    """Assembles an answer by selecting sentences from the retrieved context.

    Scores each sentence by overlap with the question's content words, keeps the
    best few, and presents them in the order the passages were ranked, each
    carrying its citation.

    The result is not prose. It is a set of quoted findings -- which is a
    defensible thing to show an analyst, and cannot contain a claim the sources
    do not make.
    """

    #: Sentences retained. Enough to answer a question from several angles, few
    #: enough that the result is readable.
    _MAX_SENTENCES = 5

    _SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

    #: A fragment shorter than this is a caption or a byline, not a finding.
    _MIN_SENTENCE_CHARS = 30

    #: Two-letter tokens are almost all abbreviations or noise once stopwords
    #: are removed; keeping them adds false overlap.
    _MIN_WORD_CHARS = 3

    @property
    def model_name(self) -> str:
        """Identifier stored with every answer."""
        return "extractive-v1"

    @property
    def is_generative(self) -> bool:
        """This client only selects text; it never synthesises."""
        return False

    async def complete(self, *, question: str, context: str) -> LlmResponse:
        """Select the passages' most relevant sentences."""
        terms = self._content_words(question)
        scored: list[tuple[float, int, str]] = []

        for citation, passage in self._passages(context):
            for sentence in self._SENTENCE_SPLIT.split(passage):
                cleaned = sentence.strip()
                if len(cleaned) < self._MIN_SENTENCE_CHARS:
                    continue
                overlap = len(terms & self._content_words(cleaned))
                if overlap:
                    scored.append((overlap / len(terms), citation, cleaned))

        if not scored:
            return LlmResponse(
                text=(
                    "The retrieved sources do not contain information that answers "
                    "this question. Try a different phrasing, a wider date range, "
                    "or a specific ticker."
                ),
                model_name=self.model_name,
                extractive=True,
            )

        scored.sort(key=lambda item: (-item[0], item[1]))
        best = scored[: self._MAX_SENTENCES]
        # Restored to citation order: the retrieval ranking already expresses
        # relevance, and reading findings in source order is less disorienting
        # than reading them in score order.
        best.sort(key=lambda item: item[1])

        lines = [f"- {sentence} [{citation}]" for _, citation, sentence in best]
        body = "\n".join(lines)
        return LlmResponse(
            text=(
                "Relevant findings from the retrieved sources "
                f"(no language model is configured, so these are quoted directly "
                f"rather than synthesised):\n\n{body}"
            ),
            model_name=self.model_name,
            extractive=True,
        )

    def _content_words(self, text: str) -> set[str]:
        """Lower-cased words carrying topical signal."""
        words = re.findall(r"[a-z0-9']+", text.lower())
        return {
            word for word in words if word not in _STOPWORDS and len(word) >= self._MIN_WORD_CHARS
        }

    def _passages(self, context: str) -> list[tuple[int, str]]:
        """Split the numbered context block back into ``(citation, text)`` pairs."""
        passages: list[tuple[int, str]] = []
        for block in re.split(r"\n(?=\[\d+\])", context):
            match = re.match(r"\[(\d+)\]\s*(.*)", block.strip(), re.DOTALL)
            if match:
                passages.append((int(match.group(1)), match.group(2)))
        return passages


async def build_llm_client(
    settings: LlmSettings,
    ingestion: IngestionSettings,
    ollama: OllamaSettings | None = None,
) -> LlmClient:
    """Return the best available answer generator.

    Preference order, and the reasoning behind it:

    1. **Ollama**, when a server is running and the model is pulled. It costs
       nothing, needs no key, and keeps every question on the machine.
    2. **OpenAI**, when a key is configured. Better prose, at a price.
    3. **Extraction**, always available. Reads worse and cannot fabricate.

    Local is preferred over OpenAI rather than the reverse because this platform
    constrains answers to retrieved context and requires citations, so model
    scale buys less here than it would on an open-ended task -- and zero cost
    with zero data egress buys a great deal.

    Async because step 1 requires probing a server. A synchronous version would
    have to assume Ollama is present, and assuming is how a demo dies.
    """
    # Imported here rather than at module scope: the Ollama adapter imports
    # SYSTEM_PROMPT and LlmResponse from this module, so a top-level import
    # would be circular.
    from app.services.rag.ollama import OllamaChatClient  # noqa: PLC0415

    if ollama is not None and ollama.enabled:
        available = await OllamaChatClient.probe(ollama, ingestion)
        if ollama.chat_model in available:
            logger.info("llm_selected", provider="ollama", model=ollama.chat_model)
            return OllamaChatClient(ollama, ingestion)
        if available:
            logger.warning(
                "ollama_model_missing",
                model=ollama.chat_model,
                available=sorted(available),
                hint=f"run `ollama pull {ollama.chat_model}`",
            )

    if OpenAIChatClient.is_configured(settings):
        logger.info("llm_selected", provider="openai", model=settings.chat_model)
        return OpenAIChatClient(settings, ingestion)

    logger.warning(
        "generative_llm_unavailable",
        reason="no Ollama server and no LLM_OPENAI_API_KEY",
        fallback="extractive-v1",
        consequence="answers quote sources verbatim instead of synthesising",
    )
    return ExtractiveAnswerer()
