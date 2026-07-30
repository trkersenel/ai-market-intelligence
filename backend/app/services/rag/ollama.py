"""Ollama adapters for local generation and embedding.

Both satisfy protocols that already existed, so nothing in the RAG pipeline, the
API or the frontend changes to use them. That is the payoff of having written
``LlmClient`` and ``EmbeddingProvider`` as protocols rather than as wrappers
around one vendor.

Running locally rather than calling OpenAI changes three things worth stating:

**Cost is zero and stays zero.** No key, no card, no per-token charge, and no
quota to exhaust mid-demo.

**Nothing leaves the machine.** Every article and every question stays local,
which for a tool that reads financial news is a real property rather than a
talking point.

**The models are smaller.** ``llama3.2:3b`` writes noticeably plainer prose than
``gpt-4o-mini``. For this platform that matters less than it might: the answer is
constrained to retrieved context and required to cite it, so the model is
summarising evidence rather than composing from memory. The grounding does the
work that model scale would otherwise have to.

The embedding side is the larger upgrade. The fallback in
:mod:`app.services.rag.embeddings` matches *wording* -- it cannot connect "memory
prices are climbing" to "DRAM ASPs firmed", because those share almost no
characters. ``nomic-embed-text`` connects them, which is the entire reason to use
embeddings instead of a keyword index.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from app.clients.http import HttpClient
from app.core.config import IngestionSettings, OllamaSettings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.services.rag.llm import SYSTEM_PROMPT, LlmResponse

logger = get_logger(__name__)


class OllamaUnavailableError(ExternalServiceError):
    """Ollama is not running or the model is not pulled.

    Its own type because the remedy is a specific local command rather than a
    retry, and the message carries that command. "Connection refused" tells a
    user nothing; "run `ollama pull llama3.2:3b`" tells them everything.
    """

    code = "ollama_unavailable"


def _caused_by_timeout(error: BaseException) -> bool:
    """Whether a timeout sits anywhere in the exception's cause chain.

    The chain is walked rather than the immediate cause inspected, because the
    original :class:`httpx.TimeoutException` is two links down: the HTTP client
    wraps tenacity's ``RetryError``, which in turn chains the last attempt's
    failure.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, httpx.TimeoutException):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _model_names(payload: Any) -> set[str]:
    """Extract every model name from a ``/api/tags`` body, in both spellings.

    Ollama reports the tagged form ("llama3.2:3b"); a user may configure the
    bare one ("llama3.2"). Both are returned so either spelling matches --
    otherwise selection falls back to OpenAI while the model sits pulled and
    unused, which is a silent downgrade rather than a visible failure.

    Separate from :meth:`_OllamaClientBase.probe` because this is the part with
    logic: the probe is I/O against a server that may not exist, while this is a
    pure function a test can exercise directly.
    """
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return set()

    names: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            continue
        name = str(model.get("name", ""))
        if name:
            names.add(name)
            names.add(name.split(":", 1)[0])
    return names


class _OllamaClientBase:
    """Shared transport and availability probing."""

    def __init__(
        self,
        settings: OllamaSettings,
        ingestion: IngestionSettings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the transport."""
        self._settings = settings
        self._http = HttpClient(
            settings=ingestion,
            base_url=settings.base_url,
            # No throttling: the server is this machine, and the only limit
            # that matters is how fast it can run inference.
            rate_limit=1000.0,
            provider="ollama",
            timeout_seconds=settings.request_timeout_seconds,
            client=client,
        )

    @staticmethod
    async def probe(settings: OllamaSettings, ingestion: IngestionSettings) -> set[str]:
        """Return the models Ollama currently has, or an empty set if it is down.

        Never raises. Called during provider selection, where an unreachable
        Ollama must mean "fall back" rather than "fail to start" -- the same
        rule every optional dependency in the platform follows.
        """
        try:
            async with httpx.AsyncClient(
                base_url=settings.base_url, timeout=settings.probe_timeout_seconds
            ) as client:
                response = await client.get("/api/tags")
                response.raise_for_status()
                payload = response.json()
        except Exception:  # noqa: BLE001 - unreachable is an answer, not an error
            return set()

        return _model_names(payload)

    async def _post(self, path: str, payload: dict[str, Any]) -> Any:
        """Issue a request, translating a failure into an actionable instruction.

        A stopped server and a slow one need opposite advice, so they are told
        apart rather than collapsed into one message. Reporting a timeout as
        "start it with ``ollama serve``" sends the reader to check something
        that is already true, which is worse than saying nothing.
        """
        try:
            return await self._http.post_json(path, json=payload)
        except ExternalServiceError as exc:
            if _caused_by_timeout(exc):
                seconds = self._settings.request_timeout_seconds
                msg = (
                    f"Ollama did not respond within {seconds:.0f}s. The server is "
                    "running but inference is slower than that -- raise "
                    "OLLAMA_REQUEST_TIMEOUT_SECONDS, or use a smaller model."
                )
            else:
                msg = (
                    f"Ollama is not reachable at {self._settings.base_url}. "
                    "Start it with `ollama serve`."
                )
            raise OllamaUnavailableError(
                msg, details={"base_url": self._settings.base_url}
            ) from exc

    async def aclose(self) -> None:
        """Release the transport."""
        await self._http.aclose()


class OllamaChatClient(_OllamaClientBase):
    """Grounded answer generation against a local model."""

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

        Uses the same system prompt as the OpenAI client. A smaller model needs
        the prohibitions *more*, not less: it is likelier to drift toward
        remembered facts when the context is thin, and the prompt is what holds
        it to the passages.
        """
        payload = await self._post(
            "/api/chat",
            {
                "model": self._settings.chat_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context passages:\n{context}\n\nQuestion: {question}",
                    },
                ],
                # Streaming off: the caller wants one answer, and assembling a
                # stream here would add complexity with nothing consuming it.
                "stream": False,
                "options": {
                    "temperature": self._settings.temperature,
                    "num_predict": self._settings.max_output_tokens,
                },
            },
        )

        if not isinstance(payload, dict):
            msg = "Ollama returned a malformed completion"
            raise OllamaUnavailableError(msg)

        content = (payload.get("message") or {}).get("content", "")
        if not content or not content.strip():
            model = self._settings.chat_model
            msg = (
                f"Ollama returned an empty completion. Is `{model}` pulled? "
                f"Run `ollama pull {model}`."
            )
            raise OllamaUnavailableError(msg, details={"model": model})

        return LlmResponse(
            text=content.strip(),
            model_name=self.model_name,
            # Ollama reports token counts under different names than OpenAI.
            prompt_tokens=payload.get("prompt_eval_count"),
            completion_tokens=payload.get("eval_count"),
        )


class OllamaEmbeddingProvider(_OllamaClientBase):
    """Semantic embeddings from a local model."""

    def __init__(
        self,
        settings: OllamaSettings,
        ingestion: IngestionSettings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the provider."""
        super().__init__(settings, ingestion, client=client)
        # Discovered from the first response rather than configured. A wrong
        # dimension is not a cosmetic error: vectors of different widths are
        # not comparable, so a mismatch corrupts every search silently instead
        # of raising.
        self._dimensions: int | None = None

    @property
    def model_name(self) -> str:
        """Identifier stored with every chunk."""
        return self._settings.embedding_model

    @property
    def dimensions(self) -> int:
        """Vector width.

        ``nomic-embed-text`` produces 768 dimensions, not the 1536 of OpenAI's
        ``text-embedding-3-small``. Switching between them therefore requires
        re-embedding the corpus *and* rebuilding the vector index -- a mixed
        collection returns nonsense rather than an error, because cosine over
        differently sized vectors is simply undefined.
        """
        return self._dimensions or self._settings.embedding_dimensions

    @property
    def relevance_floor(self) -> float:
        """Cosine below which a result is noise, on this model's scale.

        Measured against 400 stored articles, asking each question the way a
        user would rather than echoing a headline. Off-topic questions topped
        out at 0.526 ("Who won the Premier League on Saturday?", which finds
        purchase on the sports-adjacent language in business coverage);
        on-topic questions bottomed out at 0.616. The default sits in that gap.

        The absolute numbers are much higher than the hashing fallback's
        (0.091 off-topic, 0.247 on-topic) because ``nomic-embed-text`` reports
        substantial similarity even between unrelated prose -- which is why the
        floor cannot be carried across providers, only re-measured:
        ``python -m app.services.rag.measure_floor``
        """
        return self._settings.relevance_floor

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input, in the same order.

        Raises:
            OllamaUnavailableError: If the server is down, the model is not
                pulled, or the response count does not match the input count --
                which would misalign every chunk with its neighbour's vector.
        """
        if not texts:
            return []

        vectors: list[list[float]] = []
        batch_size = self._settings.embedding_batch_size
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            payload = await self._post(
                "/api/embed",
                {"model": self._settings.embedding_model, "input": batch},
            )
            vectors.extend(self._parse(payload, expected=len(batch)))

        if vectors and self._dimensions is None:
            self._dimensions = len(vectors[0])
            logger.info(
                "ollama_embedding_dimensions",
                model=self._settings.embedding_model,
                dimensions=self._dimensions,
            )
        return vectors

    def _parse(self, payload: Any, *, expected: int) -> list[list[float]]:
        """Extract vectors, insisting the count matches the input."""
        if not isinstance(payload, dict):
            msg = "Ollama returned a malformed embedding response"
            raise OllamaUnavailableError(msg)

        rows = payload.get("embeddings")
        if not isinstance(rows, list) or len(rows) != expected:
            model = self._settings.embedding_model
            returned = len(rows) if isinstance(rows, list) else "an unreadable number of"
            msg = (
                f"Ollama returned {returned} vectors for {expected} inputs. "
                f"Is `{model}` pulled? Run `ollama pull {model}`."
            )
            raise OllamaUnavailableError(msg, details={"model": model})

        return [[float(value) for value in row] for row in rows]
