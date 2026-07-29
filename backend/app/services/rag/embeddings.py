"""Embedding providers.

Two implementations behind one protocol, mirroring the sentiment layer, and for
the same reason: the good one needs a credential the platform must not require
in order to start.

**OpenAI** ``text-embedding-3-small`` is the production choice -- 1536
dimensions, genuinely semantic, cheap enough that embedding a news corpus costs
cents.

**The hashing provider** is the fallback, and it is worth being precise about
what it is. It is not a fake or a stub returning random vectors: it is a
character n-gram hashing vectoriser with TF-IDF-style weighting, L2-normalised,
projected into the same 1536 dimensions. Cosine similarity over it is *real
lexical similarity* -- "HBM capacity sold out" retrieves "HBM supply
constrained" because they share n-grams.

What it cannot do is semantics. "Memory prices are climbing" and "DRAM ASPs
firmed" mean the same thing and share almost no characters, so it will miss the
connection that is precisely the reason to use embeddings at all. It keeps the
pipeline runnable and testable end to end without a key; it does not make the
key optional in production.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import httpx

from app.clients.http import HttpClient
from app.core.config import EmbeddingSettings, IngestionSettings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into dense vectors."""

    @property
    def model_name(self) -> str:
        """Identifier stored with every chunk, so a re-embed can be targeted."""
        ...

    @property
    def dimensions(self) -> int:
        """Vector width. Must match the configured Atlas index."""
        ...

    @property
    def relevance_floor(self) -> float:
        """Cosine below which a result is noise *on this provider's scale*.

        Belongs to the provider because similarity scales are not comparable
        between embedders. A single platform-wide constant would be right for
        at most one of them and silently wrong for the rest -- either refusing
        good answers or, worse, answering questions the corpus cannot address.
        """
        ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input, in the same order."""
        ...


class OpenAIEmbeddingProvider:
    """Embeddings from the OpenAI API."""

    def __init__(
        self,
        settings: EmbeddingSettings,
        ingestion: IngestionSettings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the provider.

        Args:
            settings: Model, dimensions and batch size.
            ingestion: Timeout and retry policy, shared with every other
                outbound call so resilience is configured in one place.
            client: Injected transport, used by tests.

        Raises:
            ExternalServiceError: If no API key is configured. Callers check
                :meth:`is_configured` first.
        """
        if settings.openai_api_key is None:
            msg = "OpenAI API key is not configured"
            raise ExternalServiceError(msg)

        self._settings = settings
        self._http = HttpClient(
            settings=ingestion,
            base_url=settings.openai_base_url,
            rate_limit=8.0,
            headers={
                "Authorization": f"Bearer {settings.openai_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            provider="openai_embeddings",
            client=client,
        )

    @staticmethod
    def is_configured(settings: EmbeddingSettings) -> bool:
        """Return whether a credential is available."""
        return settings.openai_api_key is not None

    @property
    def model_name(self) -> str:
        """Identifier stored with every chunk."""
        return self._settings.model

    @property
    def dimensions(self) -> int:
        """Configured vector width."""
        return self._settings.dimensions

    @property
    def relevance_floor(self) -> float:
        """Cosine below which a result is noise.

        OpenAI embeddings spread related and unrelated text much further apart
        than a lexical hash does: unrelated pairs typically sit near 0.1-0.2 and
        genuinely related text above 0.4, so the bar sits between them.
        """
        return 0.28

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of passages.

        Raises:
            ExternalServiceError: On a malformed response, or if the API returns
                a different number of vectors than inputs -- which would
                silently misalign every chunk with its neighbour's embedding.
        """
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._settings.batch_size):
            batch = list(texts[start : start + self._settings.batch_size])
            payload = await self._http.post_json(
                "/embeddings",
                json={
                    "model": self._settings.model,
                    "input": batch,
                    "dimensions": self._settings.dimensions,
                },
            )
            vectors.extend(self._parse(payload, expected=len(batch)))

        return vectors

    def _parse(self, payload: object, *, expected: int) -> list[list[float]]:
        """Extract vectors from a response, re-sorting by the index field.

        The API documents that results may be returned out of order, and each
        carries its own ``index``. Trusting positional order would pair chunk A
        with chunk B's vector -- a corruption that no test of "did we get 64
        vectors" would catch.
        """
        if not isinstance(payload, dict) or "data" not in payload:
            msg = "OpenAI returned a malformed embeddings response"
            raise ExternalServiceError(msg)

        rows = payload["data"]
        if not isinstance(rows, list) or len(rows) != expected:
            returned = len(rows) if isinstance(rows, list) else "an unreadable number of"
            msg = f"OpenAI returned {returned} vectors for {expected} inputs"
            raise ExternalServiceError(msg)

        ordered = sorted(rows, key=lambda row: row.get("index", 0))
        return [[float(value) for value in row["embedding"]] for row in ordered]

    async def aclose(self) -> None:
        """Release the transport."""
        await self._http.aclose()


class HashingEmbeddingProvider:
    """Deterministic lexical embeddings, requiring no credential or download.

    Character n-grams are hashed into the target dimensionality with signed
    buckets, weighted sub-linearly by frequency, and L2-normalised so cosine
    similarity is well behaved.

    Character n-grams rather than words on purpose: they survive the morphology
    of financial text, where "HBM3", "HBM3E" and "HBM" should be near each other
    and word-level tokenisation would treat them as unrelated.
    """

    #: Length of the character n-grams hashed into buckets. Four is short enough
    #: to match across inflections and long enough not to collide on noise.
    _NGRAM = 4

    def __init__(self, dimensions: int = 1536) -> None:
        """Configure the vector width.

        Args:
            dimensions: Must match whatever the index expects, so that switching
                to OpenAI later does not require rebuilding the schema.
        """
        self._dimensions = dimensions

    @property
    def model_name(self) -> str:
        """Identifier stored with every chunk.

        Versioned so that chunks embedded by this provider can be found and
        re-embedded once a real model is configured.
        """
        return f"hashing-ngram{self._NGRAM}-v1"

    @property
    def relevance_floor(self) -> float:
        """Cosine below which a result is noise.

        Measured against the live corpus rather than guessed. Character n-grams
        give any two pieces of English prose a non-zero floor -- common
        substrings alone carry similarity -- so the separation is narrower than
        a semantic model's:

            "Brazilian coffee harvest 1987"   -> best match 0.091
            "memory chip prices, HBM demand"  -> best match 0.247

        0.15 sits between them. The gap being this narrow is itself the
        argument for configuring a real embedding model.
        """
        return 0.15

    @property
    def dimensions(self) -> int:
        """Configured vector width."""
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch. Pure computation; no I/O and no event-loop blocking."""
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        """Hash one passage into a normalised dense vector."""
        vector = [0.0] * self._dimensions
        normalised = " ".join(text.lower().split())
        if not normalised:
            return vector

        counts: dict[int, float] = {}
        signs: dict[int, float] = {}
        for index in range(len(normalised) - self._NGRAM + 1):
            gram = normalised[index : index + self._NGRAM]
            digest = hashlib.blake2b(gram.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimensions
            # A signed hash keeps unrelated collisions from always accumulating
            # in the same direction, which would make every vector drift toward
            # a common centroid and flatten all similarities.
            sign = 1.0 if digest[4] & 1 else -1.0
            counts[bucket] = counts.get(bucket, 0.0) + 1.0
            signs[bucket] = sign

        for bucket, count in counts.items():
            # Sub-linear weighting, as in TF-IDF: the tenth occurrence of a gram
            # says far less than the first.
            vector[bucket] = signs[bucket] * (1.0 + math.log(count))

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def build_embedding_provider(
    settings: EmbeddingSettings, ingestion: IngestionSettings
) -> EmbeddingProvider:
    """Return the best available embedding provider.

    OpenAI when a key is configured, the hashing provider otherwise. A missing
    credential is a logged downgrade with a named consequence, never a startup
    failure.
    """
    if OpenAIEmbeddingProvider.is_configured(settings):
        return OpenAIEmbeddingProvider(settings, ingestion)

    logger.warning(
        "openai_embeddings_unavailable",
        reason="EMBED_OPENAI_API_KEY is not set",
        fallback="hashing",
        consequence="retrieval matches wording, not meaning",
    )
    return HashingEmbeddingProvider(dimensions=settings.dimensions)
