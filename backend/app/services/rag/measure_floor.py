"""Derive the refusal threshold from the live corpus.

The relevance floor decides when the platform says "I don't have information
about that" instead of answering. Set too high it refuses real questions; set
too low it answers coffee questions out of semiconductor articles. Either
failure is invisible in the code, because both look like a working system until
somebody asks the wrong question.

So it is measured rather than chosen. This module asks a set of on-topic and
off-topic questions against the actual stored articles, reports the top-1 cosine
each one achieves, and proposes the midpoint of the gap between them. Run it
after changing the embedding model or once the corpus has grown substantially::

    python -m app.services.rag.measure_floor

The output is advisory: it prints a number and the evidence for it, and the
number goes into ``OLLAMA_RELEVANCE_FLOOR``. Nothing is written automatically,
because a threshold that changes itself is a threshold nobody can reason about.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence

from app.core.config import Settings, get_settings
from app.db.mongo import Collection, MongoDatabase
from app.services.rag.embeddings import EmbeddingProvider

#: Questions the platform exists to answer. Deliberately phrased the way a user
#: would, not the way an article is written -- matching an article's own
#: headline back to itself would measure nothing.
ON_TOPIC: tuple[str, ...] = (
    "Why did HBM demand rise?",
    "What happened to DRAM prices?",
    "How is Nvidia's data centre business performing?",
    "What did Micron report last quarter?",
    "Why did semiconductor stocks fall?",
    "What is SK Hynix's outlook for AI memory?",
)

#: Questions with no answer anywhere in the corpus. The highest score any of
#: these reaches is the floor's true lower bound: anything below it admits a
#: question the platform cannot answer.
OFF_TOPIC: tuple[str, ...] = (
    "What is the best way to brew coffee?",
    "Who won the Premier League on Saturday?",
    "How do I fix a leaking kitchen tap?",
    "What is the capital of Peru?",
    "Give me a recipe for sourdough bread.",
    "How tall is the Eiffel Tower?",
)

#: Enough articles for the maximum to be stable without making a local run slow.
SAMPLE_SIZE = 400


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two vectors."""
    dot = sum(x * y for x, y in zip(left, right, strict=True))
    norm = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    return dot / norm if norm else 0.0


async def measure(settings: Settings, embedder: EmbeddingProvider) -> dict[str, object]:
    """Score both question sets against the stored corpus.

    Returns:
        The per-question top-1 similarities, the gap between the two sets, and
        the suggested floor -- or ``None`` for the suggestion when the sets
        overlap, which means no single threshold separates them and the
        embedding model, not the number, is what needs changing.
    """
    mongo = MongoDatabase(settings.mongo)
    await mongo.ping()
    try:
        documents = (
            await mongo.collection(Collection.NEWS_ARTICLES)
            .find({}, {"title": 1, "summary": 1})
            .limit(SAMPLE_SIZE)
            .to_list(length=SAMPLE_SIZE)
        )
    finally:
        await mongo.close()

    texts = [
        f"{doc.get('title', '')} {doc.get('summary', '') or ''}".strip()[:900] for doc in documents
    ]
    if not texts:
        msg = "No articles stored; ingest news before measuring the floor."
        raise RuntimeError(msg)

    corpus = await embedder.embed(texts)
    questions = await embedder.embed([*ON_TOPIC, *OFF_TOPIC])

    def best(vector: Sequence[float]) -> float:
        return max(cosine(vector, row) for row in corpus)

    on = [best(vector) for vector in questions[: len(ON_TOPIC)]]
    off = [best(vector) for vector in questions[len(ON_TOPIC) :]]
    separable = min(on) > max(off)

    return {
        "articles": len(texts),
        "on_topic": dict(zip(ON_TOPIC, on, strict=True)),
        "off_topic": dict(zip(OFF_TOPIC, off, strict=True)),
        "lowest_on_topic": min(on),
        "highest_off_topic": max(off),
        "suggested_floor": round((min(on) + max(off)) / 2, 3) if separable else None,
    }


async def _main() -> None:
    """Run the measurement and print a report."""
    settings = get_settings()
    from app.services.rag.ollama import OllamaEmbeddingProvider  # noqa: PLC0415

    embedder = OllamaEmbeddingProvider(settings.ollama, settings.ingestion)
    try:
        result = await measure(settings, embedder)
    finally:
        await embedder.aclose()

    print(f"corpus: {result['articles']} articles, model: {embedder.model_name}\n")
    for label in ("on_topic", "off_topic"):
        print(label.replace("_", "-") + ":")
        scores: dict[str, float] = result[label]  # type: ignore[assignment]
        for question, score in sorted(scores.items(), key=lambda item: item[1]):
            print(f"  {score:.3f}  {question}")
        print()

    print(f"lowest on-topic   {result['lowest_on_topic']}")
    print(f"highest off-topic {result['highest_off_topic']}")
    floor = result["suggested_floor"]
    if floor is None:
        print("\nThe sets overlap: no threshold separates them. Change the embedding model.")
    else:
        print(f"\nOLLAMA_RELEVANCE_FLOOR={floor}")


if __name__ == "__main__":
    asyncio.run(_main())
