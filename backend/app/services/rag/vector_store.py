"""Vector search over the chunk collection.

Two backends behind one protocol, and here the fallback is not a convenience --
it is a hard requirement. ``$vectorSearch`` is an Atlas-only aggregation stage;
a local MongoDB rejects it outright. Without a second implementation the entire
retrieval half of the platform would be untestable and undemonstrable outside a
paid cluster, which is exactly the situation where a RAG pipeline quietly rots.

**Atlas** runs approximate nearest-neighbour search inside the database, applying
metadata filters *during* the vector scan. That last detail matters: filtering
afterwards returns the global top-k and then discards non-matching hits, often
leaving far fewer than k.

**Brute force** computes exact cosine similarity in the application. It is
O(corpus) and honest about it -- fine for the tens of thousands of chunks a
local corpus holds, wrong for millions. Capability is detected once at runtime
rather than configured, so the same deployment works against either.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pymongo.errors import OperationFailure

from app.core.logging import get_logger
from app.db.mongo import Collection, MongoDatabase

logger = get_logger(__name__)

#: MongoDB error codes returned when an aggregation stage is unrecognised or
#: unsupported by the deployment. Used to detect the local-versus-Atlas split
#: once, rather than probing on every query.
_UNSUPPORTED_STAGE_CODES = frozenset({40324, 31082, 8000})


@dataclass(frozen=True)
class VectorHit:
    """One retrieved chunk with its similarity score."""

    chunk_id: str
    source_id: str
    source_collection: str
    text: str
    score: float
    title: str | None = None
    source_url: str | None = None
    tickers: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    published_at: datetime | None = None


@dataclass(frozen=True)
class VectorFilter:
    """Metadata predicates applied during retrieval."""

    tickers: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    since: datetime | None = None

    @property
    def is_empty(self) -> bool:
        """Whether any predicate is set."""
        return not self.tickers and not self.tags and self.since is None


@runtime_checkable
class VectorStore(Protocol):
    """Retrieves chunks by embedding similarity."""

    @property
    def backend(self) -> str:
        """Identifier surfaced in search responses, so results are attributable."""
        ...

    async def search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
        candidates: int,
        filters: VectorFilter | None = None,
    ) -> list[VectorHit]:
        """Return the nearest chunks to ``embedding``."""
        ...


class AtlasVectorStore:
    """Approximate nearest-neighbour search using Atlas Vector Search."""

    def __init__(self, mongo: MongoDatabase) -> None:
        """Bind the store to the document database."""
        self._mongo = mongo
        self._collection = mongo.collection(Collection.RAG_DOCUMENTS)
        self._index = mongo.vector_index_name

    @property
    def backend(self) -> str:
        """Identifier surfaced in search responses."""
        return "atlas"

    async def search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
        candidates: int,
        filters: VectorFilter | None = None,
    ) -> list[VectorHit]:
        """Run ``$vectorSearch`` and map the results.

        Raises:
            OperationFailure: If the deployment does not support the stage.
                :func:`build_vector_store` catches this once at startup.
        """
        stage: dict[str, Any] = {
            "index": self._index,
            "path": "embedding",
            "queryVector": list(embedding),
            "numCandidates": candidates,
            "limit": limit,
        }
        if filters is not None and not filters.is_empty:
            stage["filter"] = _atlas_filter(filters)

        pipeline: list[dict[str, Any]] = [
            {"$vectorSearch": stage},
            {"$set": {"score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"embedding": 0}},
        ]
        cursor = self._collection.aggregate(pipeline)
        return [_to_hit(document, document.get("score", 0.0)) async for document in cursor]


class BruteForceVectorStore:
    """Exact cosine similarity computed in the application.

    The correctness baseline as well as the local fallback: an approximate index
    can be checked against it, which is otherwise hard to do.
    """

    #: Chunks pulled into memory per scan. Bounds peak memory on a corpus that
    #: has grown past what this backend should really be serving.
    _SCAN_LIMIT = 50_000

    def __init__(self, mongo: MongoDatabase) -> None:
        """Bind the store to the document database."""
        self._mongo = mongo
        self._collection = mongo.collection(Collection.RAG_DOCUMENTS)

    @property
    def backend(self) -> str:
        """Identifier surfaced in search responses."""
        return "brute_force"

    async def search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
        candidates: int,
        filters: VectorFilter | None = None,
    ) -> list[VectorHit]:
        """Score every stored chunk and return the closest.

        ``candidates`` is accepted and ignored: the scan is exhaustive, so there
        is no candidate pool to widen. Keeping the parameter means callers do not
        branch on which backend they got.
        """
        query = _mongo_filter(filters)
        cursor = self._collection.find(query).limit(self._SCAN_LIMIT)

        query_vector = list(embedding)
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        if query_norm == 0:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        async for document in cursor:
            stored = document.get("embedding")
            if not stored or len(stored) != len(query_vector):
                # A chunk embedded by a different model, or with a different
                # width. Silently comparing them would produce meaningless
                # scores, so it is skipped.
                continue
            score = _cosine(query_vector, stored, query_norm)
            scored.append((score, document))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [_to_hit(document, score) for score, document in scored[:limit]]


async def build_vector_store(mongo: MongoDatabase) -> VectorStore:
    """Return the vector store this deployment can actually use.

    Probes for ``$vectorSearch`` support once. Detecting rather than configuring
    means a developer running local MongoDB and a production deployment on Atlas
    run the same code path with no environment-specific flag to forget.
    """
    collection = mongo.collection(Collection.RAG_DOCUMENTS)
    probe: list[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": mongo.vector_index_name,
                "path": "embedding",
                "queryVector": [0.0],
                "numCandidates": 1,
                "limit": 1,
            }
        }
    ]

    try:
        cursor = collection.aggregate(probe)
        await cursor.to_list(length=1)
    except OperationFailure as exc:
        # An unsupported stage means local MongoDB. A *missing index* on Atlas
        # is a different failure and is deliberately not swallowed here -- it
        # would silently downgrade production to an O(corpus) scan.
        if exc.code in _UNSUPPORTED_STAGE_CODES or "only allowed on MongoDB Atlas" in str(exc):
            logger.info(
                "vector_backend_selected",
                backend="brute_force",
                reason="$vectorSearch is unavailable on this deployment",
            )
            return BruteForceVectorStore(mongo)
        logger.warning("vector_probe_failed", error=str(exc), code=exc.code)
        return BruteForceVectorStore(mongo)

    logger.info("vector_backend_selected", backend="atlas")
    return AtlasVectorStore(mongo)


def _atlas_filter(filters: VectorFilter) -> dict[str, Any]:
    """Build the filter Atlas applies during the vector scan."""
    clauses: list[dict[str, Any]] = []
    if filters.tickers:
        clauses.append({"tickers": {"$in": list(filters.tickers)}})
    if filters.tags:
        clauses.append({"tags": {"$in": list(filters.tags)}})
    if filters.since is not None:
        clauses.append({"published_at": {"$gte": filters.since}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _mongo_filter(filters: VectorFilter | None) -> dict[str, Any]:
    """Build the equivalent find() query for the brute-force backend."""
    if filters is None or filters.is_empty:
        return {}
    query: dict[str, Any] = {}
    if filters.tickers:
        query["tickers"] = {"$in": list(filters.tickers)}
    if filters.tags:
        query["tags"] = {"$in": list(filters.tags)}
    if filters.since is not None:
        query["published_at"] = {"$gte": filters.since}
    return query


def _cosine(query: Sequence[float], stored: Sequence[float], query_norm: float) -> float:
    """Cosine similarity between a query vector and a stored one."""
    dot = 0.0
    stored_norm_sq = 0.0
    for left, right in zip(query, stored, strict=True):
        dot += left * right
        stored_norm_sq += right * right
    if stored_norm_sq == 0:
        return 0.0
    return dot / (query_norm * math.sqrt(stored_norm_sq))


def _to_hit(document: dict[str, Any], score: float) -> VectorHit:
    """Map a stored chunk document onto a retrieval hit."""
    return VectorHit(
        chunk_id=str(document.get("_id", "")),
        source_id=str(document.get("source_id", "")),
        source_collection=str(document.get("source_collection", "")),
        text=str(document.get("text", "")),
        score=float(score),
        title=document.get("title"),
        source_url=document.get("source_url"),
        tickers=tuple(document.get("tickers", [])),
        tags=tuple(document.get("tags", [])),
        published_at=document.get("published_at"),
    )
