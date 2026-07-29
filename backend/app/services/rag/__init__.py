"""Retrieval: chunking, embeddings, vector storage and hybrid search.

Every external dependency here is optional at runtime. Without an OpenAI key the
platform embeds lexically; without Atlas it searches by brute force. Both
degrade with a named consequence rather than refusing to start, because a
retrieval pipeline that only runs against paid infrastructure is one nobody
exercises until it is already broken.
"""

from app.services.rag.chunking import Chunk, chunk_text
from app.services.rag.embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    OpenAIEmbeddingProvider,
    build_embedding_provider,
)
from app.services.rag.indexing_service import DocumentIndexingService, IndexingReport
from app.services.rag.search_service import (
    HybridSearchService,
    SearchMode,
    SearchResponse,
    SearchResult,
)
from app.services.rag.vector_store import (
    BruteForceVectorStore,
    VectorFilter,
    VectorHit,
    VectorStore,
    build_vector_store,
)

__all__ = [
    "BruteForceVectorStore",
    "Chunk",
    "DocumentIndexingService",
    "EmbeddingProvider",
    "HashingEmbeddingProvider",
    "HybridSearchService",
    "IndexingReport",
    "OpenAIEmbeddingProvider",
    "SearchMode",
    "SearchResponse",
    "SearchResult",
    "VectorFilter",
    "VectorHit",
    "VectorStore",
    "build_embedding_provider",
    "build_vector_store",
    "chunk_text",
]
