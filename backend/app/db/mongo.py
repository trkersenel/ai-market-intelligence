"""Async MongoDB client lifecycle and collection accessors.

MongoDB stores the unstructured half of the platform: news articles, filings,
earnings-call transcripts, LLM summaries and the embedded chunks that Atlas
Vector Search queries. Collection names are centralised in :class:`Collection`
so that a rename is a one-line change rather than a string hunt.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)

from app.core.config import MongoSettings
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Motor's classes are generic over the document type. The platform stores
#: heterogeneous documents, so BSON documents are modelled as plain dicts and
#: given structure by Pydantic schemas at the service boundary.
type MongoDocument = dict[str, Any]


class Collection(StrEnum):
    """Canonical names of every MongoDB collection used by the platform."""

    NEWS_ARTICLES = "news_articles"
    COMPANY_REPORTS = "company_reports"
    EARNINGS_CALL_TRANSCRIPTS = "earnings_call_transcripts"
    RAG_DOCUMENTS = "rag_documents"
    LLM_SUMMARIES = "llm_summaries"
    CHAT_HISTORY = "chat_history"
    USER_PREFERENCES = "user_preferences"


class MongoDatabase:
    """Owns the Motor client and exposes typed collection accessors."""

    def __init__(self, settings: MongoSettings) -> None:
        """Create the Motor client.

        Args:
            settings: Connection URI, database name and pool configuration.

        Notes:
            Motor connects lazily; construction performs no I/O. Readiness is
            asserted by :meth:`ping` during the startup health gate.
        """
        self._settings = settings
        self._client: AsyncIOMotorClient[MongoDocument] = AsyncIOMotorClient(
            settings.uri.get_secret_value(),
            maxPoolSize=settings.max_pool_size,
            minPoolSize=settings.min_pool_size,
            serverSelectionTimeoutMS=settings.server_selection_timeout_ms,
            appname="market-intel-api",
            tz_aware=True,
        )
        self._database: AsyncIOMotorDatabase[MongoDocument] = self._client[settings.database]

    @property
    def client(self) -> AsyncIOMotorClient[MongoDocument]:
        """The underlying Motor client."""
        return self._client

    @property
    def database(self) -> AsyncIOMotorDatabase[MongoDocument]:
        """The configured application database."""
        return self._database

    @property
    def vector_index_name(self) -> str:
        """Name of the Atlas Vector Search index used for semantic retrieval."""
        return self._settings.vector_index_name

    def collection(self, name: Collection) -> AsyncIOMotorCollection[MongoDocument]:
        """Return a handle to one of the platform's collections.

        Args:
            name: Member of :class:`Collection`; passing a raw string is a type
                error, which prevents typo-driven collection creation.

        Returns:
            The Motor collection handle.
        """
        return self._database[name.value]

    async def ping(self) -> None:
        """Round-trip the server to confirm connectivity.

        Raises:
            pymongo.errors.PyMongoError: If the server cannot be reached within
                the configured selection timeout.
        """
        await self._client.admin.command("ping")

    async def close(self) -> None:
        """Close all sockets. Called on application shutdown."""
        self._client.close()
        logger.info("mongo_client_closed")
