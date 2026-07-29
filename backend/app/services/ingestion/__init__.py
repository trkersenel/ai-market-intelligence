"""Ingestion services: orchestration of fetch, normalise, persist.

These own the policy -- which entities to fetch, over what window, how many at
once, what to do on failure. Transport lives in :mod:`app.clients` and
persistence in :mod:`app.repositories`, so nothing here contains HTTP or SQL.
"""

from app.services.ingestion.news_ingestion import (
    NewsIngestionReport,
    NewsIngestionService,
    ProviderResult,
)
from app.services.ingestion.price_ingestion import (
    IngestionReport,
    PriceIngestionService,
    TickerIngestionResult,
)

__all__ = [
    "IngestionReport",
    "NewsIngestionReport",
    "NewsIngestionService",
    "PriceIngestionService",
    "ProviderResult",
    "TickerIngestionResult",
]
