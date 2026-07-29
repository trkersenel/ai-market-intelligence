"""Adapters for external data sources.

Each client translates one provider's wire format into the platform's normalised
DTOs and raises the platform's own exceptions. Services depend on the protocols
in :mod:`app.clients.protocols`, never on these concrete classes, so the whole
ingestion pipeline can be exercised without a network.
"""

from app.clients.http import HttpClient
from app.clients.protocols import NewsProvider, PriceBar, PriceProvider, RawArticle
from app.clients.rate_limiter import RateLimiter

__all__ = [
    "HttpClient",
    "NewsProvider",
    "PriceBar",
    "PriceProvider",
    "RateLimiter",
    "RawArticle",
]
