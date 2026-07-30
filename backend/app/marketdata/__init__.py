"""Provider-agnostic market data layer.

The application depends on :mod:`app.marketdata.domain` types and the
:class:`~app.marketdata.provider.MarketDataProvider` protocol. Vendors live
behind adapters in ``providers/`` and are selected by configuration, so swapping
one is a key and an adapter -- never a change to an endpoint or a component.
"""

from app.marketdata.domain import (
    Candle,
    CandleSeries,
    CompanyProfile,
    Interval,
    KeyMetrics,
    Listing,
    Quote,
)
from app.marketdata.provider import (
    BaseProvider,
    Capability,
    CapabilityNotSupportedError,
    MarketDataProvider,
    ProviderQuotaExceededError,
)

__all__ = [
    "BaseProvider",
    "Candle",
    "CandleSeries",
    "Capability",
    "CapabilityNotSupportedError",
    "CompanyProfile",
    "Interval",
    "KeyMetrics",
    "Listing",
    "MarketDataProvider",
    "ProviderQuotaExceededError",
    "Quote",
]
