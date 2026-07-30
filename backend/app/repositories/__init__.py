"""Repository layer: the only place that knows how persistence works.

Services depend on these classes, never on ``AsyncSession`` directly. That keeps
query construction in one place and lets a service be tested against an
in-memory fake exposing the same method surface.
"""

from app.repositories.anomaly import AnomalyRepository
from app.repositories.base import BaseRepository
from app.repositories.company import CompanyRepository, TickerRepository
from app.repositories.listing import ListingRepository
from app.repositories.market import MarketCalendarRepository, MarketSummaryRepository
from app.repositories.price import DailyPriceRepository, TechnicalIndicatorRepository
from app.repositories.user import (
    PortfolioRepository,
    UserRepository,
    WatchlistRepository,
)

__all__ = [
    "AnomalyRepository",
    "BaseRepository",
    "CompanyRepository",
    "DailyPriceRepository",
    "ListingRepository",
    "MarketCalendarRepository",
    "MarketSummaryRepository",
    "PortfolioRepository",
    "TechnicalIndicatorRepository",
    "TickerRepository",
    "UserRepository",
    "WatchlistRepository",
]
