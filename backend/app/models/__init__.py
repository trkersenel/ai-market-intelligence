"""SQLAlchemy ORM models.

Importing this package registers every model on ``Base.metadata``. Alembic's
``env.py`` imports it for exactly that reason: a model that is defined but never
imported is invisible to autogenerate, and silently missing from migrations.
"""

from app.db.base import Base
from app.models.anomaly import Anomaly
from app.models.company import Company, Ticker
from app.models.enums import (
    AnomalyType,
    AssetType,
    DataSource,
    DetectionMethod,
    Direction,
    EcosystemTag,
    EntityKind,
    EvidenceSource,
    MarketRegime,
    ProposalStatus,
    RelationKind,
    Sentiment,
    Severity,
)
from app.models.graph import Entity, Relationship, RelationshipProposal
from app.models.listing import Listing
from app.models.market import DailyMarketSummary, MarketCalendar
from app.models.price import DailyPrice, TechnicalIndicator
from app.models.user import (
    Portfolio,
    PortfolioPosition,
    User,
    Watchlist,
    WatchlistItem,
)

__all__ = [
    "Anomaly",
    "AnomalyType",
    "AssetType",
    "Base",
    "Company",
    "DailyMarketSummary",
    "DailyPrice",
    "DataSource",
    "DetectionMethod",
    "Direction",
    "EcosystemTag",
    "Entity",
    "EntityKind",
    "EvidenceSource",
    "Listing",
    "MarketCalendar",
    "MarketRegime",
    "Portfolio",
    "PortfolioPosition",
    "ProposalStatus",
    "RelationKind",
    "Relationship",
    "RelationshipProposal",
    "Sentiment",
    "Severity",
    "TechnicalIndicator",
    "Ticker",
    "User",
    "Watchlist",
    "WatchlistItem",
]
