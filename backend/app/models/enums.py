"""Domain enumerations shared by the ORM models and the API schemas.

Every enum is persisted as a native PostgreSQL ``ENUM`` type rather than a
free-text column. The database then rejects a typo at write time instead of the
platform discovering, months later, that ``"isolation_forest"`` and
``"IsolationForest"`` both exist in the anomalies table.
"""

from __future__ import annotations

from enum import StrEnum

from sqlalchemy import Enum as SqlEnum


def pg_enum[EnumT: StrEnum](enum_type: type[EnumT], name: str) -> SqlEnum:
    """Build the PostgreSQL ENUM column type for a Python enum.

    Args:
        enum_type: The ``StrEnum`` to persist.
        name: Name of the PostgreSQL type to create.

    Returns:
        A configured SQLAlchemy ``Enum``.

    Notes:
        ``values_callable`` is the important argument. By default SQLAlchemy
        persists a Python enum's member *names*, so ``DetectionMethod.Z_SCORE``
        would be stored as ``'Z_SCORE'`` while every ``StrEnum`` comparison in
        the codebase -- and every CHECK constraint written against it -- uses
        ``'z_score'``. Storing values keeps the database, the ORM and raw SQL
        speaking the same language.
    """
    return SqlEnum(
        enum_type,
        name=name,
        native_enum=True,
        validate_strings=True,
        values_callable=lambda enum: [member.value for member in enum],
    )


class AssetType(StrEnum):
    """What kind of instrument a ticker represents."""

    EQUITY = "equity"
    ETF = "etf"
    INDEX = "index"


class DataSource(StrEnum):
    """Origin of an ingested record, kept for provenance and reconciliation."""

    YFINANCE = "yfinance"
    NEWSAPI = "newsapi"
    RSS = "rss"
    SEC_EDGAR = "sec_edgar"
    INVESTOR_RELATIONS = "investor_relations"
    MANUAL = "manual"


class AnomalyType(StrEnum):
    """The market quantity that behaved abnormally."""

    RETURN = "return"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    GAP = "gap"


class DetectionMethod(StrEnum):
    """Algorithm that flagged an anomaly.

    Both methods run over the same feature window: the Z-score is interpretable
    and catches univariate outliers, Isolation Forest catches multivariate ones
    that no single feature would reveal. Storing the method makes their
    disagreement visible rather than averaged away.
    """

    Z_SCORE = "z_score"
    ISOLATION_FOREST = "isolation_forest"


class Severity(StrEnum):
    """Bucketed anomaly magnitude, used for filtering and alert thresholds."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class Direction(StrEnum):
    """Sign of a move."""

    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


class Sentiment(StrEnum):
    """FinBERT classification of a news article or filing."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class MarketRegime(StrEnum):
    """Coarse characterisation of a trading session."""

    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    MIXED = "mixed"
    QUIET = "quiet"


class EcosystemTag(StrEnum):
    """Segments of the AI-infrastructure value chain a company participates in.

    Stored as a PostgreSQL array on ``companies`` with a GIN index, so
    "which companies are exposed to HBM?" is one indexed containment query
    rather than a join through a tag table.
    """

    HBM = "hbm"
    DRAM = "dram"
    NAND = "nand"
    GPU = "gpu"
    CPU = "cpu"
    FOUNDRY = "foundry"
    LITHOGRAPHY = "lithography"
    ADVANCED_PACKAGING = "advanced_packaging"
    NETWORKING = "networking"
    SERVERS = "servers"
    HYPERSCALER = "hyperscaler"
    POWER_INFRASTRUCTURE = "power_infrastructure"
    EDA = "eda"
