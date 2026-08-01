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


class EntityKind(StrEnum):
    """What a knowledge-graph node represents.

    Wider than "company" on purpose. The AI ecosystem's most consequential
    actors include one that has never been listed (OpenAI), one that is a
    manufacturing process rather than an organisation (EUV lithography), and one
    that is a building (TSMC Arizona). A vocabulary restricted to issuers could
    not express why any of the three matters.
    """

    COMPANY = "company"
    #: Privately held or non-corporate: research labs, foundations, consortia.
    ORGANISATION = "organisation"
    #: A capability or process: EUV lithography, HBM, CoWoS packaging.
    TECHNOLOGY = "technology"
    #: A specific shipping thing: H100, MI300X, N3 process node.
    PRODUCT = "product"
    #: A released model: GPT-5, Claude, Llama.
    AI_MODEL = "ai_model"
    #: A physical site: a fab, a data centre campus.
    FACILITY = "facility"
    COUNTRY = "country"
    PERSON = "person"


class RelationKind(StrEnum):
    """How two entities are connected.

    Directed unless listed in :data:`SYMMETRIC_RELATIONS`. Direction is the
    difference between "TSMC supplies NVIDIA" and the reverse, which is the
    whole point of the edge.
    """

    #: A provides physical inputs to B. The backbone of the supply chain.
    SUPPLIES = "supplies"
    #: A fabricates B's designs. Narrower than SUPPLIES and worth its own type,
    #: because foundry concentration is the ecosystem's central fragility.
    MANUFACTURES = "manufactures"
    #: A buys from B. The inverse of SUPPLIES, stored when the customer side is
    #: what the source disclosed.
    CUSTOMER_OF = "customer_of"
    COMPETES_WITH = "competes_with"
    PARTNERS_WITH = "partners_with"
    #: A relies on B without a direct commercial relationship -- the
    #: second-order exposure that makes an ecosystem map worth drawing.
    DEPENDS_ON = "depends_on"
    USES = "uses"
    PRODUCES = "produces"
    INVESTS_IN = "invests_in"
    ACQUIRED = "acquired"
    #: A runs B's technology at scale: a cloud deploying accelerators.
    DEPLOYS = "deploys"
    #: A operates or owns B, for facilities and subsidiaries.
    OPERATES = "operates"
    #: A is located in B, for facilities and countries.
    LOCATED_IN = "located_in"


#: Relations where direction carries no meaning. Stored once and traversed both
#: ways; storing both directions would double the rows and let the two copies
#: drift apart, which is worse than the join.
SYMMETRIC_RELATIONS: frozenset[RelationKind] = frozenset(
    {RelationKind.COMPETES_WITH, RelationKind.PARTNERS_WITH}
)


class EvidenceSource(StrEnum):
    """Where a claim came from, and therefore how far to trust it.

    Ordered loosely by reliability. The UI renders the difference rather than
    presenting every edge as equally established -- a platform that says
    "NVIDIA depends on TSMC" without saying how it knows is asking to be
    believed rather than checked.
    """

    #: Hand-entered from a primary source, with the citation recorded.
    CURATED = "curated"
    #: Extracted from an SEC filing.
    FILING = "filing"
    #: Extracted from a company's own announcement.
    PRESS_RELEASE = "press_release"
    #: Extracted from journalism.
    NEWS = "news"
    #: Proposed by a language model from text. The lowest tier, and the reason
    #: confidence is a column: these arrive unverified and must look it.
    INFERRED = "inferred"


class ProposalStatus(StrEnum):
    """Where a model-proposed edge sits in review."""

    PENDING = "pending"
    #: Copied into the graph as an INFERRED edge. The proposal row is kept so
    #: the decision, and the sentence behind it, remain auditable.
    ACCEPTED = "accepted"
    REJECTED = "rejected"
