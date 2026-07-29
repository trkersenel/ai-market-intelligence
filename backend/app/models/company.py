"""Company and ticker models -- the reference data every other table hangs off.

The split matters: a *company* is the economic entity analysts reason about
("Micron"), a *ticker* is a tradable listing of it ("MU" on NASDAQ). SK Hynix
and Samsung list in Seoul, TSMC has both a Taipei listing and a US ADR, and ETFs
are tickers with no company at all. Collapsing the two would make every one of
those a special case.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, IntIdMixin, TimestampMixin
from app.models.enums import AssetType, DataSource, EcosystemTag, pg_enum

if TYPE_CHECKING:
    from app.models.anomaly import Anomaly
    from app.models.price import DailyPrice, TechnicalIndicator


class Company(IntIdMixin, TimestampMixin, Base):
    """An issuer tracked by the platform."""

    __table_args__ = (
        Index("ix_company_tags_gin", "tags", postgresql_using="gin"),
        Index("ix_company_sector", "sector"),
        {"comment": "Issuers tracked across the AI-infrastructure value chain."},
    )

    slug: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    legal_name: Mapped[str | None] = mapped_column(String(255))

    sector: Mapped[str | None] = mapped_column(String(100))
    industry: Mapped[str | None] = mapped_column(String(120))
    country: Mapped[str | None] = mapped_column(String(2), comment="ISO 3166-1 alpha-2.")
    website: Mapped[str | None] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)

    #: Value-chain segments. A PostgreSQL array with a GIN index answers
    #: "every company exposed to HBM" without a join table.
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(String(40)),
        default=list,
        server_default="{}",
    )

    #: Excluded from ingestion when false, without deleting historical rows.
    is_tracked: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")

    tickers: Mapped[list[Ticker]] = relationship(
        back_populates="company",
        cascade="all, delete-orphan",
        lazy="raise_on_sql",
    )

    def __repr__(self) -> str:
        """Return a debugging representation."""
        return f"<Company id={self.id} slug={self.slug!r}>"

    def has_tag(self, tag: EcosystemTag) -> bool:
        """Return whether the company is tagged with ``tag``."""
        return tag.value in self.tags


class Ticker(IntIdMixin, TimestampMixin, Base):
    """A tradable listing: a symbol on an exchange, in a currency."""

    __table_args__ = (
        CheckConstraint(
            "(asset_type = 'equity' AND company_id IS NOT NULL) OR asset_type <> 'equity'",
            name="equity_requires_company",
        ),
        Index("ix_ticker_active_symbol", "is_active", "symbol"),
        {"comment": "Tradable listings; ETFs and indices have no parent company."},
    )

    #: Nullable by design -- an ETF such as SMH has no issuing company.
    company_id: Mapped[int | None] = mapped_column(
        ForeignKey("company.id", ondelete="CASCADE"), index=True
    )

    symbol: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(200))
    exchange: Mapped[str | None] = mapped_column(String(30))
    currency: Mapped[str] = mapped_column(String(3), default="USD", server_default="USD")

    asset_type: Mapped[AssetType] = mapped_column(
        pg_enum(AssetType, "asset_type"),
        default=AssetType.EQUITY,
    )
    data_source: Mapped[DataSource] = mapped_column(
        pg_enum(DataSource, "data_source"),
        default=DataSource.YFINANCE,
    )

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")

    #: Ingestion watermarks. Denormalised on purpose: the scheduler reads them on
    #: every run to decide the incremental fetch window, and a MAX() over tens of
    #: millions of price rows would be the most expensive query in the platform.
    first_price_date: Mapped[date | None] = mapped_column(Date)
    last_price_date: Mapped[date | None] = mapped_column(Date)
    last_ingested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    company: Mapped[Company | None] = relationship(back_populates="tickers", lazy="raise_on_sql")
    prices: Mapped[list[DailyPrice]] = relationship(
        back_populates="ticker",
        cascade="all, delete-orphan",
        lazy="raise_on_sql",
        passive_deletes=True,
    )
    indicators: Mapped[list[TechnicalIndicator]] = relationship(
        back_populates="ticker",
        cascade="all, delete-orphan",
        lazy="raise_on_sql",
        passive_deletes=True,
    )
    anomalies: Mapped[list[Anomaly]] = relationship(
        back_populates="ticker",
        cascade="all, delete-orphan",
        lazy="raise_on_sql",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        """Return a debugging representation."""
        return f"<Ticker id={self.id} symbol={self.symbol!r}>"
