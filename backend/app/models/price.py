"""Daily prices and derived technical indicators.

These are the two highest-volume tables in the platform -- roughly 250 rows per
ticker per year, each -- so their constraints and indexes are chosen to make
ingestion idempotent and time-window queries cheap, in that order.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, IntIdMixin, TimestampMixin
from app.models.enums import DataSource, pg_enum

if TYPE_CHECKING:
    from app.models.company import Ticker


class DailyPrice(IntIdMixin, TimestampMixin, Base):
    """One OHLCV bar for one ticker on one session."""

    __table_args__ = (
        # The idempotency guarantee: re-running an ingestion job for a window
        # already stored updates rows instead of duplicating them. Every upsert
        # in DailyPriceRepository targets this constraint.
        # This constraint's backing btree on (ticker_id, trade_date) is also the
        # read index. "The last N sessions for this ticker" is an equality on
        # the leading column followed by a backward scan of the second, which
        # PostgreSQL serves from an ascending index at the same cost -- an
        # explicit DESC index here would be a redundant copy, paid for on every
        # ingested row. (DESC only earns its keep for *mixed* orderings.)
        UniqueConstraint("ticker_id", "trade_date"),
        # Cross-sectional reads: every ticker on one date, for the heatmap.
        Index("ix_daily_price_trade_date", "trade_date"),
        CheckConstraint("high >= low", name="high_at_least_low"),
        CheckConstraint("volume >= 0", name="volume_non_negative"),
        {"comment": "Daily OHLCV bars. Unique per (ticker, session)."},
    )

    ticker_id: Mapped[int] = mapped_column(ForeignKey("ticker.id", ondelete="CASCADE"))
    trade_date: Mapped[date]

    open: Mapped[Decimal]
    high: Mapped[Decimal]
    low: Mapped[Decimal]
    close: Mapped[Decimal]
    #: Split- and dividend-adjusted close. Returns are computed from this column;
    #: using raw close would invent a -50% "anomaly" on every stock split.
    adjusted_close: Mapped[Decimal]
    volume: Mapped[int] = mapped_column(BigInteger)

    source: Mapped[DataSource] = mapped_column(
        pg_enum(DataSource, "data_source"),
        default=DataSource.YFINANCE,
    )

    #: True while the session is still trading. Vendors return the current day's
    #: bar mid-session, so its volume and close are partial -- NVIDIA showing
    #: 16M shares against a 130M average is an incomplete day, not a collapse in
    #: participation. Statistics must never be derived from such a bar: it would
    #: register as a volume anomaly every single day the market is open.
    #:
    #: Self-correcting by design: the next run's overlapping window re-fetches
    #: the same session, now dated in the past, and the upsert clears the flag.
    #:
    #: Deliberately unindexed. At most one row per ticker is ever true, so a
    #: btree here could never be selective. Every query filtering on it also
    #: carries a ``ticker_id`` predicate, already served by the unique
    #: constraint's index; the boolean is a cheap recheck on the rows that
    #: index has already located.
    is_provisional: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="false",
        comment="True while the session is still trading; its OHLCV is partial.",
    )

    ticker: Mapped[Ticker] = relationship(back_populates="prices", lazy="raise_on_sql")

    def __repr__(self) -> str:
        """Return a debugging representation."""
        return f"<DailyPrice ticker_id={self.ticker_id} date={self.trade_date}>"


class TechnicalIndicator(IntIdMixin, TimestampMixin, Base):
    """Derived features for one ticker on one session.

    Modelled as a wide table rather than a ``(name, value)`` long table. The
    consumers -- anomaly detection, the charting API, the RAG context builder --
    always want many indicators for the same row at once; long form would turn
    every one of those reads into a pivot, and would lose per-indicator types.
    """

    __table_args__ = (
        # As on daily_price, the unique constraint's index serves the reads.
        UniqueConstraint("ticker_id", "trade_date"),
        {"comment": "Derived technical features, one row per (ticker, session)."},
    )

    ticker_id: Mapped[int] = mapped_column(ForeignKey("ticker.id", ondelete="CASCADE"))
    trade_date: Mapped[date]

    # --- Returns ----------------------------------------------------------
    daily_return: Mapped[Decimal | None]
    weekly_return: Mapped[Decimal | None]
    monthly_return: Mapped[Decimal | None]

    # --- Trend ------------------------------------------------------------
    sma_20: Mapped[Decimal | None]
    sma_50: Mapped[Decimal | None]
    sma_200: Mapped[Decimal | None]
    ema_12: Mapped[Decimal | None]
    ema_26: Mapped[Decimal | None]

    # --- Momentum ---------------------------------------------------------
    rsi_14: Mapped[Decimal | None]
    macd: Mapped[Decimal | None]
    macd_signal: Mapped[Decimal | None]
    macd_histogram: Mapped[Decimal | None]

    # --- Volatility -------------------------------------------------------
    bollinger_upper: Mapped[Decimal | None]
    bollinger_middle: Mapped[Decimal | None]
    bollinger_lower: Mapped[Decimal | None]
    atr_14: Mapped[Decimal | None]
    volatility_20: Mapped[Decimal | None]

    # --- Volume -----------------------------------------------------------
    volume_sma_20: Mapped[Decimal | None]
    #: Volume divided by its 20-session average. The primary volume-spike
    #: feature; 1.0 is an average day.
    volume_ratio: Mapped[Decimal | None]

    # --- Cross-sectional --------------------------------------------------
    #: Return relative to SMH, the semiconductor ETF benchmark. Separates a
    #: company-specific move from a sector-wide one, which is what makes the
    #: news-correlation engine's explanations meaningful.
    relative_strength_smh: Mapped[Decimal | None]

    ticker: Mapped[Ticker] = relationship(back_populates="indicators", lazy="raise_on_sql")

    def __repr__(self) -> str:
        """Return a debugging representation."""
        return f"<TechnicalIndicator ticker_id={self.ticker_id} date={self.trade_date}>"
