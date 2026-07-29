"""Exchange calendar and the generated daily market briefing."""

from __future__ import annotations

from datetime import date, time
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IntIdMixin, TimestampMixin
from app.models.enums import MarketRegime, pg_enum


class MarketCalendar(IntIdMixin, TimestampMixin, Base):
    """Trading sessions per exchange.

    Without this, a holiday is indistinguishable from missing data: the
    ingestion pipeline would retry forever, and the anomaly detectors would read
    a four-day weekend as a volume collapse. Multiple exchanges are modelled
    because the tracked universe spans NASDAQ, TWSE and KRX.
    """

    __table_args__ = (
        UniqueConstraint("exchange", "session_date"),
        Index("ix_market_calendar_session_date", "session_date"),
        {"comment": "Per-exchange trading sessions and holidays."},
    )

    exchange: Mapped[str] = mapped_column(String(30))
    session_date: Mapped[date]
    is_trading_day: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    open_time: Mapped[time | None]
    close_time: Mapped[time | None]
    #: Set for non-trading days, e.g. "Thanksgiving" or "early close".
    note: Mapped[str | None] = mapped_column(String(120))

    def __repr__(self) -> str:
        """Return a debugging representation."""
        return f"<MarketCalendar {self.exchange} {self.session_date}>"


class DailyMarketSummary(IntIdMixin, TimestampMixin, Base):
    """The LLM-generated briefing for one session, plus the breadth stats it cites.

    The quantitative fields are stored alongside the prose deliberately: they are
    computed from the database, passed to the model as context, and kept so any
    claim in the narrative can be checked against the numbers that produced it.
    """

    __table_args__ = (
        UniqueConstraint("summary_date"),
        {"comment": "Generated daily market briefings with their supporting statistics."},
    )

    summary_date: Mapped[date]
    headline: Mapped[str] = mapped_column(String(300))
    body: Mapped[str] = mapped_column(Text)

    regime: Mapped[MarketRegime] = mapped_column(
        pg_enum(MarketRegime, "market_regime"),
        default=MarketRegime.MIXED,
    )

    advancers: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    decliners: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    unchanged: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    average_return: Mapped[Decimal | None]
    anomaly_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    #: Ranked movers and the articles cited, denormalised so rendering the
    #: briefing is a single read.
    top_movers: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default="{}")
    source_document_ids: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default="{}")

    #: Provenance of the generated text -- which model wrote it, and how much it
    #: cost. Needed to reproduce or re-cost a briefing after a model upgrade.
    generated_by_model: Mapped[str | None] = mapped_column(String(80))
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)

    def __repr__(self) -> str:
        """Return a debugging representation."""
        return f"<DailyMarketSummary {self.summary_date}>"
