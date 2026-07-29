"""Detected market anomalies.

An anomaly row is the join point between the quantitative and the narrative
halves of the platform: it records *what* was unusual (score, magnitude, method)
and carries the references to the MongoDB news documents that explain *why*.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, IntIdMixin, TimestampMixin
from app.models.enums import AnomalyType, DetectionMethod, Direction, Severity, pg_enum

if TYPE_CHECKING:
    from app.models.company import Ticker


class Anomaly(IntIdMixin, TimestampMixin, Base):
    """A statistically unusual observation for one ticker on one session."""

    __table_args__ = (
        # Re-running detection over a historical window must refresh rows, not
        # accumulate duplicates -- the same guarantee prices have.
        # Its leading (ticker_id, trade_date) prefix also serves per-ticker
        # history reads, so no separate index is needed for them.
        UniqueConstraint("ticker_id", "trade_date", "anomaly_type", "method"),
        # Powers the "unusual movements this month" feed across all tickers,
        # filtered to the ones actually worth showing.
        Index("ix_anomaly_date_severity", "trade_date", "severity"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_is_a_ratio"),
        {"comment": "Anomalies detected by the statistical and ML detectors."},
    )

    ticker_id: Mapped[int] = mapped_column(ForeignKey("ticker.id", ondelete="CASCADE"))
    trade_date: Mapped[date]
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    anomaly_type: Mapped[AnomalyType] = mapped_column(pg_enum(AnomalyType, "anomaly_type"))
    method: Mapped[DetectionMethod] = mapped_column(pg_enum(DetectionMethod, "detection_method"))
    direction: Mapped[Direction] = mapped_column(
        pg_enum(Direction, "direction"),
        default=Direction.NEUTRAL,
    )
    severity: Mapped[Severity] = mapped_column(
        pg_enum(Severity, "severity"),
        default=Severity.LOW,
    )

    #: Raw detector output: a Z-score, or Isolation Forest's anomaly score.
    #: Not comparable across methods, which is why ``method`` is part of the key.
    score: Mapped[float] = mapped_column(Float)
    #: Normalised 0-1 confidence, comparable across methods and shown to users.
    confidence: Mapped[float] = mapped_column(Float)

    observed_value: Mapped[Decimal | None]
    expected_value: Mapped[Decimal | None]
    deviation: Mapped[Decimal | None]

    #: Human-readable explanation produced by the correlation engine.
    explanation: Mapped[str | None] = mapped_column(Text)

    #: MongoDB ``_id`` values of the news documents ranked as likely causes.
    #: A soft cross-store reference: PostgreSQL cannot enforce it, and a
    #: distributed transaction to make it enforceable would cost far more than
    #: the reconciliation job that repairs it.
    related_document_ids: Mapped[list[str]] = mapped_column(
        ARRAY(String(64)), default=list, server_default="{}"
    )

    #: Detector-specific context: feature vector, window size, contributing
    #: features. Schemaless because it differs per method and is diagnostic only.
    context: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default="{}")

    ticker: Mapped[Ticker] = relationship(back_populates="anomalies", lazy="raise_on_sql")

    def __repr__(self) -> str:
        """Return a debugging representation."""
        return (
            f"<Anomaly ticker_id={self.ticker_id} date={self.trade_date} "
            f"type={self.anomaly_type} severity={self.severity}>"
        )
