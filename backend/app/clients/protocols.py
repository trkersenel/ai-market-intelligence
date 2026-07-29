"""Provider protocols and the normalised shapes they return.

Ingestion services depend on these ``Protocol`` classes, never on a concrete
client. That is what makes the pipeline testable without a network: a fake
provider satisfies the protocol structurally, with no base class to inherit and
no registration step.

The DTOs are Pydantic models rather than dataclasses because they carry
*untrusted* data. A vendor returning a null volume, a negative price or a naive
timestamp should fail loudly at the boundary, not three stages later inside a
feature calculation.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.enums import DataSource


class PriceBar(BaseModel):
    """One normalised OHLCV bar, independent of the provider that supplied it."""

    model_config = ConfigDict(frozen=True)

    trade_date: date
    open: Annotated[Decimal, Field(ge=0)]
    high: Annotated[Decimal, Field(ge=0)]
    low: Annotated[Decimal, Field(ge=0)]
    close: Annotated[Decimal, Field(ge=0)]
    adjusted_close: Annotated[Decimal, Field(ge=0)]
    volume: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _check_range_is_coherent(self) -> PriceBar:
        """Reject bars whose high is below its low.

        The database enforces this too, but failing here names the provider and
        the session in the error instead of surfacing as an opaque IntegrityError
        after a thousand-row batch was assembled.
        """
        if self.high < self.low:
            msg = f"bar for {self.trade_date} has high {self.high} below low {self.low}"
            raise ValueError(msg)
        return self


class RawArticle(BaseModel):
    """A news item as retrieved, before company tagging or sentiment scoring."""

    model_config = ConfigDict(frozen=True)

    url: Annotated[str, Field(min_length=1)]
    title: Annotated[str, Field(min_length=1)]
    summary: str | None = None
    content: str | None = None
    published_at: datetime
    source: DataSource
    source_name: str | None = None
    author: str | None = None
    language: str = "en"

    @field_validator("published_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        """Reject naive timestamps.

        Feeds report in local time, UTC and occasionally with no offset at all.
        Comparing a naive timestamp against an anomaly's session date silently
        shifts articles across day boundaries, which is precisely the kind of
        error that makes a correlation look plausible and be wrong.
        """
        if value.tzinfo is None:
            msg = "published_at must be timezone-aware"
            raise ValueError(msg)
        return value

    @property
    def searchable_text(self) -> str:
        """Title, summary and body joined, for tagging and keyword extraction."""
        parts = [self.title, self.summary or "", self.content or ""]
        return "\n".join(part for part in parts if part)


@runtime_checkable
class PriceProvider(Protocol):
    """Supplies historical daily bars for a symbol."""

    @property
    def source(self) -> DataSource:
        """Provenance recorded on every row this provider produces."""
        ...

    async def fetch_daily_bars(self, symbol: str, *, start: date, end: date) -> list[PriceBar]:
        """Return bars for ``symbol`` within an inclusive date window.

        Implementations return an empty list for a symbol with no data in the
        window -- a quiet ticker is not an error.

        Raises:
            ExternalServiceError: If the provider is unreachable or returns an
                unusable response.
        """
        ...


@runtime_checkable
class NewsProvider(Protocol):
    """Supplies news articles."""

    @property
    def source(self) -> DataSource:
        """Provenance recorded on every article this provider produces."""
        ...

    @property
    def name(self) -> str:
        """Human-readable provider name, used in logs and ingestion reports."""
        ...

    async def fetch_articles(
        self, *, since: datetime, query: str | None = None, limit: int = 100
    ) -> list[RawArticle]:
        """Return articles published at or after ``since``.

        Args:
            since: Lower bound on publication time.
            query: Optional keyword filter, for providers that support one.
            limit: Maximum articles to return.

        Raises:
            ExternalServiceError: If the provider is unreachable or returns an
                unusable response.
        """
        ...
