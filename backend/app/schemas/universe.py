"""Schemas for the browsable exchange universe."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.listing import Listing


class ListingSummary(BaseModel):
    """One security in a search result or browse list."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str = Field(description="Exchange ticker, upper-case.")
    name: str = Field(description="Security name as the provider reports it.")
    exchange: str = Field(description="Market Identifier Code, e.g. XNAS.")
    currency: str = Field(description="ISO 4217 code.")
    security_type: str = Field(description="Common stock, ETF, ADR, and so on.")

    #: The distinction the whole platform turns on. A tracked symbol has price
    #: history, indicators, anomaly detection and embedded news behind it; an
    #: untracked one can be browsed and quoted on demand but has no analysis.
    #: Surfacing it here lets the UI say so plainly instead of rendering empty
    #: charts that look like a bug.
    is_tracked: bool = Field(
        default=False,
        description="Whether the platform runs analysis on this symbol.",
    )

    @classmethod
    def from_model(cls, listing: Listing, *, tracked: bool = False) -> ListingSummary:
        """Build a summary, stamping the tracked flag from the caller's set."""
        return cls(
            symbol=listing.symbol,
            name=listing.name,
            exchange=listing.exchange,
            currency=listing.currency,
            security_type=listing.security_type,
            is_tracked=tracked,
        )


class UniverseStats(BaseModel):
    """How much of the exchange is stored, and how fresh it is."""

    listings: int = Field(description="Active listings available to browse.")
    tracked: int = Field(description="Symbols the platform analyses.")
    last_synced_at: datetime | None = Field(
        default=None,
        description="When the universe was last reconciled with the provider.",
    )


class UniverseSyncResult(BaseModel):
    """Outcome of a sync run."""

    fetched: int = Field(description="Listings returned by the provider.")
    written: int = Field(description="Rows inserted or refreshed.")
    deactivated: int = Field(description="Stored symbols no longer listed.")
    succeeded: bool = Field(description="Whether the run completed.")
    error: str | None = Field(default=None, description="Why it did not, if it did not.")
