"""The searchable universe: every listing on the exchange, browse-only.

Kept separate from :class:`~app.models.company.Ticker`, which is the *tracked*
set. The distinction is the platform's central trade-off made concrete:

**Listing** is breadth. Thousands of rows synced daily from the provider's
symbol file, carrying only what a search result needs -- symbol, name, exchange,
type. Nothing here costs an API call to display, so all of NASDAQ can be
browsed for free.

**Ticker** is depth. A few dozen rows the platform actually spends quota on:
price history, indicators, anomaly detection, embedded news. Each carries
ingestion watermarks the scheduler reads on every run.

Merging them would break both. The scheduler iterates tickers, so five thousand
browse-only rows would turn a two-minute ingestion run into an all-day one; and
``Ticker`` requires a parent company for equities, which would mean fabricating
five thousand empty ``Company`` rows to hold nothing.

The two are joined on ``symbol``. A listing with a matching ticker is tracked;
one without is browsable, and tracking it is a deliberate act.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IntIdMixin, TimestampMixin


class Listing(IntIdMixin, TimestampMixin, Base):
    """One security in the browsable universe."""

    __table_args__ = (
        Index("ix_listing_exchange", "exchange"),
        # No composite (symbol, is_active) index either: ``symbol`` is unique,
        # so a composite leading with it can never beat the unique index alone.
        #
        # No index on ``name``. Search matches it as a substring, which a b-tree
        # cannot serve anyway, and the whole table is a few thousand rows -- a
        # sequential scan over it is sub-millisecond. An index here would be
        # decoration: it would look like tuning while changing nothing. If the
        # universe ever spans every US exchange, the answer is a pg_trgm GIN
        # index, not a b-tree.
        {"comment": "Browsable exchange universe; tracked symbols also exist in ticker."},
    )

    symbol: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))

    #: MIC, not a colloquial name: "XNAS" rather than "NASDAQ". Providers
    #: disagree about the latter and agree about the former.
    exchange: Mapped[str] = mapped_column(String(20), default="", server_default="")
    currency: Mapped[str] = mapped_column(String(3), default="USD", server_default="USD")

    #: Common stock, ETF, ADR. A free-text column rather than an enum: providers
    #: use different vocabularies, and adapters pass anything unrecognised
    #: through verbatim rather than guessing. An enum would reject a new value
    #: at write time and lose the row.
    security_type: Mapped[str] = mapped_column(
        String(40), default="common", server_default="common"
    )

    #: FIGI where the provider supplies one. It survives ticker changes -- so a
    #: renamed symbol can still be recognised as the same security -- and unlike
    #: CUSIP it is not licensed.
    figi: Mapped[str | None] = mapped_column(String(20), index=True)

    #: Set false when a symbol disappears from the provider's file rather than
    #: deleting the row. A delisted symbol still appears in stored news and in
    #: anyone's watchlist, and a dangling reference is worse than a flag.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")

    #: Which provider supplied the row, and when. Both are diagnostic: after a
    #: provider swap the mixture is visible, and a stale timestamp explains a
    #: missing recent IPO without anyone having to guess.
    source: Mapped[str] = mapped_column(String(30), default="", server_default="")
    synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    def __repr__(self) -> str:
        """Return a debugging representation."""
        return f"<Listing symbol={self.symbol!r} exchange={self.exchange!r}>"
