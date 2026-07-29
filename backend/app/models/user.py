"""User accounts and the objects they own: watchlists and portfolios.

Every user-owned table cascades on delete at the *database* level
(``ondelete="CASCADE"``), not just in the ORM. Deleting an account must remove
their data even when the delete comes from a migration or a psql session, which
is what makes an erasure request actually verifiable.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.db.base import Base, IntIdMixin, TimestampMixin, UuidIdMixin

if TYPE_CHECKING:
    from app.models.company import Ticker


class User(UuidIdMixin, TimestampMixin, Base):
    """An authenticated account."""

    # ``user`` is a reserved word in PostgreSQL: an unquoted
    # ``SELECT * FROM user`` returns the session role, not the table. Naming
    # it ``users`` keeps ad-hoc psql queries and dump tooling working.
    __tablename__ = "users"
    __table_args__ = ({"comment": "Platform user accounts."},)

    #: Stored lower-cased so the unique constraint is genuinely case-insensitive
    #: without a functional index; normalisation is enforced by the validator
    #: below so it cannot be bypassed by writing through the ORM directly.
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str | None] = mapped_column(String(150))

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    watchlists: Mapped[list[Watchlist]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="raise_on_sql",
        passive_deletes=True,
    )
    portfolios: Mapped[list[Portfolio]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="raise_on_sql",
        passive_deletes=True,
    )

    @validates("email")
    def _normalise_email(self, _key: str, value: str) -> str:
        """Lower-case and strip the address before it reaches the database."""
        return value.strip().lower()

    def __repr__(self) -> str:
        """Return a debugging representation without leaking the address."""
        return f"<User id={self.id}>"


class Watchlist(UuidIdMixin, TimestampMixin, Base):
    """A named collection of tickers a user follows."""

    __table_args__ = (
        UniqueConstraint("user_id", "name"),
        {"comment": "User-defined ticker collections."},
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str | None] = mapped_column(Text)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")

    user: Mapped[User] = relationship(back_populates="watchlists", lazy="raise_on_sql")
    items: Mapped[list[WatchlistItem]] = relationship(
        back_populates="watchlist",
        cascade="all, delete-orphan",
        lazy="raise_on_sql",
        passive_deletes=True,
        order_by="WatchlistItem.position",
    )

    def __repr__(self) -> str:
        """Return a debugging representation."""
        return f"<Watchlist id={self.id} name={self.name!r}>"


class WatchlistItem(IntIdMixin, TimestampMixin, Base):
    """Membership of one ticker in one watchlist."""

    __table_args__ = (
        UniqueConstraint("watchlist_id", "ticker_id"),
        {"comment": "Ordered ticker membership within a watchlist."},
    )

    watchlist_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("watchlist.id", ondelete="CASCADE"), index=True
    )
    ticker_id: Mapped[int] = mapped_column(ForeignKey("ticker.id", ondelete="CASCADE"), index=True)
    #: User-controlled display order.
    position: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    note: Mapped[str | None] = mapped_column(Text)

    watchlist: Mapped[Watchlist] = relationship(back_populates="items", lazy="raise_on_sql")
    ticker: Mapped[Ticker] = relationship(lazy="raise_on_sql")

    def __repr__(self) -> str:
        """Return a debugging representation."""
        return f"<WatchlistItem watchlist_id={self.watchlist_id} ticker_id={self.ticker_id}>"


class Portfolio(UuidIdMixin, TimestampMixin, Base):
    """A set of positions whose performance the platform tracks."""

    __table_args__ = (
        UniqueConstraint("user_id", "name"),
        {"comment": "User portfolios."},
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str | None] = mapped_column(Text)
    base_currency: Mapped[str] = mapped_column(String(3), default="USD", server_default="USD")

    user: Mapped[User] = relationship(back_populates="portfolios", lazy="raise_on_sql")
    positions: Mapped[list[PortfolioPosition]] = relationship(
        back_populates="portfolio",
        cascade="all, delete-orphan",
        lazy="raise_on_sql",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        """Return a debugging representation."""
        return f"<Portfolio id={self.id} name={self.name!r}>"


class PortfolioPosition(IntIdMixin, TimestampMixin, Base):
    """A holding of one ticker within one portfolio."""

    __table_args__ = (
        UniqueConstraint("portfolio_id", "ticker_id"),
        CheckConstraint("quantity >= 0", name="quantity_non_negative"),
        CheckConstraint("average_cost >= 0", name="average_cost_non_negative"),
        {"comment": "Holdings within a portfolio."},
    )

    portfolio_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("portfolio.id", ondelete="CASCADE"), index=True
    )
    ticker_id: Mapped[int] = mapped_column(ForeignKey("ticker.id", ondelete="CASCADE"), index=True)

    #: NUMERIC, never float: fractional shares and cost bases must stay exact.
    quantity: Mapped[Decimal]
    average_cost: Mapped[Decimal]
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    note: Mapped[str | None] = mapped_column(Text)

    portfolio: Mapped[Portfolio] = relationship(back_populates="positions", lazy="raise_on_sql")
    ticker: Mapped[Ticker] = relationship(lazy="raise_on_sql")

    @property
    def cost_basis(self) -> Decimal:
        """Total amount paid for the position."""
        return self.quantity * self.average_cost

    def __repr__(self) -> str:
        """Return a debugging representation."""
        return f"<PortfolioPosition portfolio_id={self.portfolio_id} ticker_id={self.ticker_id}>"
