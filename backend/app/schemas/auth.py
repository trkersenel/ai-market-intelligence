"""Request and response schemas for authentication and user-owned resources."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, EmailStr, Field, computed_field

from app.models.user import Portfolio, PortfolioPosition, User, Watchlist, WatchlistItem

#: Minimum password length. NIST SP 800-63B recommends length over composition
#: rules: forced symbols and digits push users toward predictable substitutions
#: ("Password1!") while a longer passphrase is both stronger and easier to
#: remember. No composition requirement is imposed here for that reason.
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 128


class RegisterRequest(BaseModel):
    """Account creation payload."""

    email: EmailStr
    password: Annotated[str, Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)]
    full_name: Annotated[str | None, Field(max_length=150)] = None


class LoginRequest(BaseModel):
    """Credential payload."""

    email: EmailStr
    password: Annotated[str, Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)]


class RefreshRequest(BaseModel):
    """Token refresh payload."""

    refresh_token: Annotated[str, Field(min_length=1)]


class TokenResponse(BaseModel):
    """An issued token pair.

    Field names follow the OAuth2 bearer-token conventions so standard clients
    and the Swagger UI's authorise button work without translation.
    """

    access_token: str
    refresh_token: str
    token_type: str = "bearer"  # noqa: S105 - the OAuth2 scheme name
    expires_in: int


class UserResponse(BaseModel):
    """A user account, without the password hash."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    full_name: str | None
    is_active: bool
    created_at: datetime
    last_login_at: datetime | None


class AuthResponse(BaseModel):
    """The result of registering or logging in."""

    user: UserResponse
    tokens: TokenResponse


# --- Watchlists ------------------------------------------------------------


class WatchlistItemResponse(BaseModel):
    """One ticker's membership in a watchlist."""

    ticker_id: int
    symbol: str
    display_name: str
    position: int
    note: str | None

    @classmethod
    def from_model(cls, item: WatchlistItem) -> Self:
        """Build from an item whose ``ticker`` was eagerly loaded."""
        return cls(
            ticker_id=item.ticker_id,
            symbol=item.ticker.symbol,
            display_name=item.ticker.display_name,
            position=item.position,
            note=item.note,
        )


class WatchlistSummary(BaseModel):
    """A watchlist without its members."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    is_default: bool
    created_at: datetime


class WatchlistDetail(WatchlistSummary):
    """A watchlist with its members, in display order."""

    items: list[WatchlistItemResponse] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count(self) -> int:
        """Number of tickers on the list."""
        return len(self.items)

    @classmethod
    def from_model(cls, watchlist: Watchlist) -> Self:
        """Build from a watchlist whose items and tickers were eagerly loaded."""
        return cls(
            id=watchlist.id,
            name=watchlist.name,
            description=watchlist.description,
            is_default=watchlist.is_default,
            created_at=watchlist.created_at,
            items=[WatchlistItemResponse.from_model(item) for item in watchlist.items],
        )


class WatchlistCreateRequest(BaseModel):
    """Payload for creating a watchlist."""

    name: Annotated[str, Field(min_length=1, max_length=120)]
    description: Annotated[str | None, Field(max_length=1000)] = None
    symbols: Annotated[list[str], Field(max_length=100)] = []


class WatchlistItemRequest(BaseModel):
    """Payload for adding a ticker to a watchlist."""

    symbol: Annotated[str, Field(min_length=1, max_length=20)]
    note: Annotated[str | None, Field(max_length=1000)] = None


# --- Portfolios ------------------------------------------------------------


class PositionResponse(BaseModel):
    """One holding, valued at the latest stored close."""

    ticker_id: int
    symbol: str
    display_name: str
    quantity: Decimal
    average_cost: Decimal
    opened_at: datetime | None
    note: str | None

    #: Latest close, and the session it came from. Both nullable: a position can
    #: exist before the platform has ingested a price for it.
    last_close: Decimal | None = None
    last_close_date: date | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cost_basis(self) -> Decimal:
        """Total amount paid for the position."""
        return self.quantity * self.average_cost

    @computed_field  # type: ignore[prop-decorator]
    @property
    def market_value(self) -> Decimal | None:
        """Current value, or ``None`` when no price is stored."""
        return None if self.last_close is None else self.quantity * self.last_close

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unrealised_pnl(self) -> Decimal | None:
        """Gain or loss against cost basis, or ``None`` without a price."""
        value = self.market_value
        return None if value is None else value - self.cost_basis

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unrealised_pnl_percent(self) -> Decimal | None:
        """Return on cost, or ``None`` without a price or with a zero basis."""
        pnl = self.unrealised_pnl
        basis = self.cost_basis
        if pnl is None or basis == 0:
            return None
        return pnl / basis * 100

    @classmethod
    def from_model(
        cls,
        position: PortfolioPosition,
        *,
        last_close: Decimal | None = None,
        last_close_date: date | None = None,
    ) -> Self:
        """Build from a position whose ``ticker`` was eagerly loaded."""
        return cls(
            ticker_id=position.ticker_id,
            symbol=position.ticker.symbol,
            display_name=position.ticker.display_name,
            quantity=position.quantity,
            average_cost=position.average_cost,
            opened_at=position.opened_at,
            note=position.note,
            last_close=last_close,
            last_close_date=last_close_date,
        )


class PortfolioSummary(BaseModel):
    """A portfolio without its positions."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    base_currency: str
    created_at: datetime


class PortfolioDetail(PortfolioSummary):
    """A portfolio with its valued positions and the totals."""

    positions: list[PositionResponse] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_cost_basis(self) -> Decimal:
        """Sum of every position's cost."""
        return sum((position.cost_basis for position in self.positions), start=Decimal(0))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_market_value(self) -> Decimal | None:
        """Sum of position values, or ``None`` if any position lacks a price.

        Deliberately all-or-nothing. Silently omitting an unpriced holding would
        understate the total while still presenting it as the portfolio's value,
        which is worse than declining to compute it.
        """
        values = [position.market_value for position in self.positions]
        if not values or any(value is None for value in values):
            return None
        return sum((value for value in values if value is not None), start=Decimal(0))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_unrealised_pnl(self) -> Decimal | None:
        """Portfolio-level gain or loss, or ``None`` when unpriceable."""
        value = self.total_market_value
        return None if value is None else value - self.total_cost_basis

    @classmethod
    def from_model(
        cls,
        portfolio: Portfolio,
        prices: dict[int, tuple[Decimal, date]],
    ) -> Self:
        """Build from a portfolio whose positions and tickers were loaded.

        Args:
            portfolio: The loaded portfolio.
            prices: Latest ``(close, session)`` per ticker id. Missing entries
                leave that position unvalued rather than valued at zero.
        """
        return cls(
            id=portfolio.id,
            name=portfolio.name,
            description=portfolio.description,
            base_currency=portfolio.base_currency,
            created_at=portfolio.created_at,
            positions=[
                PositionResponse.from_model(
                    position,
                    last_close=prices.get(position.ticker_id, (None, None))[0],
                    last_close_date=prices.get(position.ticker_id, (None, None))[1],
                )
                for position in portfolio.positions
            ],
        )


class PortfolioCreateRequest(BaseModel):
    """Payload for creating a portfolio."""

    name: Annotated[str, Field(min_length=1, max_length=120)]
    description: Annotated[str | None, Field(max_length=1000)] = None
    base_currency: Annotated[str, Field(min_length=3, max_length=3)] = "USD"


class PositionRequest(BaseModel):
    """Payload for adding or updating a holding."""

    symbol: Annotated[str, Field(min_length=1, max_length=20)]
    quantity: Annotated[Decimal, Field(ge=0)]
    average_cost: Annotated[Decimal, Field(ge=0)]
    opened_at: datetime | None = None
    note: Annotated[str | None, Field(max_length=1000)] = None


def user_response(user: User) -> UserResponse:
    """Convert a user model to its API representation."""
    return UserResponse.model_validate(user)
