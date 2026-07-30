"""Watchlist and portfolio operations, with ownership enforced in one place.

Every method takes the acting user and resolves the resource *through* them.
That is deliberate: an authorisation check written as a separate `if` after the
load is a check somebody eventually forgets, and the failure mode is silent --
the endpoint works perfectly and serves another user's data.

Resources belonging to someone else raise :class:`NotFoundError`, not a
permission error. A 403 confirms the resource exists, which lets an attacker
enumerate other users' watchlist and portfolio ids by watching status codes. A
404 tells them nothing.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from app.core.exceptions import ConflictError, NotFoundError
from app.core.logging import get_logger
from app.models.company import Ticker
from app.models.user import Portfolio, PortfolioPosition, Watchlist
from app.repositories.company import TickerRepository
from app.repositories.price import DailyPriceRepository
from app.repositories.user import PortfolioRepository, WatchlistRepository

logger = get_logger(__name__)


@dataclass(frozen=True)
class WatchlistService:
    """Watchlist reads and writes, scoped to one user."""

    watchlists: WatchlistRepository
    tickers: TickerRepository

    async def list_for(self, user_id: uuid.UUID) -> Sequence[Watchlist]:
        """Return the user's watchlists."""
        return await self.watchlists.list_for_user(user_id)

    async def get_owned(self, user_id: uuid.UUID, watchlist_id: uuid.UUID) -> Watchlist:
        """Return one of the user's watchlists, with items loaded.

        Raises:
            NotFoundError: If it does not exist *or* belongs to someone else.
                The two are deliberately indistinguishable.
        """
        watchlist = await self.watchlists.get_with_items(watchlist_id)
        if watchlist is None or watchlist.user_id != user_id:
            msg = "Watchlist not found."
            raise NotFoundError(msg, details={"watchlist_id": str(watchlist_id)})
        return watchlist

    async def create(
        self,
        user_id: uuid.UUID,
        *,
        name: str,
        description: str | None,
        symbols: Sequence[str],
    ) -> Watchlist:
        """Create a watchlist, optionally pre-populated.

        Raises:
            ConflictError: If the user already has a list with this name.
            NotFoundError: If any supplied symbol is not tracked.
        """
        existing = await self.watchlists.list_for_user(user_id)
        if any(item.name == name for item in existing):
            msg = f"You already have a watchlist named {name!r}."
            raise ConflictError(msg, details={"field": "name"})

        watchlist = Watchlist(
            user_id=user_id,
            name=name,
            description=description,
            # The first list a user creates becomes their default, so there is
            # always exactly one obvious list for the dashboard to open with.
            is_default=not existing,
        )
        self.watchlists.add(watchlist)
        await self.watchlists.flush()

        for symbol in symbols:
            ticker = await self._resolve(symbol)
            await self.watchlists.add_ticker(watchlist.id, ticker.id)

        await self.watchlists.flush()
        return await self.get_owned(user_id, watchlist.id)

    async def delete(self, user_id: uuid.UUID, watchlist_id: uuid.UUID) -> None:
        """Delete one of the user's watchlists.

        Raises:
            NotFoundError: If it does not exist or belongs to someone else.
        """
        watchlist = await self.get_owned(user_id, watchlist_id)
        await self.watchlists.delete(watchlist)

    async def add_symbol(
        self,
        user_id: uuid.UUID,
        watchlist_id: uuid.UUID,
        *,
        symbol: str,
        note: str | None = None,
    ) -> Watchlist:
        """Add a ticker to one of the user's watchlists.

        Adding a symbol already on the list is a no-op rather than an error: the
        user's intent is "this should be on my list", and it already is.

        Raises:
            NotFoundError: If the watchlist is not the user's, or the symbol is
                not tracked.
        """
        watchlist = await self.get_owned(user_id, watchlist_id)
        ticker = await self._resolve(symbol)

        if any(item.ticker_id == ticker.id for item in watchlist.items):
            return watchlist

        await self.watchlists.add_ticker(watchlist.id, ticker.id, note=note)
        await self.watchlists.flush()
        return await self.get_owned(user_id, watchlist_id)

    async def remove_symbol(
        self, user_id: uuid.UUID, watchlist_id: uuid.UUID, *, symbol: str
    ) -> Watchlist:
        """Remove a ticker from one of the user's watchlists.

        Raises:
            NotFoundError: If the watchlist is not the user's, or the symbol is
                not tracked.
        """
        await self.get_owned(user_id, watchlist_id)
        ticker = await self._resolve(symbol)

        await self.watchlists.remove_ticker(watchlist_id, ticker.id)
        await self.watchlists.flush()
        return await self.get_owned(user_id, watchlist_id)

    async def _resolve(self, symbol: str) -> Ticker:
        """Resolve a symbol to a tracked listing.

        Raises:
            NotFoundError: If the symbol is not tracked. Reported distinctly from
                a missing watchlist, because it is the user's own input and they
                can correct it.
        """
        ticker = await self.tickers.get_by_symbol(symbol)
        if ticker is None:
            msg = f"Ticker {symbol.upper()!r} is not tracked."
            raise NotFoundError(msg, details={"symbol": symbol.upper()})
        return ticker


@dataclass(frozen=True)
class PortfolioService:
    """Portfolio reads and writes, scoped to one user."""

    portfolios: PortfolioRepository
    tickers: TickerRepository
    prices: DailyPriceRepository

    async def list_for(self, user_id: uuid.UUID) -> Sequence[Portfolio]:
        """Return the user's portfolios."""
        return await self.portfolios.list_for_user(user_id)

    async def get_owned(self, user_id: uuid.UUID, portfolio_id: uuid.UUID) -> Portfolio:
        """Return one of the user's portfolios, with positions loaded.

        Raises:
            NotFoundError: If it does not exist or belongs to someone else.
        """
        portfolio = await self.portfolios.get_with_positions(portfolio_id)
        if portfolio is None or portfolio.user_id != user_id:
            msg = "Portfolio not found."
            raise NotFoundError(msg, details={"portfolio_id": str(portfolio_id)})
        return portfolio

    async def create(
        self,
        user_id: uuid.UUID,
        *,
        name: str,
        description: str | None,
        base_currency: str,
    ) -> Portfolio:
        """Create an empty portfolio.

        Raises:
            ConflictError: If the user already has one with this name.
        """
        existing = await self.portfolios.list_for_user(user_id)
        if any(item.name == name for item in existing):
            msg = f"You already have a portfolio named {name!r}."
            raise ConflictError(msg, details={"field": "name"})

        portfolio = Portfolio(
            user_id=user_id,
            name=name,
            description=description,
            base_currency=base_currency.upper(),
        )
        self.portfolios.add(portfolio)
        await self.portfolios.flush()
        return await self.get_owned(user_id, portfolio.id)

    async def delete(self, user_id: uuid.UUID, portfolio_id: uuid.UUID) -> None:
        """Delete one of the user's portfolios and its positions."""
        portfolio = await self.get_owned(user_id, portfolio_id)
        await self.portfolios.delete(portfolio)

    async def upsert_position(
        self,
        user_id: uuid.UUID,
        portfolio_id: uuid.UUID,
        *,
        symbol: str,
        quantity: Decimal,
        average_cost: Decimal,
        opened_at: datetime | None = None,
        note: str | None = None,
    ) -> Portfolio:
        """Add a holding, or replace it if the ticker is already held.

        Upsert rather than insert because a portfolio holds one line per
        instrument. Two rows for the same ticker would make every total
        ambiguous, and the unique constraint would reject the second write
        anyway -- as an opaque IntegrityError rather than the intended update.

        Raises:
            NotFoundError: If the portfolio is not the user's, or the symbol is
                not tracked.
        """
        portfolio = await self.get_owned(user_id, portfolio_id)
        ticker = await self.tickers.get_by_symbol(symbol)
        if ticker is None:
            msg = f"Ticker {symbol.upper()!r} is not tracked."
            raise NotFoundError(msg, details={"symbol": symbol.upper()})

        existing = next((p for p in portfolio.positions if p.ticker_id == ticker.id), None)
        if existing is not None:
            existing.quantity = quantity
            existing.average_cost = average_cost
            existing.note = note
            if opened_at is not None:
                existing.opened_at = opened_at
        else:
            self.portfolios.session.add(
                PortfolioPosition(
                    portfolio_id=portfolio.id,
                    ticker_id=ticker.id,
                    quantity=quantity,
                    average_cost=average_cost,
                    opened_at=opened_at,
                    note=note,
                )
            )

        await self.portfolios.flush()
        return await self.get_owned(user_id, portfolio_id)

    async def remove_position(
        self, user_id: uuid.UUID, portfolio_id: uuid.UUID, *, symbol: str
    ) -> Portfolio:
        """Remove a holding.

        Raises:
            NotFoundError: If the portfolio is not the user's, the symbol is not
                tracked, or the position is not held.
        """
        portfolio = await self.get_owned(user_id, portfolio_id)
        ticker = await self.tickers.get_by_symbol(symbol)
        if ticker is None:
            msg = f"Ticker {symbol.upper()!r} is not tracked."
            raise NotFoundError(msg, details={"symbol": symbol.upper()})

        position = next((p for p in portfolio.positions if p.ticker_id == ticker.id), None)
        if position is None:
            msg = f"{ticker.symbol} is not held in this portfolio."
            raise NotFoundError(msg, details={"symbol": ticker.symbol})

        await self.portfolios.session.delete(position)
        await self.portfolios.flush()
        return await self.get_owned(user_id, portfolio_id)

    async def latest_prices(self, portfolio: Portfolio) -> dict[int, tuple[Decimal, date]]:
        """Return the latest close per held ticker, for valuation.

        Uses the completed-session close rather than the newest bar. Valuing a
        portfolio at a mid-session price would make the same holdings worth
        different amounts depending on when the page was loaded, with no
        indication that the number was provisional.
        """
        prices: dict[int, tuple[Decimal, date]] = {}
        for position in portfolio.positions:
            bars = await self.prices.get_recent(position.ticker_id, sessions=1, completed_only=True)
            if bars:
                prices[position.ticker_id] = (bars[-1].close, bars[-1].trade_date)
        return prices
