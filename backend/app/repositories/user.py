"""Repositories for users and the objects they own."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.models.company import Ticker
from app.models.user import Portfolio, PortfolioPosition, User, Watchlist, WatchlistItem
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User, uuid.UUID]):
    """Queries over accounts."""

    model = User

    async def get_by_email(self, email: str) -> User | None:
        """Return the account with this address, or ``None``.

        The address is normalised the same way the model normalises it on write,
        so lookup and storage cannot disagree about case or whitespace.
        """
        result = await self._session.execute(
            select(User).where(User.email == email.strip().lower())
        )
        return result.scalar_one_or_none()

    async def email_exists(self, email: str) -> bool:
        """Return whether an account already uses this address."""
        result = await self._session.execute(
            select(select(User).where(User.email == email.strip().lower()).exists())
        )
        return bool(result.scalar())


class WatchlistRepository(BaseRepository[Watchlist, uuid.UUID]):
    """Queries over user watchlists and their members."""

    model = Watchlist

    async def list_for_user(self, user_id: uuid.UUID) -> Sequence[Watchlist]:
        """Return a user's watchlists, default first."""
        result = await self._session.execute(
            select(Watchlist)
            .where(Watchlist.user_id == user_id)
            .order_by(Watchlist.is_default.desc(), Watchlist.name)
        )
        return result.scalars().all()

    async def get_with_items(self, watchlist_id: uuid.UUID) -> Watchlist | None:
        """Return a watchlist with its items and their tickers loaded.

        Two ``selectinload`` levels resolve the whole tree in three queries.
        Relationships are ``lazy="raise_on_sql"``, so forgetting this would
        raise rather than fan out into an N+1.
        """
        result = await self._session.execute(
            select(Watchlist)
            .where(Watchlist.id == watchlist_id)
            .options(selectinload(Watchlist.items).selectinload(WatchlistItem.ticker))
            # `populate_existing` is required, not an optimisation toggle.
            # Services write then re-read within one transaction to return the
            # new state. Without it the identity map hands back the instance
            # loaded by the earlier query, `selectinload` declines to refresh an
            # already-populated collection, and the response reports the state
            # from *before* the write -- a caller adding a ticker is told the
            # list still has the old count.
            .execution_options(populate_existing=True)
        )
        return result.scalar_one_or_none()

    async def get_default(self, user_id: uuid.UUID) -> Watchlist | None:
        """Return the user's default watchlist, if they have one."""
        result = await self._session.execute(
            select(Watchlist).where(Watchlist.user_id == user_id, Watchlist.is_default.is_(True))
        )
        return result.scalar_one_or_none()

    async def add_ticker(
        self, watchlist_id: uuid.UUID, ticker_id: int, *, note: str | None = None
    ) -> WatchlistItem:
        """Append a ticker to a watchlist.

        The new item is placed last. Adding a ticker that is already present
        violates the ``(watchlist_id, ticker_id)`` unique constraint; the
        service layer decides whether that is an error or a no-op.
        """
        next_position = await self._next_position(watchlist_id)
        item = WatchlistItem(
            watchlist_id=watchlist_id,
            ticker_id=ticker_id,
            position=next_position,
            note=note,
        )
        self._session.add(item)
        return item

    async def remove_ticker(self, watchlist_id: uuid.UUID, ticker_id: int) -> bool:
        """Remove a ticker from a watchlist.

        Returns:
            Whether an item was actually removed.
        """
        result = await self._session.execute(
            select(WatchlistItem).where(
                WatchlistItem.watchlist_id == watchlist_id,
                WatchlistItem.ticker_id == ticker_id,
            )
        )
        item = result.scalar_one_or_none()
        if item is None:
            return False
        await self._session.delete(item)
        return True

    async def list_tracked_symbols(self, user_id: uuid.UUID) -> Sequence[str]:
        """Return every distinct symbol across a user's watchlists.

        The input to watchlist-scoped digests and alerts.
        """
        result = await self._session.execute(
            select(Ticker.symbol)
            .join(WatchlistItem, WatchlistItem.ticker_id == Ticker.id)
            .join(Watchlist, Watchlist.id == WatchlistItem.watchlist_id)
            .where(Watchlist.user_id == user_id)
            .distinct()
            .order_by(Ticker.symbol)
        )
        return list(result.scalars().all())

    async def _next_position(self, watchlist_id: uuid.UUID) -> int:
        """Return the display position to assign to a newly added item."""
        result = await self._session.execute(
            select(WatchlistItem.position)
            .where(WatchlistItem.watchlist_id == watchlist_id)
            .order_by(WatchlistItem.position.desc())
            .limit(1)
        )
        highest = result.scalar_one_or_none()
        return 0 if highest is None else highest + 1


class PortfolioRepository(BaseRepository[Portfolio, uuid.UUID]):
    """Queries over portfolios and their holdings."""

    model = Portfolio

    async def list_for_user(self, user_id: uuid.UUID) -> Sequence[Portfolio]:
        """Return a user's portfolios by name."""
        result = await self._session.execute(
            select(Portfolio).where(Portfolio.user_id == user_id).order_by(Portfolio.name)
        )
        return result.scalars().all()

    async def get_with_positions(self, portfolio_id: uuid.UUID) -> Portfolio | None:
        """Return a portfolio with its positions and their tickers loaded."""
        result = await self._session.execute(
            select(Portfolio)
            .where(Portfolio.id == portfolio_id)
            .options(selectinload(Portfolio.positions).selectinload(PortfolioPosition.ticker))
            # As on watchlists: the write-then-read pattern needs the collection
            # refreshed, or a newly added position is missing from the response
            # that is supposed to confirm it.
            .execution_options(populate_existing=True)
        )
        return result.scalar_one_or_none()
