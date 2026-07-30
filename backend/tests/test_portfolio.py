"""Tests for watchlist and portfolio ownership and valuation.

The ownership tests are the important half. An authorisation bug does not throw:
the endpoint returns 200 with another user's data, and every functional test
still passes. So each read and write path is checked against a second account.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import date
from decimal import Decimal

import pytest

from app.core.exceptions import ConflictError, NotFoundError
from app.schemas.auth import PortfolioDetail, PositionResponse
from app.services.portfolio_service import PortfolioService, WatchlistService

OWNER = uuid.uuid4()
INTRUDER = uuid.uuid4()
TODAY = date(2026, 7, 30)


class FakeTicker:
    """Stands in for a listing."""

    def __init__(self, ticker_id: int, symbol: str) -> None:
        self.id = ticker_id
        self.symbol = symbol
        self.display_name = f"{symbol} Inc."


TICKERS = {"NVDA": FakeTicker(1, "NVDA"), "MU": FakeTicker(2, "MU")}


class FakeTickerRepository:
    """Resolves symbols from a fixed universe."""

    async def get_by_symbol(self, symbol: str) -> FakeTicker | None:
        return TICKERS.get(symbol.strip().upper())


class FakeItem:
    """Stands in for a watchlist membership row."""

    def __init__(self, ticker: FakeTicker, position: int = 0, note: str | None = None) -> None:
        self.ticker_id = ticker.id
        self.ticker = ticker
        self.position = position
        self.note = note


class FakeWatchlist:
    """Stands in for a watchlist."""

    def __init__(self, user_id: uuid.UUID, name: str = "List") -> None:
        self.id = uuid.uuid4()
        self.user_id = user_id
        self.name = name
        self.description: str | None = None
        self.is_default = False
        self.created_at = TODAY
        self.items: list[FakeItem] = []


class FakeWatchlistRepository:
    """In-memory watchlist store.

    Keeps its own mirrors rather than mutating the objects handed to it. The
    service constructs real ORM ``Watchlist`` instances, and appending to a
    SQLAlchemy instrumented collection outside a session raises -- so the fake
    stores plain equivalents and serves those back.
    """

    def __init__(self, watchlists: Sequence[FakeWatchlist] = ()) -> None:
        self.records: dict[uuid.UUID, FakeWatchlist] = {w.id: w for w in watchlists}
        self.deleted: list[FakeWatchlist] = []
        self.added_tickers: list[tuple[uuid.UUID, int]] = []
        self.removed_tickers: list[tuple[uuid.UUID, int]] = []

    @property
    def watchlists(self) -> list[FakeWatchlist]:
        """Every stored watchlist."""
        return list(self.records.values())

    async def list_for_user(self, user_id: uuid.UUID) -> list[FakeWatchlist]:
        return [w for w in self.records.values() if w.user_id == user_id]

    async def get_with_items(self, watchlist_id: uuid.UUID) -> FakeWatchlist | None:
        return self.records.get(watchlist_id)

    def add(self, watchlist: object) -> object:
        # The real model's UUID default is applied at INSERT, so with no database
        # the id is still None here; assigning one is what a flush would do.
        if getattr(watchlist, "id", None) is None:
            watchlist.id = uuid.uuid4()  # type: ignore[attr-defined]
        mirror = FakeWatchlist(
            watchlist.user_id,  # type: ignore[attr-defined]
            watchlist.name,  # type: ignore[attr-defined]
        )
        mirror.id = watchlist.id  # type: ignore[attr-defined]
        mirror.is_default = bool(getattr(watchlist, "is_default", False))
        mirror.description = getattr(watchlist, "description", None)
        self.records[mirror.id] = mirror
        return watchlist

    async def add_ticker(
        self, watchlist_id: uuid.UUID, ticker_id: int, *, note: str | None = None
    ) -> None:
        self.added_tickers.append((watchlist_id, ticker_id))
        watchlist = self.records.get(watchlist_id)
        if watchlist is not None:
            ticker = next(t for t in TICKERS.values() if t.id == ticker_id)
            watchlist.items.append(FakeItem(ticker, len(watchlist.items), note))

    async def remove_ticker(self, watchlist_id: uuid.UUID, ticker_id: int) -> bool:
        self.removed_tickers.append((watchlist_id, ticker_id))
        watchlist = self.records.get(watchlist_id)
        if watchlist is None:
            return False
        before = len(watchlist.items)
        watchlist.items = [i for i in watchlist.items if i.ticker_id != ticker_id]
        return len(watchlist.items) < before

    async def delete(self, watchlist: FakeWatchlist) -> None:
        self.deleted.append(watchlist)
        self.records.pop(watchlist.id, None)

    async def flush(self) -> None:
        return None


def _watchlist_service(
    watchlists: Sequence[FakeWatchlist] = (),
) -> tuple[WatchlistService, FakeWatchlistRepository]:
    repo = FakeWatchlistRepository(watchlists)
    service = WatchlistService(
        watchlists=repo,  # type: ignore[arg-type]
        tickers=FakeTickerRepository(),  # type: ignore[arg-type]
    )
    return service, repo


class TestWatchlistOwnership:
    """Every path must be scoped to the acting user."""

    async def test_reading_another_users_watchlist_is_not_found(self) -> None:
        """404 rather than 403.

        A 403 confirms the id exists, which turns the endpoint into an oracle for
        enumerating other accounts' watchlist ids.
        """
        owned = FakeWatchlist(OWNER)
        service, _ = _watchlist_service([owned])

        with pytest.raises(NotFoundError) as error:
            await service.get_owned(INTRUDER, owned.id)

        assert error.value.status_code == 404

    async def test_a_missing_and_a_foreign_watchlist_are_indistinguishable(self) -> None:
        owned = FakeWatchlist(OWNER)
        service, _ = _watchlist_service([owned])

        with pytest.raises(NotFoundError) as foreign:
            await service.get_owned(INTRUDER, owned.id)
        with pytest.raises(NotFoundError) as missing:
            await service.get_owned(INTRUDER, uuid.uuid4())

        assert foreign.value.message == missing.value.message

    async def test_deleting_another_users_watchlist_is_refused(self) -> None:
        owned = FakeWatchlist(OWNER)
        service, repo = _watchlist_service([owned])

        with pytest.raises(NotFoundError):
            await service.delete(INTRUDER, owned.id)

        assert repo.deleted == []
        assert owned in repo.watchlists

    async def test_adding_to_another_users_watchlist_is_refused(self) -> None:
        owned = FakeWatchlist(OWNER)
        service, repo = _watchlist_service([owned])

        with pytest.raises(NotFoundError):
            await service.add_symbol(INTRUDER, owned.id, symbol="NVDA")

        assert repo.added_tickers == []

    async def test_removing_from_another_users_watchlist_is_refused(self) -> None:
        owned = FakeWatchlist(OWNER)
        owned.items.append(FakeItem(TICKERS["NVDA"]))
        service, repo = _watchlist_service([owned])

        with pytest.raises(NotFoundError):
            await service.remove_symbol(INTRUDER, owned.id, symbol="NVDA")

        assert repo.removed_tickers == []

    async def test_listing_returns_only_your_own(self) -> None:
        service, _ = _watchlist_service([FakeWatchlist(OWNER), FakeWatchlist(INTRUDER)])

        found = await service.list_for(OWNER)

        assert len(found) == 1
        assert found[0].user_id == OWNER


class TestWatchlistOperations:
    """Behaviour within one user's own data."""

    async def test_the_first_watchlist_becomes_the_default(self) -> None:
        """The dashboard needs exactly one obvious list to open with."""
        service, _ = _watchlist_service()

        created = await service.create(OWNER, name="First", description=None, symbols=[])

        assert created.is_default is True

    async def test_a_later_watchlist_does_not(self) -> None:
        service, _ = _watchlist_service([FakeWatchlist(OWNER, "Existing")])

        created = await service.create(OWNER, name="Second", description=None, symbols=[])

        assert created.is_default is False

    async def test_a_duplicate_name_is_rejected(self) -> None:
        service, _ = _watchlist_service([FakeWatchlist(OWNER, "Semis")])

        with pytest.raises(ConflictError):
            await service.create(OWNER, name="Semis", description=None, symbols=[])

    async def test_the_same_name_is_allowed_for_a_different_user(self) -> None:
        """Names are unique per account, not globally."""
        service, _ = _watchlist_service([FakeWatchlist(OWNER, "Semis")])

        created = await service.create(INTRUDER, name="Semis", description=None, symbols=[])

        assert created.name == "Semis"

    async def test_creation_can_prepopulate_symbols(self) -> None:
        service, _ = _watchlist_service()

        created = await service.create(
            OWNER, name="Semis", description=None, symbols=["NVDA", "MU"]
        )

        assert {item.ticker.symbol for item in created.items} == {"NVDA", "MU"}

    async def test_an_unknown_symbol_is_rejected(self) -> None:
        service, _ = _watchlist_service()

        with pytest.raises(NotFoundError, match="not tracked"):
            await service.create(OWNER, name="Bad", description=None, symbols=["ZZZZ"])

    async def test_adding_a_symbol_already_present_is_a_no_op(self) -> None:
        """The user's intent is "this should be on my list", and it already is."""
        owned = FakeWatchlist(OWNER)
        owned.items.append(FakeItem(TICKERS["NVDA"]))
        service, repo = _watchlist_service([owned])

        result = await service.add_symbol(OWNER, owned.id, symbol="NVDA")

        assert len(result.items) == 1
        assert repo.added_tickers == []

    async def test_symbols_are_matched_case_insensitively(self) -> None:
        owned = FakeWatchlist(OWNER)
        service, _ = _watchlist_service([owned])

        result = await service.add_symbol(OWNER, owned.id, symbol="nvda")

        assert result.items[0].ticker.symbol == "NVDA"


# --- Portfolios ------------------------------------------------------------


class FakePosition:
    """Stands in for a holding."""

    def __init__(
        self, ticker: FakeTicker, quantity: str, average_cost: str, note: str | None = None
    ) -> None:
        self.ticker_id = ticker.id
        self.ticker = ticker
        self.quantity = Decimal(quantity)
        self.average_cost = Decimal(average_cost)
        self.opened_at = None
        self.note = note


class FakePortfolio:
    """Stands in for a portfolio."""

    def __init__(self, user_id: uuid.UUID, name: str = "Main") -> None:
        self.id = uuid.uuid4()
        self.user_id = user_id
        self.name = name
        self.description: str | None = None
        self.base_currency = "USD"
        self.created_at = TODAY
        self.positions: list[FakePosition] = []


class FakeSession:
    """Records ORM-level adds and deletes."""

    def __init__(self) -> None:
        self.added: list[object] = []
        self.deleted: list[object] = []

    def add(self, entity: object) -> None:
        self.added.append(entity)

    async def delete(self, entity: object) -> None:
        self.deleted.append(entity)


class FakePortfolioRepository:
    """In-memory portfolio store."""

    def __init__(self, portfolios: Sequence[FakePortfolio] = ()) -> None:
        self.portfolios = list(portfolios)
        self.session = FakeSession()
        self.deleted: list[FakePortfolio] = []

    async def list_for_user(self, user_id: uuid.UUID) -> list[FakePortfolio]:
        return [p for p in self.portfolios if p.user_id == user_id]

    async def get_with_positions(self, portfolio_id: uuid.UUID) -> FakePortfolio | None:
        return next((p for p in self.portfolios if p.id == portfolio_id), None)

    def add(self, portfolio: FakePortfolio) -> FakePortfolio:
        self.portfolios.append(portfolio)
        return portfolio

    async def delete(self, portfolio: FakePortfolio) -> None:
        self.deleted.append(portfolio)
        self.portfolios.remove(portfolio)

    async def flush(self) -> None:
        return None


class FakeBar:
    """Stands in for a price bar."""

    def __init__(self, close: str, trade_date: date = TODAY) -> None:
        self.close = Decimal(close)
        self.trade_date = trade_date


class FakePriceRepository:
    """Serves the latest close per ticker, and records the flags requested."""

    def __init__(self, closes: dict[int, str] | None = None) -> None:
        self.closes = closes or {}
        self.completed_only_requested: list[bool] = []

    async def get_recent(
        self, ticker_id: int, *, sessions: int | None = None, completed_only: bool = False
    ) -> list[FakeBar]:
        self.completed_only_requested.append(completed_only)
        close = self.closes.get(ticker_id)
        return [FakeBar(close)] if close else []


def _portfolio_service(
    portfolios: Sequence[FakePortfolio] = (),
    closes: dict[int, str] | None = None,
) -> tuple[PortfolioService, FakePortfolioRepository, FakePriceRepository]:
    repo = FakePortfolioRepository(portfolios)
    prices = FakePriceRepository(closes)
    service = PortfolioService(
        portfolios=repo,  # type: ignore[arg-type]
        tickers=FakeTickerRepository(),  # type: ignore[arg-type]
        prices=prices,  # type: ignore[arg-type]
    )
    return service, repo, prices


class TestPortfolioOwnership:
    """Same scoping requirement as watchlists."""

    async def test_reading_another_users_portfolio_is_not_found(self) -> None:
        owned = FakePortfolio(OWNER)
        service, _, _ = _portfolio_service([owned])

        with pytest.raises(NotFoundError) as error:
            await service.get_owned(INTRUDER, owned.id)

        assert error.value.status_code == 404

    async def test_deleting_another_users_portfolio_is_refused(self) -> None:
        owned = FakePortfolio(OWNER)
        service, repo, _ = _portfolio_service([owned])

        with pytest.raises(NotFoundError):
            await service.delete(INTRUDER, owned.id)

        assert repo.deleted == []

    async def test_writing_a_position_to_another_users_portfolio_is_refused(self) -> None:
        owned = FakePortfolio(OWNER)
        service, repo, _ = _portfolio_service([owned])

        with pytest.raises(NotFoundError):
            await service.upsert_position(
                INTRUDER,
                owned.id,
                symbol="NVDA",
                quantity=Decimal("10"),
                average_cost=Decimal("100"),
            )

        assert repo.session.added == []

    async def test_removing_a_position_from_another_users_portfolio_is_refused(self) -> None:
        owned = FakePortfolio(OWNER)
        owned.positions.append(FakePosition(TICKERS["NVDA"], "10", "100"))
        service, repo, _ = _portfolio_service([owned])

        with pytest.raises(NotFoundError):
            await service.remove_position(INTRUDER, owned.id, symbol="NVDA")

        assert repo.session.deleted == []


class TestPositions:
    """Holdings within one's own portfolio."""

    async def test_a_position_is_added(self) -> None:
        owned = FakePortfolio(OWNER)
        service, repo, _ = _portfolio_service([owned])

        await service.upsert_position(
            OWNER,
            owned.id,
            symbol="NVDA",
            quantity=Decimal("10"),
            average_cost=Decimal("120.50"),
        )

        assert len(repo.session.added) == 1

    async def test_re_adding_a_symbol_updates_rather_than_duplicating(self) -> None:
        """One line per instrument, or every total becomes ambiguous."""
        owned = FakePortfolio(OWNER)
        owned.positions.append(FakePosition(TICKERS["NVDA"], "10", "100"))
        service, repo, _ = _portfolio_service([owned])

        await service.upsert_position(
            OWNER,
            owned.id,
            symbol="NVDA",
            quantity=Decimal("25"),
            average_cost=Decimal("130"),
        )

        assert repo.session.added == []
        assert owned.positions[0].quantity == Decimal("25")
        assert owned.positions[0].average_cost == Decimal("130")

    async def test_removing_a_position_not_held_is_reported(self) -> None:
        owned = FakePortfolio(OWNER)
        service, _, _ = _portfolio_service([owned])

        with pytest.raises(NotFoundError, match="not held"):
            await service.remove_position(OWNER, owned.id, symbol="NVDA")

    async def test_an_unknown_symbol_is_rejected(self) -> None:
        owned = FakePortfolio(OWNER)
        service, _, _ = _portfolio_service([owned])

        with pytest.raises(NotFoundError, match="not tracked"):
            await service.upsert_position(
                OWNER,
                owned.id,
                symbol="ZZZZ",
                quantity=Decimal("1"),
                average_cost=Decimal("1"),
            )


class TestValuation:
    """Prices, totals and what happens when a price is missing."""

    async def test_valuation_uses_completed_sessions_only(self) -> None:
        """Valuation must use settled closes.

        A mid-session price would make the same holdings worth different amounts
        depending on when the page was loaded, with no indication that the figure
        was provisional.
        """
        owned = FakePortfolio(OWNER)
        owned.positions.append(FakePosition(TICKERS["NVDA"], "10", "100"))
        service, _, prices = _portfolio_service([owned], {1: "150"})

        await service.latest_prices(owned)  # type: ignore[arg-type]

        assert all(prices.completed_only_requested)

    def test_cost_basis_and_pnl_are_computed_from_decimals(self) -> None:
        position = PositionResponse(
            ticker_id=1,
            symbol="NVDA",
            display_name="NVDA Inc.",
            quantity=Decimal("10"),
            average_cost=Decimal("100"),
            opened_at=None,
            note=None,
            last_close=Decimal("150"),
            last_close_date=TODAY,
        )

        assert position.cost_basis == Decimal("1000")
        assert position.market_value == Decimal("1500")
        assert position.unrealised_pnl == Decimal("500")
        assert position.unrealised_pnl_percent == Decimal("50")

    def test_an_unpriced_position_reports_no_value_rather_than_zero(self) -> None:
        position = PositionResponse(
            ticker_id=1,
            symbol="NVDA",
            display_name="NVDA Inc.",
            quantity=Decimal("10"),
            average_cost=Decimal("100"),
            opened_at=None,
            note=None,
        )

        assert position.market_value is None
        assert position.unrealised_pnl is None
        assert position.cost_basis == Decimal("1000")

    async def test_a_portfolio_total_is_all_or_nothing(self) -> None:
        """A total is reported only when every holding can be priced.

        Silently omitting an unpriced holding would understate the total while
        still presenting it as the portfolio's value.
        """
        owned = FakePortfolio(OWNER)
        owned.positions.append(FakePosition(TICKERS["NVDA"], "10", "100"))
        owned.positions.append(FakePosition(TICKERS["MU"], "20", "50"))
        service, _, _ = _portfolio_service([owned], {1: "150"})  # MU unpriced

        prices = await service.latest_prices(owned)  # type: ignore[arg-type]
        detail = PortfolioDetail.from_model(owned, prices)  # type: ignore[arg-type]

        assert detail.total_cost_basis == Decimal("2000")
        assert detail.total_market_value is None
        assert detail.total_unrealised_pnl is None

    async def test_a_fully_priced_portfolio_totals_correctly(self) -> None:
        owned = FakePortfolio(OWNER)
        owned.positions.append(FakePosition(TICKERS["NVDA"], "10", "100"))
        owned.positions.append(FakePosition(TICKERS["MU"], "20", "50"))
        service, _, _ = _portfolio_service([owned], {1: "150", 2: "60"})

        prices = await service.latest_prices(owned)  # type: ignore[arg-type]
        detail = PortfolioDetail.from_model(owned, prices)  # type: ignore[arg-type]

        assert detail.total_cost_basis == Decimal("2000")
        assert detail.total_market_value == Decimal("2700")
        assert detail.total_unrealised_pnl == Decimal("700")

    def test_a_zero_cost_basis_does_not_divide_by_zero(self) -> None:
        position = PositionResponse(
            ticker_id=1,
            symbol="NVDA",
            display_name="NVDA Inc.",
            quantity=Decimal("10"),
            average_cost=Decimal("0"),
            opened_at=None,
            note=None,
            last_close=Decimal("150"),
            last_close_date=TODAY,
        )

        assert position.unrealised_pnl_percent is None
