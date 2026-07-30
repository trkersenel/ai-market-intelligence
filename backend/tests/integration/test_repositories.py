"""Repository behaviour verified against a real PostgreSQL instance.

Only the things a fake cannot prove are tested here: that ON CONFLICT clauses
target the constraints they are meant to, that CHECK constraints reject bad
rows, that ON DELETE CASCADE reaches grandchildren, and that array containment
uses the operators the GIN index serves.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.seed import seed_reference_data
from app.models.anomaly import Anomaly
from app.models.company import Company, Ticker
from app.models.enums import (
    AnomalyType,
    AssetType,
    DetectionMethod,
    EcosystemTag,
    Severity,
)
from app.models.user import Portfolio, PortfolioPosition, User, Watchlist
from app.repositories import (
    AnomalyRepository,
    CompanyRepository,
    DailyPriceRepository,
    PortfolioRepository,
    TechnicalIndicatorRepository,
    TickerRepository,
    UserRepository,
    WatchlistRepository,
)

pytestmark = pytest.mark.integration


async def _make_ticker(session: AsyncSession, symbol: str = "TEST") -> Ticker:
    """Insert a minimal company and listing, returning the listing."""
    company = Company(slug=f"{symbol.lower()}-co", name=f"{symbol} Co", tags=["gpu"])
    session.add(company)
    await session.flush()
    ticker = Ticker(
        company_id=company.id,
        symbol=symbol,
        display_name=f"{symbol} Inc.",
        exchange="NASDAQ",
    )
    session.add(ticker)
    await session.flush()
    return ticker


def _bar(ticker_id: int, day: date, *, close: str, volume: int = 1_000) -> dict[str, Any]:
    """Build one OHLCV row mapping."""
    price = Decimal(close)
    return {
        "ticker_id": ticker_id,
        "trade_date": day,
        "open": price,
        "high": price + Decimal("1"),
        "low": price - Decimal("1"),
        "close": price,
        "adjusted_close": price,
        "volume": volume,
    }


class TestDailyPriceUpsert:
    """The idempotency guarantee the whole ingestion pipeline rests on."""

    async def test_reingesting_the_same_window_updates_rather_than_duplicates(
        self, session: AsyncSession
    ) -> None:
        ticker = await _make_ticker(session)
        repository = DailyPriceRepository(session)
        window = [_bar(ticker.id, date(2026, 1, day), close="100") for day in (5, 6, 7)]

        await repository.bulk_upsert(window)
        await session.flush()

        # A vendor correction arrives for one session; the job re-runs the window.
        window[1] = _bar(ticker.id, date(2026, 1, 6), close="123.456789", volume=9_999)
        await repository.bulk_upsert(window)
        await session.flush()

        stored = await repository.get_range(
            ticker.id, start=date(2026, 1, 1), end=date(2026, 1, 31)
        )
        assert len(stored) == 3, "re-ingestion must not duplicate rows"
        corrected = next(bar for bar in stored if bar.trade_date == date(2026, 1, 6))
        assert corrected.close == Decimal("123.456789")
        assert corrected.volume == 9_999

    async def test_prices_keep_full_decimal_precision(self, session: AsyncSession) -> None:
        """A float column would silently lose these digits."""
        ticker = await _make_ticker(session)
        repository = DailyPriceRepository(session)
        exact = Decimal("1234.567891")

        await repository.bulk_upsert(
            [{**_bar(ticker.id, date(2026, 2, 2), close="1"), "close": exact}]
        )
        await session.flush()

        stored = await repository.get_latest(ticker.id)
        assert stored is not None
        assert stored.close == exact

    async def test_empty_batch_is_a_no_op(self, session: AsyncSession) -> None:
        assert await DailyPriceRepository(session).bulk_upsert([]) == 0

    async def test_check_constraint_rejects_high_below_low(self, session: AsyncSession) -> None:
        ticker = await _make_ticker(session)
        bad = _bar(ticker.id, date(2026, 3, 3), close="10")
        bad["high"] = Decimal("1")
        bad["low"] = Decimal("100")

        with pytest.raises(IntegrityError):
            await DailyPriceRepository(session).bulk_upsert([bad])
            await session.flush()

    async def test_recent_returns_oldest_first(self, session: AsyncSession) -> None:
        ticker = await _make_ticker(session)
        repository = DailyPriceRepository(session)
        await repository.bulk_upsert(
            [_bar(ticker.id, date(2026, 4, day), close=str(day)) for day in range(1, 11)]
        )
        await session.flush()

        recent = await repository.get_recent(ticker.id, sessions=3)

        assert [bar.trade_date.day for bar in recent] == [8, 9, 10]

    async def test_date_bounds(self, session: AsyncSession) -> None:
        ticker = await _make_ticker(session)
        repository = DailyPriceRepository(session)
        assert await repository.get_date_bounds(ticker.id) is None

        await repository.bulk_upsert(
            [_bar(ticker.id, date(2026, 5, day), close="1") for day in (4, 18, 11)]
        )
        await session.flush()

        assert await repository.get_date_bounds(ticker.id) == (
            date(2026, 5, 4),
            date(2026, 5, 18),
        )


class TestIndicatorUpsert:
    """Recomputing features must overwrite, including back to NULL."""

    async def test_recomputation_overwrites_previous_values(self, session: AsyncSession) -> None:
        ticker = await _make_ticker(session)
        repository = TechnicalIndicatorRepository(session)
        row = {
            "ticker_id": ticker.id,
            "trade_date": date(2026, 6, 1),
            "rsi_14": Decimal("70.5"),
            "daily_return": Decimal("0.031"),
        }

        await repository.bulk_upsert([row])
        await session.flush()

        # The recomputation emits only rsi_14 this time -- e.g. because the
        # return could not be derived from the available window.
        await repository.bulk_upsert(
            [
                {
                    "ticker_id": ticker.id,
                    "trade_date": date(2026, 6, 1),
                    "rsi_14": Decimal("25.25"),
                }
            ]
        )
        await session.flush()

        stored = await repository.get_latest(ticker.id)
        assert stored is not None
        assert stored.rsi_14 == Decimal("25.250000")
        # Omitted from the second batch, so the upsert resets it to NULL. A
        # stale feature must not outlive the data it was derived from.
        assert stored.daily_return is None


class TestAnomalyRepository:
    """Detector output, its idempotency key, and the anomaly feed."""

    def _anomaly(
        self,
        ticker_id: int,
        *,
        day: date,
        method: DetectionMethod,
        severity: Severity = Severity.HIGH,
        confidence: float = 0.9,
    ) -> dict[str, Any]:
        return {
            "ticker_id": ticker_id,
            "trade_date": day,
            "anomaly_type": AnomalyType.RETURN,
            "method": method,
            "severity": severity,
            "score": 4.2,
            "confidence": confidence,
        }

    async def test_two_detectors_may_flag_the_same_day(self, session: AsyncSession) -> None:
        """The key includes ``method``, so detectors never overwrite each other."""
        ticker = await _make_ticker(session)
        repository = AnomalyRepository(session)
        day = date(2026, 7, 1)

        await repository.bulk_upsert(
            [
                self._anomaly(ticker.id, day=day, method=DetectionMethod.Z_SCORE),
                self._anomaly(ticker.id, day=day, method=DetectionMethod.ISOLATION_FOREST),
            ]
        )
        await session.flush()

        count = await session.scalar(select(func.count()).select_from(Anomaly))
        assert count == 2

    async def test_rerunning_a_detector_refreshes_its_own_row(self, session: AsyncSession) -> None:
        ticker = await _make_ticker(session)
        repository = AnomalyRepository(session)
        day = date(2026, 7, 2)
        row = self._anomaly(ticker.id, day=day, method=DetectionMethod.Z_SCORE)

        await repository.bulk_upsert([row])
        await repository.bulk_upsert([{**row, "confidence": 0.42, "severity": Severity.LOW}])
        await session.flush()

        stored = await repository.list_for_ticker(ticker.id, start=day, end=day)
        assert len(stored) == 1
        assert stored[0].confidence == pytest.approx(0.42)
        assert stored[0].severity is Severity.LOW

    async def test_confidence_outside_zero_to_one_is_rejected(self, session: AsyncSession) -> None:
        ticker = await _make_ticker(session)
        with pytest.raises(IntegrityError):
            await AnomalyRepository(session).bulk_upsert(
                [
                    self._anomaly(
                        ticker.id,
                        day=date(2026, 7, 3),
                        method=DetectionMethod.Z_SCORE,
                        confidence=1.5,
                    )
                ]
            )
            await session.flush()

    async def test_feed_filters_by_minimum_severity_and_joins_the_symbol(
        self, session: AsyncSession
    ) -> None:
        ticker = await _make_ticker(session, symbol="FEED")
        repository = AnomalyRepository(session)
        await repository.bulk_upsert(
            [
                self._anomaly(
                    ticker.id,
                    day=date(2026, 8, 1),
                    method=DetectionMethod.Z_SCORE,
                    severity=Severity.LOW,
                ),
                self._anomaly(
                    ticker.id,
                    day=date(2026, 8, 2),
                    method=DetectionMethod.Z_SCORE,
                    severity=Severity.EXTREME,
                ),
            ]
        )
        await session.flush()

        rows = await repository.list_recent(
            start=date(2026, 8, 1), end=date(2026, 8, 31), min_severity=Severity.HIGH
        )

        assert len(rows) == 1
        anomaly, symbol = rows[0]
        assert symbol == "FEED"
        assert anomaly.severity is Severity.EXTREME

    async def test_explanation_moves_an_anomaly_off_the_work_queue(
        self, session: AsyncSession
    ) -> None:
        ticker = await _make_ticker(session)
        repository = AnomalyRepository(session)
        await repository.bulk_upsert(
            [self._anomaly(ticker.id, day=date(2026, 9, 1), method=DetectionMethod.Z_SCORE)]
        )
        await session.flush()

        pending = await repository.list_unexplained()
        assert len(pending) == 1

        await repository.attach_explanation(
            pending[0].id,
            explanation="HBM demand forecasts were raised.",
            document_ids=["64f0c0ffee", "64f0c0fff0"],
        )
        await session.flush()

        assert await repository.list_unexplained() == []
        refreshed = await repository.get_or_raise(pending[0].id)
        assert refreshed.related_document_ids == ["64f0c0ffee", "64f0c0fff0"]


class TestCompanyRepository:
    """Tag containment, the query the GIN index exists for."""

    async def test_tag_search_matches_any_and_all(self, session: AsyncSession) -> None:
        session.add_all(
            [
                Company(slug="memco", name="MemCo", tags=["hbm", "dram"]),
                Company(slug="fabco", name="FabCo", tags=["foundry"]),
                Company(slug="bothco", name="BothCo", tags=["hbm", "foundry"]),
            ]
        )
        await session.flush()
        repository = CompanyRepository(session)

        any_match = await repository.list_by_tags([EcosystemTag.HBM])
        all_match = await repository.list_by_tags(
            [EcosystemTag.HBM, EcosystemTag.FOUNDRY], match_all=True
        )

        assert {company.slug for company in any_match} == {"memco", "bothco"}
        assert {company.slug for company in all_match} == {"bothco"}

    async def test_search_is_case_insensitive_substring_matching(
        self, session: AsyncSession
    ) -> None:
        session.add(Company(slug="micron", name="Micron Technology"))
        await session.flush()
        repository = CompanyRepository(session)

        assert [c.slug for c in await repository.search("MICRON tech")] == ["micron"]
        assert [c.slug for c in await repository.search("micron")] == ["micron"]
        assert [c.slug for c in await repository.search("cron")] == ["micron"]
        # Substring, not fuzzy: a transposition finds nothing. Semantic company
        # search is the vector index's job, not SQL's.
        assert await repository.search("Mircon") == []

    async def test_eager_loading_avoids_the_lazy_load_error(self, session: AsyncSession) -> None:
        """Relationships are ``lazy="raise_on_sql"``; the loader must be explicit."""
        ticker = await _make_ticker(session, symbol="EAGER")
        session.expunge_all()

        company = await CompanyRepository(session).get_with_tickers(ticker.company_id or 0)

        assert company is not None
        assert [listing.symbol for listing in company.tickers] == ["EAGER"]


class TestTickerRepository:
    """Symbol normalisation and the ingestion work queue."""

    async def test_symbol_lookup_is_case_insensitive(self, session: AsyncSession) -> None:
        await _make_ticker(session, symbol="NVDA")
        repository = TickerRepository(session)

        assert await repository.get_by_symbol("nvda") is not None
        assert await repository.get_by_symbol(" NvDa ") is not None

    async def test_stale_listing_queue_includes_never_ingested(self, session: AsyncSession) -> None:
        fresh = await _make_ticker(session, symbol="FRESH")
        await _make_ticker(session, symbol="NEVER")
        fresh.last_price_date = date(2026, 10, 10)
        await session.flush()

        stale = await TickerRepository(session).list_stale(before=date(2026, 10, 10))

        assert {ticker.symbol for ticker in stale} == {"NEVER"}

    async def test_watermarks_only_widen(self, session: AsyncSession) -> None:
        """A backfill and an incremental run must not undo each other."""
        ticker = await _make_ticker(session, symbol="MARK")
        repository = TickerRepository(session)
        now = datetime.now(UTC)

        await repository.update_watermarks(
            ticker.id,
            first_price_date=date(2026, 1, 10),
            last_price_date=date(2026, 6, 30),
            ingested_at=now,
        )
        # A backfill of older history arrives afterwards.
        await repository.update_watermarks(
            ticker.id,
            first_price_date=date(2020, 1, 1),
            last_price_date=date(2026, 3, 1),
            ingested_at=now,
        )
        await session.flush()
        await session.refresh(ticker)

        assert ticker.first_price_date == date(2020, 1, 1)
        assert ticker.last_price_date == date(2026, 6, 30)


class TestUserOwnedData:
    """Cascades and the constraints protecting user data."""

    async def _user(self, session: AsyncSession, email: str = "Analyst@Example.COM") -> User:
        user = User(email=email, hashed_password="not-a-real-hash")
        session.add(user)
        await session.flush()
        return user

    async def test_email_is_normalised_on_write(self, session: AsyncSession) -> None:
        user = await self._user(session)
        assert user.email == "analyst@example.com"
        assert await UserRepository(session).get_by_email("ANALYST@example.com") is not None

    async def test_duplicate_email_is_rejected(self, session: AsyncSession) -> None:
        await self._user(session, email="dup@example.com")
        session.add(User(email="DUP@example.com", hashed_password="x"))

        with pytest.raises(IntegrityError):
            await session.flush()

    async def test_deleting_a_user_cascades_to_grandchildren(self, session: AsyncSession) -> None:
        """Enforced by the database, not by ORM cascade rules."""
        user = await self._user(session, email="cascade@example.com")
        ticker = await _make_ticker(session, symbol="CASC")
        watchlist = Watchlist(user_id=user.id, name="AI Infra")
        portfolio = Portfolio(user_id=user.id, name="Core")
        session.add_all([watchlist, portfolio])
        await session.flush()

        await WatchlistRepository(session).add_ticker(watchlist.id, ticker.id)
        session.add(
            PortfolioPosition(
                portfolio_id=portfolio.id,
                ticker_id=ticker.id,
                quantity=Decimal("10"),
                average_cost=Decimal("100.50"),
            )
        )
        await session.flush()

        await session.execute(
            User.__table__.delete().where(User.id == user.id)  # bypasses the ORM
        )

        for model in (Watchlist, Portfolio, PortfolioPosition):
            remaining = await session.scalar(select(func.count()).select_from(model))
            assert remaining == 0, f"{model.__name__} rows survived the cascade"

    async def test_watchlist_positions_increment(self, session: AsyncSession) -> None:
        user = await self._user(session, email="order@example.com")
        watchlist = Watchlist(user_id=user.id, name="Ordered")
        session.add(watchlist)
        first = await _make_ticker(session, symbol="AAA")
        second = await _make_ticker(session, symbol="BBB")
        await session.flush()

        repository = WatchlistRepository(session)
        item_a = await repository.add_ticker(watchlist.id, first.id)
        await session.flush()
        item_b = await repository.add_ticker(watchlist.id, second.id)
        await session.flush()

        assert (item_a.position, item_b.position) == (0, 1)
        assert await repository.list_tracked_symbols(user.id) == ["AAA", "BBB"]

        assert await repository.remove_ticker(watchlist.id, first.id) is True
        assert await repository.remove_ticker(watchlist.id, first.id) is False

    async def test_negative_quantity_is_rejected(self, session: AsyncSession) -> None:
        user = await self._user(session, email="neg@example.com")
        ticker = await _make_ticker(session, symbol="NEG")
        portfolio = Portfolio(user_id=user.id, name="Bad")
        session.add(portfolio)
        await session.flush()
        session.add(
            PortfolioPosition(
                portfolio_id=portfolio.id,
                ticker_id=ticker.id,
                quantity=Decimal("-1"),
                average_cost=Decimal("1"),
            )
        )

        with pytest.raises(IntegrityError):
            await session.flush()

    async def test_portfolio_positions_load_eagerly(self, session: AsyncSession) -> None:
        user = await self._user(session, email="eager@example.com")
        ticker = await _make_ticker(session, symbol="PEAG")
        portfolio = Portfolio(user_id=user.id, name="Eager")
        session.add(portfolio)
        await session.flush()
        session.add(
            PortfolioPosition(
                portfolio_id=portfolio.id,
                ticker_id=ticker.id,
                quantity=Decimal("3"),
                average_cost=Decimal("50"),
            )
        )
        await session.flush()
        session.expunge_all()

        loaded = await PortfolioRepository(session).get_with_positions(portfolio.id)

        assert loaded is not None
        assert len(loaded.positions) == 1
        assert loaded.positions[0].cost_basis == Decimal("150")
        assert loaded.positions[0].ticker.symbol == "PEAG"


class TestSeed:
    """The seed must be safe to run on every deployment."""

    async def test_seeding_twice_is_idempotent(self, session: AsyncSession) -> None:
        first_companies, first_tickers = await seed_reference_data(session)
        second_companies, second_tickers = await seed_reference_data(session)

        assert (first_companies, first_tickers) == (second_companies, second_tickers)
        assert await session.scalar(select(func.count()).select_from(Company)) == (first_companies)
        assert await session.scalar(select(func.count()).select_from(Ticker)) == first_tickers

    async def test_etfs_are_seeded_without_a_company(self, session: AsyncSession) -> None:
        await seed_reference_data(session)

        etf = await TickerRepository(session).get_by_symbol("SMH")

        assert etf is not None
        assert etf.asset_type is AssetType.ETF
        assert etf.company_id is None

    async def test_hbm_suppliers_are_discoverable_by_tag(self, session: AsyncSession) -> None:
        """The domain question the tag index exists to answer."""
        await seed_reference_data(session)

        suppliers = await CompanyRepository(session).list_by_tags([EcosystemTag.HBM])

        assert {company.slug for company in suppliers} == {
            "micron",
            "sk-hynix",
            "samsung-electronics",
        }


@pytest.mark.integration
class TestWriteThenReadVisibility:
    """A write followed by a read in one transaction must see the write.

    Regression test for a bug found by driving the live API: adding a ticker
    returned a watchlist whose count was the value from *before* the add. The
    identity map returned the already-loaded instance and `selectinload`
    declined to refresh a populated collection, so the response contradicted
    the database.

    Only reproducible against a real session -- an in-memory fake has no
    identity map to go stale.
    """

    async def test_a_newly_added_watchlist_item_is_visible_on_re_read(
        self, session: AsyncSession
    ) -> None:
        users = UserRepository(session)
        watchlists = WatchlistRepository(session)
        tickers = TickerRepository(session)

        user = User(email="visibility@marketintel.io", hashed_password="x")
        users.add(user)
        await users.flush()

        watchlist = Watchlist(user_id=user.id, name="Visibility")
        watchlists.add(watchlist)
        await watchlists.flush()

        ticker = Ticker(
            symbol="VISI",
            display_name="Visibility Test",
            exchange="NASDAQ",
            asset_type=AssetType.ETF,
        )
        tickers.add(ticker)
        await tickers.flush()

        # Load once, so the identity map holds an instance with an empty
        # `items` collection -- exactly the state the bug depended on.
        first = await watchlists.get_with_items(watchlist.id)
        assert first is not None
        assert first.items == []

        await watchlists.add_ticker(watchlist.id, ticker.id)
        await watchlists.flush()

        second = await watchlists.get_with_items(watchlist.id)
        assert second is not None
        assert len(second.items) == 1
        assert second.items[0].ticker_id == ticker.id

    async def test_a_newly_added_position_is_visible_on_re_read(
        self, session: AsyncSession
    ) -> None:
        users = UserRepository(session)
        portfolios = PortfolioRepository(session)
        tickers = TickerRepository(session)

        user = User(email="visibility-pf@marketintel.io", hashed_password="x")
        users.add(user)
        await users.flush()

        portfolio = Portfolio(user_id=user.id, name="Visibility")
        portfolios.add(portfolio)
        await portfolios.flush()

        ticker = Ticker(
            symbol="VISP",
            display_name="Visibility PF Test",
            exchange="NASDAQ",
            asset_type=AssetType.ETF,
        )
        tickers.add(ticker)
        await tickers.flush()

        first = await portfolios.get_with_positions(portfolio.id)
        assert first is not None
        assert first.positions == []

        session.add(
            PortfolioPosition(
                portfolio_id=portfolio.id,
                ticker_id=ticker.id,
                quantity=Decimal("10"),
                average_cost=Decimal("100"),
            )
        )
        await portfolios.flush()

        second = await portfolios.get_with_positions(portfolio.id)
        assert second is not None
        assert len(second.positions) == 1
