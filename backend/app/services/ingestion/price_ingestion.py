"""Price ingestion: fetch, normalise, upsert, advance the watermark.

The service owns the *policy* -- which tickers to fetch, over what window, how
many at once, what to do when one fails. The provider owns the transport and
the repository owns persistence, so this module contains no HTTP and no SQL.

Every run is safe to repeat. Windows deliberately overlap the last stored
session, upserts absorb the overlap, and watermarks only ever widen, so a retry
after a partial failure converges rather than compounding.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from app.clients.protocols import PriceBar, PriceProvider
from app.core.config import IngestionSettings
from app.core.exceptions import ExternalServiceError, NotFoundError
from app.core.logging import get_logger
from app.models.company import Ticker
from app.repositories.company import TickerRepository
from app.repositories.price import DailyPriceRepository

logger = get_logger(__name__)


@dataclass(frozen=True)
class TickerIngestionResult:
    """Outcome of ingesting one listing."""

    symbol: str
    bars_written: int
    first_date: date | None = None
    last_date: date | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the fetch completed without an error."""
        return self.error is None


@dataclass(frozen=True)
class IngestionReport:
    """Aggregate outcome of one ingestion run."""

    started_at: datetime
    finished_at: datetime
    results: tuple[TickerIngestionResult, ...]

    @property
    def bars_written(self) -> int:
        """Total rows inserted or updated across every listing."""
        return sum(result.bars_written for result in self.results)

    @property
    def failures(self) -> tuple[TickerIngestionResult, ...]:
        """Listings whose fetch failed."""
        return tuple(result for result in self.results if not result.succeeded)

    @property
    def duration_seconds(self) -> float:
        """Wall-clock duration of the run."""
        return (self.finished_at - self.started_at).total_seconds()


class PriceIngestionService:
    """Keeps the daily price table current for every active listing."""

    def __init__(
        self,
        *,
        provider: PriceProvider,
        tickers: TickerRepository,
        prices: DailyPriceRepository,
        settings: IngestionSettings,
    ) -> None:
        """Wire the service to its collaborators.

        Args:
            provider: Anything satisfying :class:`PriceProvider` -- the real
                yfinance client in production, a fake in tests.
            tickers: Listing repository, read for the work queue and written for
                the watermarks.
            prices: Bar repository.
            settings: Window sizes and concurrency limits.
        """
        self._provider = provider
        self._tickers = tickers
        self._prices = prices
        self._settings = settings

    async def ingest_all(self, *, as_of: date | None = None) -> IngestionReport:
        """Bring every active listing up to date.

        Args:
            as_of: Treat this as today. Lets a backfill or a test target a past
                date without patching the clock.

        Returns:
            A report naming every listing processed and every failure.

        Notes:
            Failures are collected, not raised. One delisted or rate-limited
            symbol must not abandon the other thirteen -- the report is what the
            scheduler logs and what an operator acts on.
        """
        started = datetime.now(UTC)
        target_date = as_of or started.date()
        listings = await self._tickers.list_active()

        semaphore = asyncio.Semaphore(self._settings.max_concurrent_fetches)
        results = await asyncio.gather(
            *(self._ingest_one(listing, target_date, semaphore) for listing in listings)
        )

        report = IngestionReport(
            started_at=started,
            finished_at=datetime.now(UTC),
            results=tuple(results),
        )
        logger.info(
            "price_ingestion_complete",
            listings=len(listings),
            bars_written=report.bars_written,
            failures=len(report.failures),
            duration_seconds=round(report.duration_seconds, 2),
        )
        return report

    async def ingest_symbol(
        self, symbol: str, *, as_of: date | None = None
    ) -> TickerIngestionResult:
        """Bring a single listing up to date, by symbol.

        Raises:
            NotFoundError: If the symbol is not tracked.
        """
        listing = await self._tickers.get_by_symbol(symbol)
        if listing is None:
            msg = f"Ticker {symbol!r} is not tracked."
            raise NotFoundError(msg, details={"symbol": symbol})

        target_date = as_of or datetime.now(UTC).date()
        return await self._ingest_one(listing, target_date, asyncio.Semaphore(1))

    async def _ingest_one(
        self, listing: Ticker, target_date: date, semaphore: asyncio.Semaphore
    ) -> TickerIngestionResult:
        """Fetch and persist one listing's window, converting failures to a result."""
        window_start = self._window_start(listing, target_date)
        if window_start > target_date:
            return TickerIngestionResult(symbol=listing.symbol, bars_written=0)

        async with semaphore:
            try:
                bars = await self._provider.fetch_daily_bars(
                    listing.symbol, start=window_start, end=target_date
                )
            except ExternalServiceError as exc:
                logger.warning("price_fetch_failed", symbol=listing.symbol, error=exc.message)
                return TickerIngestionResult(
                    symbol=listing.symbol, bars_written=0, error=exc.message
                )

        if not bars:
            return TickerIngestionResult(symbol=listing.symbol, bars_written=0)

        written = await self._prices.bulk_upsert(
            [self._to_row(listing.id, bar, target_date) for bar in bars]
        )
        await self._tickers.update_watermarks(
            listing.id,
            first_price_date=bars[0].trade_date,
            last_price_date=bars[-1].trade_date,
            ingested_at=datetime.now(UTC),
        )
        return TickerIngestionResult(
            symbol=listing.symbol,
            bars_written=written,
            first_date=bars[0].trade_date,
            last_date=bars[-1].trade_date,
        )

    def _window_start(self, listing: Ticker, target_date: date) -> date:
        """Decide the first session to request for a listing.

        A listing with no history gets the full backfill. One that is current
        gets a short window that *overlaps* what is already stored: vendors
        restate recent bars for splits and late corrections, and re-fetching a
        few sessions costs almost nothing because the upsert makes the overlap
        free.
        """
        if listing.last_price_date is None:
            return target_date - timedelta(days=self._settings.initial_backfill_days)
        return listing.last_price_date - timedelta(days=self._settings.incremental_overlap_days)

    @staticmethod
    def _to_row(ticker_id: int, bar: PriceBar, target_date: date) -> dict[str, object]:
        """Flatten a bar into the mapping the repository upserts.

        A bar dated on the run date is marked provisional: the session may still
        be trading, so its volume and close are partial. Whether the market has
        actually closed is not knowable here without an exchange calendar, so
        the conservative reading is taken and corrected on the next run.
        """
        return {
            "ticker_id": ticker_id,
            "trade_date": bar.trade_date,
            "is_provisional": bar.trade_date >= target_date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "adjusted_close": bar.adjusted_close,
            "volume": bar.volume,
        }
