"""Feature engineering: compute indicators from stored prices and persist them.

The service owns the *policy* -- how much history to load, how many sessions to
rewrite, what to do when a listing is too young -- while the arithmetic lives in
:mod:`app.services.features.indicators` as pure functions and persistence lives
in the repository. Nothing here does arithmetic on prices or writes SQL.

The central concern is the warm-up problem. An SMA-200 needs 200 prior sessions;
computing features for "the last 30 days" from only 30 days of prices would
produce nulls where real values exist, and re-running later would overwrite good
rows with worse ones. So each run loads a long history, computes across all of
it, and writes back only the recent tail.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.db.seed_data import BENCHMARK_SYMBOL
from app.models.price import DailyPrice
from app.repositories.company import TickerRepository
from app.repositories.price import DailyPriceRepository, TechnicalIndicatorRepository
from app.services.features import indicators as ind

logger = get_logger(__name__)

#: Sessions of history loaded before the window being written. Sized by the
#: longest indicator (SMA-200) plus headroom, so every feature is fully warmed
#: up by the time the written window begins. Too small and the platform would
#: quietly store nulls for its slowest-moving trend signal.
WARMUP_SESSIONS = 260

#: Sessions actually rewritten on a routine run. Recent rows are refreshed
#: because a late vendor correction to a price changes every indicator that
#: touched it.
DEFAULT_REWRITE_SESSIONS = 40

#: Precision used when converting a float indicator back to NUMERIC. Six places
#: is far beyond the meaningful precision of these statistics and matches the
#: column scale, so nothing is lost at the boundary.
STORAGE_PRECISION = Decimal("0.000001")


@dataclass(frozen=True)
class FeatureResult:
    """Outcome of computing features for one listing."""

    symbol: str
    rows_written: int
    first_date: date | None = None
    last_date: date | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether computation completed without an error."""
        return self.error is None


@dataclass(frozen=True)
class FeatureReport:
    """Aggregate outcome of one feature-engineering run."""

    started_at: datetime
    finished_at: datetime
    results: tuple[FeatureResult, ...]

    @property
    def rows_written(self) -> int:
        """Total indicator rows inserted or updated."""
        return sum(result.rows_written for result in self.results)

    @property
    def failures(self) -> tuple[FeatureResult, ...]:
        """Listings whose computation failed."""
        return tuple(result for result in self.results if not result.succeeded)

    @property
    def duration_seconds(self) -> float:
        """Wall-clock duration of the run."""
        return (self.finished_at - self.started_at).total_seconds()


class FeatureEngineeringService:
    """Computes and stores technical indicators for every tracked listing."""

    def __init__(
        self,
        *,
        tickers: TickerRepository,
        prices: DailyPriceRepository,
        features: TechnicalIndicatorRepository,
        benchmark_symbol: str = BENCHMARK_SYMBOL,
        rewrite_sessions: int = DEFAULT_REWRITE_SESSIONS,
    ) -> None:
        """Wire the service to its collaborators.

        Args:
            tickers: Listing repository, read for the work queue.
            prices: Source of the OHLCV history features are derived from.
            features: Destination repository for computed indicators.
            benchmark_symbol: Listing used for relative strength, normally SMH.
            rewrite_sessions: How many recent sessions each run rewrites.
        """
        self._tickers = tickers
        self._prices = prices
        self._features = features
        self._benchmark_symbol = benchmark_symbol
        self._rewrite_sessions = rewrite_sessions

    async def compute_all(self, *, full_history: bool = False) -> FeatureReport:
        """Compute features for every active listing.

        Args:
            full_history: Rewrite every session rather than the recent tail.
                Used after changing an indicator's definition, when previously
                stored values are no longer trustworthy.

        Returns:
            A report naming every listing processed and every failure.

        Notes:
            Listings are processed sequentially, not concurrently. Unlike
            ingestion, this work is CPU-bound and shares one database session;
            running it in parallel on a single event loop would add contention
            without adding throughput.
        """
        started = datetime.now(UTC)
        listings = await self._tickers.list_active()
        benchmark_returns = await self._load_benchmark_returns()

        results: list[FeatureResult] = []
        for listing in listings:
            results.append(
                await self._compute_one(
                    ticker_id=listing.id,
                    symbol=listing.symbol,
                    benchmark_returns=benchmark_returns,
                    full_history=full_history,
                )
            )

        report = FeatureReport(
            started_at=started, finished_at=datetime.now(UTC), results=tuple(results)
        )
        logger.info(
            "feature_run_complete",
            listings=len(listings),
            rows_written=report.rows_written,
            failures=len(report.failures),
            duration_seconds=round(report.duration_seconds, 2),
        )
        return report

    async def compute_symbol(self, symbol: str, *, full_history: bool = False) -> FeatureResult:
        """Compute features for one listing.

        Raises:
            NotFoundError: If the symbol is not tracked.
        """
        listing = await self._tickers.get_by_symbol(symbol)
        if listing is None:
            msg = f"Ticker {symbol.upper()!r} is not tracked."
            raise NotFoundError(msg, details={"symbol": symbol.upper()})

        return await self._compute_one(
            ticker_id=listing.id,
            symbol=listing.symbol,
            benchmark_returns=await self._load_benchmark_returns(),
            full_history=full_history,
        )

    async def _compute_one(
        self,
        *,
        ticker_id: int,
        symbol: str,
        benchmark_returns: dict[date, float],
        full_history: bool,
    ) -> FeatureResult:
        """Load history, compute every indicator, and persist the tail."""
        # `None` means the entire stored history, used when an indicator's
        # definition changed and previously written values are untrustworthy.
        sessions = None if full_history else self._rewrite_sessions + WARMUP_SESSIONS
        # `completed_only`: a still-trading session has partial volume and a
        # close that is not yet the close. Every indicator here is a
        # statistic, and a statistic computed from half a day is wrong in a
        # way that looks like a signal -- a volume ratio of 0.13 reads as a
        # collapse in participation rather than as lunchtime.
        bars = await self._prices.get_recent(ticker_id, sessions=sessions, completed_only=True)

        if len(bars) < ind.SESSIONS_PER_WEEK:
            # Too young to say anything: even a weekly return is undefined.
            logger.info("features_skipped", symbol=symbol, bars=len(bars))
            return FeatureResult(symbol=symbol, rows_written=0)

        rows = self._build_rows(ticker_id, bars, benchmark_returns)
        if not full_history:
            rows = rows[-self._rewrite_sessions :]
        if not rows:
            return FeatureResult(symbol=symbol, rows_written=0)

        written = await self._features.bulk_upsert(rows)

        # Retire rows for sessions no longer computable -- the bar that produced
        # them has since been reclassified as still-trading. Without this the
        # stale row stays the newest one the API serves.
        await self._features.delete_after(ticker_id, after=rows[-1]["trade_date"])

        return FeatureResult(
            symbol=symbol,
            rows_written=written,
            first_date=rows[0]["trade_date"],
            last_date=rows[-1]["trade_date"],
        )

    def _build_rows(
        self,
        ticker_id: int,
        bars: Sequence[DailyPrice],
        benchmark_returns: dict[date, float],
    ) -> list[dict[str, Any]]:
        """Compute every indicator over ``bars`` and flatten to upsertable rows."""
        dates = [bar.trade_date for bar in bars]
        # Adjusted closes drive every return-based feature; raw OHLC drives the
        # range-based ones, because ATR describes what actually traded.
        adjusted = [float(bar.adjusted_close) for bar in bars]
        closes = [float(bar.close) for bar in bars]
        highs = [float(bar.high) for bar in bars]
        lows = [float(bar.low) for bar in bars]
        volumes = [float(bar.volume) for bar in bars]

        daily = ind.simple_returns(adjusted)
        weekly = ind.simple_returns(adjusted, ind.SESSIONS_PER_WEEK)
        monthly = ind.simple_returns(adjusted, ind.SESSIONS_PER_MONTH)

        macd_result = ind.macd(adjusted)
        bands = ind.bollinger_bands(adjusted)

        aligned_benchmark = [benchmark_returns.get(day) for day in dates]
        excess = ind.relative_strength(daily, aligned_benchmark)

        computed = {
            "daily_return": daily,
            "weekly_return": weekly,
            "monthly_return": monthly,
            "sma_20": ind.simple_moving_average(adjusted, 20),
            "sma_50": ind.simple_moving_average(adjusted, 50),
            "sma_200": ind.simple_moving_average(adjusted, 200),
            "ema_12": ind.exponential_moving_average(adjusted, 12),
            "ema_26": ind.exponential_moving_average(adjusted, 26),
            "rsi_14": ind.relative_strength_index(adjusted),
            "macd": macd_result.macd,
            "macd_signal": macd_result.signal,
            "macd_histogram": macd_result.histogram,
            "bollinger_upper": bands.upper,
            "bollinger_middle": bands.middle,
            "bollinger_lower": bands.lower,
            "atr_14": ind.average_true_range(highs, lows, closes),
            "volatility_20": ind.realised_volatility(adjusted),
            "volume_sma_20": ind.simple_moving_average(volumes, 20),
            "volume_ratio": ind.volume_ratio(volumes),
            "relative_strength_smh": excess,
        }

        return [
            {
                "ticker_id": ticker_id,
                "trade_date": day,
                **{name: _to_decimal(series[index]) for name, series in computed.items()},
            }
            for index, day in enumerate(dates)
        ]

    async def _load_benchmark_returns(self) -> dict[date, float]:
        """Return the benchmark's daily returns, keyed by session.

        Loaded once per run and shared across listings, rather than re-read per
        symbol. Returns an empty mapping when the benchmark has no history yet;
        relative strength is then ``None`` and every other feature still
        computes -- a missing benchmark must degrade one column, not the run.
        """
        benchmark = await self._tickers.get_by_symbol(self._benchmark_symbol)
        if benchmark is None:
            logger.warning("benchmark_missing", symbol=self._benchmark_symbol)
            return {}

        bars = await self._prices.get_recent(
            benchmark.id,
            sessions=self._rewrite_sessions + WARMUP_SESSIONS,
            completed_only=True,
        )
        if not bars:
            logger.warning("benchmark_has_no_prices", symbol=self._benchmark_symbol)
            return {}

        adjusted = [float(bar.adjusted_close) for bar in bars]
        returns = ind.simple_returns(adjusted)
        return {
            bar.trade_date: value
            for bar, value in zip(bars, returns, strict=True)
            if value is not None
        }


def _to_decimal(value: float | None) -> Decimal | None:
    """Convert an indicator value to the exact type the column stores.

    Non-finite results -- a division that produced ``inf`` or ``nan`` on
    pathological data -- are stored as ``NULL`` rather than raising. A single
    unusable feature must not discard the row's other nineteen.
    """
    if value is None:
        return None
    try:
        converted = Decimal(str(value))
    except (ValueError, ArithmeticError):
        return None
    if not converted.is_finite():
        return None
    return converted.quantize(STORAGE_PRECISION)
