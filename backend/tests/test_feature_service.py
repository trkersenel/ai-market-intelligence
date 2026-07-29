"""Tests for the feature engineering service.

The arithmetic is covered by ``test_indicators``. These cover the *policy*: how
much history is loaded, how much is rewritten, what happens to a listing with
almost no data, and whether a missing benchmark degrades one column or the whole
run.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

import pytest

from app.core.exceptions import NotFoundError
from app.services.features.feature_service import (
    WARMUP_SESSIONS,
    FeatureEngineeringService,
)

START = date(2024, 1, 1)


class FakeTicker:
    """Stands in for the Ticker ORM model."""

    def __init__(self, ticker_id: int, symbol: str) -> None:
        self.id = ticker_id
        self.symbol = symbol


class FakeBar:
    """Stands in for a DailyPrice row."""

    def __init__(
        self,
        trade_date: date,
        close: float,
        volume: int = 1_000_000,
        *,
        is_provisional: bool = False,
    ) -> None:
        self.is_provisional = is_provisional
        price = Decimal(str(close))
        self.trade_date = trade_date
        self.open = price
        self.high = price + Decimal("1")
        self.low = price - Decimal("1")
        self.close = price
        self.adjusted_close = price
        self.volume = volume


class FakeTickerRepository:
    """Returns a fixed universe."""

    def __init__(self, tickers: Sequence[FakeTicker]) -> None:
        self._tickers = list(tickers)

    async def list_active(self, *, asset_type: object = None) -> list[FakeTicker]:
        return list(self._tickers)

    async def get_by_symbol(self, symbol: str) -> FakeTicker | None:
        wanted = symbol.strip().upper()
        return next((t for t in self._tickers if t.symbol == wanted), None)


class FakePriceRepository:
    """Serves scripted history and records how much was requested."""

    def __init__(self, bars_by_ticker: dict[int, list[FakeBar]]) -> None:
        self._bars = bars_by_ticker
        self.requested_sessions: list[int | None] = []
        self.completed_only_requested: list[bool] = []

    async def get_recent(
        self,
        ticker_id: int,
        *,
        sessions: int | None = None,
        completed_only: bool = False,
    ) -> list[FakeBar]:
        self.requested_sessions.append(sessions)
        self.completed_only_requested.append(completed_only)
        bars = self._bars.get(ticker_id, [])
        if completed_only:
            bars = [bar for bar in bars if not bar.is_provisional]
        return bars if sessions is None else bars[-sessions:]


class FakeFeatureRepository:
    """Collects the rows it was asked to upsert."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.deleted_after: list[tuple[int, date]] = []

    async def bulk_upsert(self, rows: Sequence[dict[str, Any]]) -> int:
        self.rows.extend(rows)
        return len(rows)

    async def delete_after(self, ticker_id: int, *, after: date) -> int:
        removed = [
            row for row in self.rows if row["ticker_id"] == ticker_id and row["trade_date"] > after
        ]
        self.rows = [row for row in self.rows if row not in removed]
        self.deleted_after.append((ticker_id, after))
        return len(removed)


def _series(count: int, *, start_price: float = 100.0, step: float = 0.5) -> list[FakeBar]:
    """Build a synthetic ascending price history over consecutive sessions."""
    return [
        FakeBar(START + timedelta(days=index), start_price + index * step) for index in range(count)
    ]


def _service(
    *,
    bars: dict[int, list[FakeBar]],
    tickers: Sequence[FakeTicker],
    features: FakeFeatureRepository | None = None,
    rewrite_sessions: int = 40,
    benchmark_symbol: str = "SMH",
) -> tuple[FeatureEngineeringService, FakePriceRepository, FakeFeatureRepository]:
    prices = FakePriceRepository(bars)
    feature_repo = features or FakeFeatureRepository()
    service = FeatureEngineeringService(
        tickers=FakeTickerRepository(tickers),  # type: ignore[arg-type]
        prices=prices,  # type: ignore[arg-type]
        features=feature_repo,  # type: ignore[arg-type]
        benchmark_symbol=benchmark_symbol,
        rewrite_sessions=rewrite_sessions,
    )
    return service, prices, feature_repo


class TestWarmupPolicy:
    """Loading enough history is what makes the stored values correct."""

    async def test_loads_warmup_history_beyond_the_rewrite_window(self) -> None:
        """Computing 40 sessions from 40 bars would null out SMA-200."""
        tickers = [FakeTicker(1, "NVDA")]
        service, prices, _ = _service(bars={1: _series(400)}, tickers=tickers, rewrite_sessions=40)

        await service.compute_all()

        assert prices.requested_sessions[-1] == 40 + WARMUP_SESSIONS

    async def test_writes_only_the_rewrite_window(self) -> None:
        tickers = [FakeTicker(1, "NVDA")]
        service, _, features = _service(
            bars={1: _series(400)}, tickers=tickers, rewrite_sessions=30
        )

        report = await service.compute_all()

        assert report.rows_written == 30
        assert len(features.rows) == 30

    async def test_full_history_mode_loads_and_writes_everything(self) -> None:
        tickers = [FakeTicker(1, "NVDA")]
        service, prices, features = _service(bars={1: _series(120)}, tickers=tickers)

        report = await service.compute_all(full_history=True)

        assert prices.requested_sessions[-1] is None
        assert report.rows_written == 120
        assert len(features.rows) == 120

    async def test_sma_200_is_populated_once_history_allows(self) -> None:
        """The end-to-end point of the warm-up: the slow trend is not null."""
        tickers = [FakeTicker(1, "NVDA")]
        service, _, features = _service(bars={1: _series(400)}, tickers=tickers)

        await service.compute_all()

        assert all(row["sma_200"] is not None for row in features.rows)


class TestShortHistory:
    """A young listing must degrade, not crash."""

    async def test_a_listing_with_almost_no_history_is_skipped(self) -> None:
        tickers = [FakeTicker(1, "NEW")]
        service, _, features = _service(bars={1: _series(3)}, tickers=tickers)

        report = await service.compute_all()

        assert report.rows_written == 0
        assert features.rows == []
        assert report.failures == ()

    async def test_a_listing_with_no_history_at_all_is_skipped(self) -> None:
        tickers = [FakeTicker(1, "EMPTY")]
        service, _, _ = _service(bars={}, tickers=tickers)

        report = await service.compute_all()

        assert report.rows_written == 0
        assert report.failures == ()

    async def test_undefined_indicators_are_stored_as_null(self) -> None:
        """A 30-bar listing has an SMA-20 but no SMA-200; both must be honest."""
        tickers = [FakeTicker(1, "YOUNG")]
        service, _, features = _service(bars={1: _series(30)}, tickers=tickers)

        await service.compute_all(full_history=True)

        last = features.rows[-1]
        assert last["sma_20"] is not None
        assert last["sma_200"] is None
        assert last["daily_return"] is not None


class TestBenchmark:
    """Relative strength depends on a second series that may be absent."""

    async def test_relative_strength_is_computed_against_the_benchmark(self) -> None:
        tickers = [FakeTicker(1, "NVDA"), FakeTicker(2, "SMH")]
        service, _, features = _service(
            bars={1: _series(300, step=1.0), 2: _series(300, step=0.1)},
            tickers=tickers,
        )

        await service.compute_all()

        nvda_rows = [row for row in features.rows if row["ticker_id"] == 1]
        assert all(row["relative_strength_smh"] is not None for row in nvda_rows[1:])

    async def test_a_faster_riser_shows_positive_excess_return(self) -> None:
        tickers = [FakeTicker(1, "NVDA"), FakeTicker(2, "SMH")]
        service, _, features = _service(
            bars={1: _series(300, step=2.0), 2: _series(300, step=0.1)},
            tickers=tickers,
        )

        await service.compute_all()

        nvda_rows = [row for row in features.rows if row["ticker_id"] == 1]
        assert nvda_rows[-1]["relative_strength_smh"] > 0

    async def test_a_missing_benchmark_degrades_one_column_not_the_run(self) -> None:
        """Every other feature must still compute when SMH is not yet tracked."""
        tickers = [FakeTicker(1, "NVDA")]
        service, _, features = _service(
            bars={1: _series(300)}, tickers=tickers, benchmark_symbol="SMH"
        )

        report = await service.compute_all()

        assert report.failures == ()
        assert report.rows_written > 0
        assert all(row["relative_strength_smh"] is None for row in features.rows)
        assert all(row["rsi_14"] is not None for row in features.rows)

    async def test_the_benchmark_is_loaded_once_per_run(self) -> None:
        """Re-reading SMH for each of fourteen listings would be pure waste."""
        tickers = [FakeTicker(i, f"SYM{i}") for i in range(1, 6)] + [FakeTicker(9, "SMH")]
        bars = {ticker.id: _series(300) for ticker in tickers}
        service, prices, _ = _service(bars=bars, tickers=tickers)

        await service.compute_all()

        # One benchmark read, then one read per listing.
        assert len(prices.requested_sessions) == 1 + len(tickers)


class TestRowShape:
    """What actually reaches the repository."""

    async def test_rows_carry_the_natural_key_and_every_indicator(self) -> None:
        tickers = [FakeTicker(1, "NVDA")]
        service, _, features = _service(bars={1: _series(300)}, tickers=tickers)

        await service.compute_all()

        row = features.rows[-1]
        assert row["ticker_id"] == 1
        assert isinstance(row["trade_date"], date)
        for column in (
            "daily_return",
            "weekly_return",
            "monthly_return",
            "sma_20",
            "sma_50",
            "sma_200",
            "ema_12",
            "ema_26",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_histogram",
            "bollinger_upper",
            "bollinger_middle",
            "bollinger_lower",
            "atr_14",
            "volatility_20",
            "volume_sma_20",
            "volume_ratio",
            "relative_strength_smh",
        ):
            assert column in row, f"{column} missing from the upsert row"

    async def test_values_are_decimals_not_floats(self) -> None:
        """The column is NUMERIC; handing it a float would round at the driver."""
        tickers = [FakeTicker(1, "NVDA")]
        service, _, features = _service(bars={1: _series(300)}, tickers=tickers)

        await service.compute_all()

        row = features.rows[-1]
        assert isinstance(row["rsi_14"], Decimal)
        assert isinstance(row["sma_20"], Decimal)

    async def test_rows_are_ordered_oldest_first(self) -> None:
        tickers = [FakeTicker(1, "NVDA")]
        service, _, features = _service(bars={1: _series(300)}, tickers=tickers)

        await service.compute_all()

        dates = [row["trade_date"] for row in features.rows]
        assert dates == sorted(dates)


class TestSingleSymbol:
    """Targeted recomputation."""

    async def test_computes_one_listing(self) -> None:
        tickers = [FakeTicker(1, "NVDA"), FakeTicker(2, "MU")]
        service, _, features = _service(bars={1: _series(300), 2: _series(300)}, tickers=tickers)

        result = await service.compute_symbol("nvda")

        assert result.succeeded
        assert result.rows_written > 0
        assert {row["ticker_id"] for row in features.rows} == {1}

    async def test_unknown_symbol_raises_not_found(self) -> None:
        service, _, _ = _service(bars={}, tickers=[])

        with pytest.raises(NotFoundError):
            await service.compute_symbol("NOPE")


class TestReport:
    """The run summary the scheduler logs."""

    async def test_report_aggregates_every_listing(self) -> None:
        tickers = [FakeTicker(1, "NVDA"), FakeTicker(2, "MU"), FakeTicker(3, "TSM")]
        bars = {ticker.id: _series(300) for ticker in tickers}
        service, _, _ = _service(bars=bars, tickers=tickers, rewrite_sessions=10)

        report = await service.compute_all()

        assert len(report.results) == 3
        assert report.rows_written == 30
        assert report.duration_seconds >= 0


class TestProvisionalBars:
    """A still-trading session must never reach a statistic."""

    async def test_features_are_computed_from_completed_sessions_only(self) -> None:
        service, prices, _ = _service(bars={1: _series(300)}, tickers=[FakeTicker(1, "NVDA")])

        await service.compute_all()

        assert all(prices.completed_only_requested)

    async def test_a_partial_bar_does_not_become_the_last_indicator_row(self) -> None:
        """Otherwise every day's newest row would carry a fake volume collapse."""
        history = _series(300)
        partial = FakeBar(START + timedelta(days=300), 250.0, volume=16_000, is_provisional=True)
        service, _, features = _service(
            bars={1: [*history, partial]}, tickers=[FakeTicker(1, "NVDA")]
        )

        await service.compute_all()

        assert features.rows[-1]["trade_date"] == history[-1].trade_date

    async def test_the_partial_bar_would_have_skewed_the_volume_ratio(self) -> None:
        """Demonstrates the bug the flag prevents, by including the bar anyway.

        With a 16k-share partial day against a 1M-share average, volume_ratio
        computes to roughly 0.016 -- indistinguishable from a genuine collapse
        in participation, and exactly what the anomaly detector would fire on.
        """
        history = _series(300)
        partial = FakeBar(START + timedelta(days=300), 250.0, volume=16_000, is_provisional=False)
        service, _, features = _service(
            bars={1: [*history, partial]}, tickers=[FakeTicker(1, "NVDA")]
        )

        await service.compute_all()

        assert float(features.rows[-1]["volume_ratio"]) < 0.1

    async def test_stale_rows_past_the_last_completed_session_are_retired(self) -> None:
        """A row computed yesterday from a then-partial bar must not survive.

        Upsert alone cannot fix it: the run no longer writes that session, so
        without an explicit sweep the wrong row stays the newest one the API
        serves.
        """
        history = _series(300)
        partial = FakeBar(START + timedelta(days=300), 250.0, volume=16_000, is_provisional=True)
        service, _, features = _service(
            bars={1: [*history, partial]}, tickers=[FakeTicker(1, "NVDA")]
        )

        await service.compute_all()

        assert features.deleted_after == [(1, history[-1].trade_date)]
