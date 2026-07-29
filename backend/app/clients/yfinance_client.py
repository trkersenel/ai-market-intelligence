"""Price provider backed by yfinance.

yfinance is a synchronous, blocking library built on ``requests`` and pandas.
Calling it directly from a coroutine would stall the event loop for the duration
of every HTTP round trip -- with fourteen tickers that is several seconds during
which no other request, health check or job can make progress.

Every call therefore runs in a worker thread via ``asyncio.to_thread``, and the
rate limiter is acquired on the event loop *before* the thread is dispatched, so
throttling still applies across concurrent fetches.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from app.clients.protocols import PriceBar
from app.clients.rate_limiter import RateLimiter
from app.core.config import IngestionSettings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.models.enums import DataSource

logger = get_logger(__name__)

#: Columns yfinance returns, mapped to our field names. ``Adj Close`` is absent
#: when auto-adjustment is on, which is why the fetch requests raw prices and
#: adjusts explicitly -- see :meth:`YFinancePriceProvider.fetch_daily_bars`.
_COLUMN_ALIASES = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adjusted_close",
    "Volume": "volume",
}


class YFinancePriceProvider:
    """Fetches daily OHLCV history from Yahoo Finance."""

    def __init__(self, settings: IngestionSettings) -> None:
        """Configure throttling for the provider.

        Args:
            settings: Rate limit and timeout configuration.
        """
        self._settings = settings
        self._limiter = RateLimiter(rate_per_second=settings.yfinance_rate_limit)

    @property
    def source(self) -> DataSource:
        """Provenance recorded on every bar this provider produces."""
        return DataSource.YFINANCE

    async def fetch_daily_bars(self, symbol: str, *, start: date, end: date) -> list[PriceBar]:
        """Return daily bars for ``symbol`` within an inclusive window.

        Args:
            symbol: Yahoo Finance symbol, e.g. ``NVDA`` or ``000660.KS``.
            start: First session to fetch.
            end: Last session to fetch, inclusive.

        Returns:
            Bars ordered oldest first. Empty when the window contains no
            sessions -- a holiday week or a delisted symbol is not an error.

        Raises:
            ExternalServiceError: If yfinance raises or returns an unusable frame.
        """
        await self._limiter.acquire()

        try:
            frame = await asyncio.to_thread(self._download, symbol, start, end)
        except Exception as exc:
            msg = f"yfinance failed for {symbol}"
            raise ExternalServiceError(msg, details={"symbol": symbol}) from exc

        bars = self._to_bars(symbol, frame)
        logger.info(
            "prices_fetched",
            symbol=symbol,
            bars=len(bars),
            start=start.isoformat(),
            end=end.isoformat(),
        )
        return bars

    def _download(self, symbol: str, start: date, end: date) -> Any:
        """Blocking download, executed in a worker thread.

        ``auto_adjust=False`` keeps the raw OHLC *and* the adjusted close in the
        frame. With adjustment on, yfinance rewrites ``Close`` in place and drops
        ``Adj Close``, which would leave no way to show a chart at the prices
        that actually traded.

        ``end`` is exclusive in the yfinance API, so the caller's inclusive
        bound is shifted by a day here rather than at every call site.
        """
        # Lazy: importing yfinance pulls in pandas and numpy, ~1s of startup
        # cost that the API process pays for code only the worker runs.
        import yfinance  # noqa: PLC0415

        ticker = yfinance.Ticker(symbol)
        return ticker.history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
            raise_errors=True,
        )

    def _to_bars(self, symbol: str, frame: Any) -> list[PriceBar]:
        """Convert a pandas frame into validated bars, skipping unusable rows."""
        if frame is None or getattr(frame, "empty", True):
            return []

        renamed = {
            column: _COLUMN_ALIASES[column] for column in frame.columns if column in _COLUMN_ALIASES
        }
        missing = {"open", "high", "low", "close", "volume"} - set(renamed.values())
        if missing:
            msg = f"yfinance response for {symbol} is missing columns: {sorted(missing)}"
            raise ExternalServiceError(msg, details={"symbol": symbol})

        bars: list[PriceBar] = []
        for index, row in frame.iterrows():
            values = {field: row[column] for column, field in renamed.items()}
            bar = self._build_bar(symbol, index.date(), values)
            if bar is not None:
                bars.append(bar)

        bars.sort(key=lambda bar: bar.trade_date)
        return bars

    def _build_bar(self, symbol: str, trade_date: date, values: dict[str, Any]) -> PriceBar | None:
        """Build one bar, returning ``None`` if the row is unusable.

        Yahoo emits NaN rows for halted sessions and for the current day before
        the open. Dropping them individually is right: one bad row must not
        discard an otherwise good two-year backfill.
        """
        try:
            close = _to_decimal(values["close"])
            adjusted = values.get("adjusted_close")
            return PriceBar(
                trade_date=trade_date,
                open=_to_decimal(values["open"]),
                high=_to_decimal(values["high"]),
                low=_to_decimal(values["low"]),
                close=close,
                adjusted_close=_to_decimal(adjusted) if adjusted is not None else close,
                volume=int(values["volume"]),
            )
        except (TypeError, ValueError, InvalidOperation) as exc:
            logger.debug(
                "price_row_skipped",
                symbol=symbol,
                trade_date=trade_date.isoformat(),
                reason=str(exc),
            )
            return None


def _to_decimal(value: Any) -> Decimal:
    """Convert a numpy or Python float to an exact Decimal.

    Goes through ``str`` deliberately: ``Decimal(0.1)`` captures the full binary
    expansion (``0.1000000000000000055511151231257827``), while
    ``Decimal("0.1")`` is the number the vendor actually reported.

    Raises:
        ValueError: If the value is NaN, infinite or missing.
    """
    if value is None:
        msg = "missing value"
        raise ValueError(msg)

    text = str(value)
    if text in {"nan", "NaN", "inf", "-inf", "None", "<NA>"}:
        msg = f"non-finite value: {text}"
        raise ValueError(msg)

    result = Decimal(text)
    if not result.is_finite():
        msg = f"non-finite value: {text}"
        raise ValueError(msg)
    return result
