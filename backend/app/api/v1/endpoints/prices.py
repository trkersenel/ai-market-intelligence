"""Price and quote endpoints."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Query, status

from app.api.deps import PriceRepoDep, TickerRepoDep
from app.core.exceptions import NotFoundError, ValidationError
from app.schemas.market import (
    PriceBarResponse,
    PriceSeries,
    TickerQuote,
    quote_from_bars,
)

router = APIRouter(tags=["prices"])

#: Upper bound on a single request's window. Two years of daily bars is roughly
#: 500 rows -- enough for any chart, small enough that no request can ask the
#: database for an unbounded scan.
MAX_WINDOW_DAYS = 365 * 2
DEFAULT_WINDOW_DAYS = 180


@router.get(
    "/{symbol}",
    response_model=PriceSeries,
    summary="Get a price series",
    responses={status.HTTP_404_NOT_FOUND: {"description": "No such ticker."}},
)
async def get_price_series(
    symbol: str,
    tickers: TickerRepoDep,
    prices: PriceRepoDep,
    start: Annotated[date | None, Query(description="Inclusive first session.")] = None,
    end: Annotated[date | None, Query(description="Inclusive last session.")] = None,
) -> PriceSeries:
    """Return daily bars for one listing over a window.

    Defaults to the last 180 sessions when no window is given.

    Raises:
        NotFoundError: If the symbol is not tracked.
        ValidationError: If the window is inverted or wider than the cap.
    """
    ticker = await tickers.get_by_symbol(symbol)
    if ticker is None:
        msg = f"Ticker {symbol.upper()!r} is not tracked."
        raise NotFoundError(msg, details={"symbol": symbol.upper()})

    window_end = end or datetime.now(UTC).date()
    window_start = start or window_end - timedelta(days=DEFAULT_WINDOW_DAYS)

    if window_start > window_end:
        msg = "start must not be after end."
        raise ValidationError(msg, details={"start": str(window_start), "end": str(window_end)})
    if (window_end - window_start).days > MAX_WINDOW_DAYS:
        msg = f"Window must not exceed {MAX_WINDOW_DAYS} days."
        raise ValidationError(msg, details={"max_days": MAX_WINDOW_DAYS})

    bars = await prices.get_range(ticker.id, start=window_start, end=window_end)
    return PriceSeries(
        symbol=ticker.symbol,
        start=bars[0].trade_date if bars else None,
        end=bars[-1].trade_date if bars else None,
        bars=[PriceBarResponse.model_validate(bar) for bar in bars],
    )


@router.get(
    "/{symbol}/latest",
    response_model=TickerQuote,
    summary="Get the latest quote",
    responses={status.HTTP_404_NOT_FOUND: {"description": "No ticker, or no bars yet."}},
)
async def get_latest_quote(
    symbol: str, tickers: TickerRepoDep, prices: PriceRepoDep
) -> TickerQuote:
    """Return the most recent bar for one listing, with its session change.

    Raises:
        NotFoundError: If the symbol is not tracked or has no stored bars.
    """
    ticker = await tickers.get_by_symbol(symbol)
    if ticker is None:
        msg = f"Ticker {symbol.upper()!r} is not tracked."
        raise NotFoundError(msg, details={"symbol": symbol.upper()})

    # Two bars in one query: the latest, and the one before it for the change.
    recent = await prices.get_recent(ticker.id, sessions=2)
    if not recent:
        msg = f"No prices stored for {ticker.symbol!r} yet."
        raise NotFoundError(msg, details={"symbol": ticker.symbol})

    latest = recent[-1]
    previous = recent[-2] if len(recent) > 1 else None
    return quote_from_bars(ticker, latest, previous)
