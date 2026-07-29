"""Technical indicator endpoints."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Query, status

from app.api.deps import FeatureServiceDep, IndicatorRepoDep, TickerRepoDep
from app.core.exceptions import NotFoundError, ValidationError
from app.schemas.market import (
    IndicatorSeries,
    IndicatorSnapshot,
    IngestionRunResponse,
)

router = APIRouter(tags=["indicators"])

MAX_WINDOW_DAYS = 365 * 2
DEFAULT_WINDOW_DAYS = 180


@router.get(
    "/{symbol}",
    response_model=IndicatorSeries,
    summary="Get a technical indicator series",
    responses={status.HTTP_404_NOT_FOUND: {"description": "No such ticker."}},
)
async def get_indicator_series(
    symbol: str,
    tickers: TickerRepoDep,
    indicators: IndicatorRepoDep,
    start: Annotated[date | None, Query(description="Inclusive first session.")] = None,
    end: Annotated[date | None, Query(description="Inclusive last session.")] = None,
) -> IndicatorSeries:
    """Return computed indicators for one listing over a window.

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

    rows = await indicators.get_range(ticker.id, start=window_start, end=window_end)
    return IndicatorSeries(
        symbol=ticker.symbol,
        start=rows[0].trade_date if rows else None,
        end=rows[-1].trade_date if rows else None,
        rows=[IndicatorSnapshot.model_validate(row) for row in rows],
    )


@router.get(
    "/{symbol}/latest",
    response_model=IndicatorSnapshot,
    summary="Get the most recent indicator snapshot",
    description=(
        "The current technical picture for one listing: trend, momentum, "
        "volatility and volume against their own baselines."
    ),
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "No ticker, or no features computed yet."}
    },
)
async def get_latest_indicators(
    symbol: str, tickers: TickerRepoDep, indicators: IndicatorRepoDep
) -> IndicatorSnapshot:
    """Return the newest computed indicator row for one listing.

    Raises:
        NotFoundError: If the symbol is not tracked or has no computed features.
    """
    ticker = await tickers.get_by_symbol(symbol)
    if ticker is None:
        msg = f"Ticker {symbol.upper()!r} is not tracked."
        raise NotFoundError(msg, details={"symbol": symbol.upper()})

    latest = await indicators.get_latest(ticker.id)
    if latest is None:
        msg = f"No indicators computed for {ticker.symbol!r} yet."
        raise NotFoundError(msg, details={"symbol": ticker.symbol})
    return IndicatorSnapshot.model_validate(latest)


@router.post(
    "/compute",
    response_model=IngestionRunResponse,
    summary="Recompute indicators for every listing",
    description=(
        "Recomputes from stored prices. Set full_history to rewrite every "
        "session, which is needed after an indicator definition changes."
    ),
)
async def compute_indicators(
    service: FeatureServiceDep,
    full_history: Annotated[
        bool, Query(description="Rewrite all sessions rather than the recent tail.")
    ] = False,
) -> IngestionRunResponse:
    """Run feature engineering across the tracked universe."""
    report = await service.compute_all(full_history=full_history)
    return IngestionRunResponse(
        started_at=report.started_at,
        finished_at=report.finished_at,
        duration_seconds=round(report.duration_seconds, 3),
        items_written=report.rows_written,
        failures=[f"{failure.symbol}: {failure.error}" for failure in report.failures],
    )
