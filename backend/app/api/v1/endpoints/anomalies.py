"""Anomaly endpoints."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Query, status

from app.api.deps import AnomalyRepoDep, AnomalyServiceDep, TickerRepoDep
from app.core.exceptions import NotFoundError
from app.models.enums import AnomalyType, Severity
from app.schemas.market import AnomalyResponse, IngestionRunResponse

router = APIRouter(tags=["anomalies"])

DEFAULT_LOOKBACK_DAYS = 30
MAX_LOOKBACK_DAYS = 365


@router.get(
    "",
    response_model=list[AnomalyResponse],
    summary="List detected anomalies",
    description=(
        "Across the tracked universe, most recent first. Each anomaly records "
        "which detector fired, how far the observation sat from its own "
        "baseline, and -- once the correlation engine has run -- why."
    ),
)
async def list_anomalies(
    anomalies: AnomalyRepoDep,
    days: Annotated[
        int, Query(ge=1, le=MAX_LOOKBACK_DAYS, description="Lookback window in days.")
    ] = DEFAULT_LOOKBACK_DAYS,
    min_severity: Annotated[
        Severity | None, Query(description="Drop anomalies below this severity.")
    ] = None,
    anomaly_type: Annotated[
        AnomalyType | None, Query(description="Restrict to one kind of anomaly.")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[AnomalyResponse]:
    """Return recent anomalies matching the filters."""
    end = datetime.now(UTC).date()
    start = end - timedelta(days=days)

    rows = await anomalies.list_recent(
        start=start,
        end=end,
        min_severity=min_severity,
        anomaly_type=anomaly_type,
        limit=limit,
    )
    return [AnomalyResponse.from_row(anomaly, symbol) for anomaly, symbol in rows]


@router.get(
    "/{symbol}",
    response_model=list[AnomalyResponse],
    summary="List one listing's anomalies",
    responses={status.HTTP_404_NOT_FOUND: {"description": "No such ticker."}},
)
async def list_symbol_anomalies(
    symbol: str,
    tickers: TickerRepoDep,
    anomalies: AnomalyRepoDep,
    start: Annotated[date | None, Query(description="Inclusive first session.")] = None,
    end: Annotated[date | None, Query(description="Inclusive last session.")] = None,
) -> list[AnomalyResponse]:
    """Return one listing's anomalies over a window.

    Raises:
        NotFoundError: If the symbol is not tracked.
    """
    ticker = await tickers.get_by_symbol(symbol)
    if ticker is None:
        msg = f"Ticker {symbol.upper()!r} is not tracked."
        raise NotFoundError(msg, details={"symbol": symbol.upper()})

    window_end = end or datetime.now(UTC).date()
    window_start = start or window_end - timedelta(days=DEFAULT_LOOKBACK_DAYS)

    rows = await anomalies.list_for_ticker(ticker.id, start=window_start, end=window_end)
    return [AnomalyResponse.from_row(anomaly, ticker.symbol) for anomaly in rows]


@router.post(
    "/detect",
    response_model=IngestionRunResponse,
    summary="Run anomaly detection across the universe",
    description=(
        "Runs the robust Z-score and Isolation Forest detectors over stored "
        "indicators. Safe to run repeatedly: results are upserted on "
        "(ticker, session, type, method)."
    ),
)
async def detect_anomalies(
    service: AnomalyServiceDep,
    lookback_sessions: Annotated[
        int | None,
        Query(ge=1, le=2000, description="Sessions to report on; the baseline is unaffected."),
    ] = None,
) -> IngestionRunResponse:
    """Run both detectors and return a summary of the run."""
    report = await service.detect_all(lookback_sessions=lookback_sessions)
    return IngestionRunResponse(
        started_at=report.started_at,
        finished_at=report.finished_at,
        duration_seconds=round(report.duration_seconds, 3),
        items_written=report.detections,
        failures=[f"{failure.symbol}: {failure.error}" for failure in report.failures],
    )
