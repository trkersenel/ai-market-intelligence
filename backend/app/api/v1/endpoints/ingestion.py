"""Manual ingestion triggers.

The scheduler runs these jobs on a cron; these endpoints exist for backfills,
for recovering a symbol that failed overnight, and for demonstrating the
pipeline without waiting for the next scheduled window.

Deliberately synchronous: the caller waits for the run and gets a report naming
every failure. Fire-and-forget would return 202 and hide exactly the information
someone triggering a manual backfill needs.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Annotated

from fastapi import APIRouter, Query, status

from app.api.deps import PriceIngestionDep
from app.schemas.market import IngestionRunResponse

router = APIRouter(tags=["ingestion"])


@router.post(
    "/prices",
    response_model=IngestionRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest prices for every active listing",
    description=(
        "Fetches each listing's missing sessions and upserts them. Safe to run "
        "repeatedly: windows overlap stored data and writes are idempotent."
    ),
)
async def ingest_prices(
    service: PriceIngestionDep,
    as_of: Annotated[
        date | None, Query(description="Treat this date as today, for backfills.")
    ] = None,
) -> IngestionRunResponse:
    """Run price ingestion across the tracked universe."""
    report = await service.ingest_all(as_of=as_of)
    return IngestionRunResponse(
        started_at=report.started_at,
        finished_at=report.finished_at,
        duration_seconds=round(report.duration_seconds, 3),
        items_written=report.bars_written,
        failures=[f"{failure.symbol}: {failure.error}" for failure in report.failures],
    )


@router.post(
    "/prices/{symbol}",
    response_model=IngestionRunResponse,
    summary="Ingest prices for one listing",
    responses={status.HTTP_404_NOT_FOUND: {"description": "No such ticker."}},
)
async def ingest_symbol_prices(
    symbol: str,
    service: PriceIngestionDep,
    as_of: Annotated[
        date | None, Query(description="Treat this date as today, for backfills.")
    ] = None,
) -> IngestionRunResponse:
    """Run price ingestion for a single symbol.

    Raises:
        NotFoundError: If the symbol is not tracked.
    """
    started = datetime.now(UTC)
    result = await service.ingest_symbol(symbol, as_of=as_of)
    finished = datetime.now(UTC)

    return IngestionRunResponse(
        started_at=started,
        finished_at=finished,
        duration_seconds=round((finished - started).total_seconds(), 3),
        items_written=result.bars_written,
        failures=[] if result.succeeded else [f"{result.symbol}: {result.error}"],
    )
