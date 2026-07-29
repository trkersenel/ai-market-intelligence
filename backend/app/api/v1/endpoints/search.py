"""Hybrid search endpoints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import IndexingServiceDep, SearchServiceDep
from app.models.enums import EcosystemTag
from app.schemas.market import IngestionRunResponse, SearchResponsePayload
from app.services.rag import SearchMode

router = APIRouter(tags=["search"])

MAX_LOOKBACK_DAYS = 365


@router.get(
    "",
    response_model=SearchResponsePayload,
    summary="Search the document corpus",
    description=(
        "Keyword, semantic, or both fused by reciprocal rank. Each result "
        "reports which retrievers found it and at what rank, so a citation "
        "can be audited after the fact."
    ),
)
async def search(
    service: SearchServiceDep,
    q: Annotated[str, Query(min_length=2, max_length=400, description="The query.")],
    mode: Annotated[SearchMode, Query(description="Which retrievers to run.")] = SearchMode.HYBRID,
    tickers: Annotated[
        list[str] | None, Query(description="Restrict to passages about these symbols.")
    ] = None,
    tags: Annotated[
        list[EcosystemTag] | None, Query(description="Restrict to these segments.")
    ] = None,
    days: Annotated[
        int | None, Query(ge=1, le=MAX_LOOKBACK_DAYS, description="Lookback window.")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> SearchResponsePayload:
    """Run a search and return the fused results."""
    since = datetime.now(UTC) - timedelta(days=days) if days else None

    response = await service.search(
        q,
        mode=mode,
        limit=limit,
        tickers=tickers,
        tags=[tag.value for tag in tags] if tags else None,
        since=since,
    )
    return SearchResponsePayload.from_response(response)


@router.post(
    "/index",
    response_model=IngestionRunResponse,
    summary="Embed pending documents",
    description=(
        "Chunks and embeds any recent article not yet indexed by the active "
        "embedding model. Safe to run repeatedly; a model change makes the "
        "whole corpus pending again."
    ),
)
async def index_documents(
    service: IndexingServiceDep,
    limit: Annotated[int | None, Query(ge=1, le=1000)] = None,
) -> IngestionRunResponse:
    """Run one indexing pass."""
    report = await service.index_pending(limit=limit)
    return IngestionRunResponse(
        started_at=report.started_at,
        finished_at=report.finished_at,
        duration_seconds=round(report.duration_seconds, 3),
        items_written=report.chunks,
        failures=[report.error] if report.error else [],
    )
