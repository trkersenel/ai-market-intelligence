"""Endpoints over the browsable exchange universe.

Every route here reads PostgreSQL, never the provider. That is the point: search
fires on each keystroke, and the free tier allows sixty requests a minute, so
proxying would exhaust the quota before a user finished typing "NVDA".
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status
from sqlalchemy import func, select

from app.api.deps import CurrentUserDep, ListingRepoDep, UniverseSyncDep
from app.core.exceptions import NotFoundError
from app.models.listing import Listing
from app.schemas.universe import ListingSummary, UniverseStats, UniverseSyncResult

router = APIRouter(tags=["universe"])


@router.get(
    "/search",
    response_model=list[ListingSummary],
    summary="Search every listing on the exchange",
    description=(
        "Matches a symbol or company name fragment. Exact symbol matches rank "
        "first, then symbol prefixes, then name matches."
    ),
)
async def search_universe(
    listings: ListingRepoDep,
    q: Annotated[
        str,
        Query(min_length=1, max_length=60, description="Symbol or name fragment."),
    ],
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> list[ListingSummary]:
    """Return listings matching the query, best first."""
    found = await listings.search(q, limit=limit)
    if not found:
        return []

    # One query for the tracked set rather than one per result. At twenty
    # results the difference is twenty round trips, and the set is tiny.
    tracked = await listings.tracked_symbols()
    return [
        ListingSummary.from_model(listing, tracked=listing.symbol in tracked) for listing in found
    ]


@router.get(
    "/stats",
    response_model=UniverseStats,
    summary="Size and freshness of the stored universe",
)
async def get_universe_stats(listings: ListingRepoDep) -> UniverseStats:
    """Return how many listings are stored and when they were last synced."""
    tracked = await listings.tracked_symbols()
    last_synced = await listings.session.scalar(select(func.max(Listing.synced_at)))
    return UniverseStats(
        listings=await listings.count(),
        tracked=len(tracked),
        last_synced_at=last_synced,
    )


@router.get(
    "/{symbol}",
    response_model=ListingSummary,
    summary="Get one listing",
    responses={status.HTTP_404_NOT_FOUND: {"description": "No such symbol."}},
)
async def get_listing(symbol: str, listings: ListingRepoDep) -> ListingSummary:
    """Return one listing.

    Raises:
        NotFoundError: If the symbol is not in the stored universe.
    """
    listing = await listings.get_by_symbol(symbol)
    if listing is None:
        msg = f"{symbol.upper()} is not in the stored universe."
        raise NotFoundError(msg, details={"symbol": symbol.upper()})

    tracked = await listings.tracked_symbols()
    return ListingSummary.from_model(listing, tracked=listing.symbol in tracked)


@router.post(
    "/sync",
    response_model=UniverseSyncResult,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Reconcile the stored universe with the provider",
    description=(
        "Normally runs on a daily schedule. Exposed for the first population of "
        "an empty database and for recovering from a failed scheduled run."
    ),
)
async def sync_universe(
    sync: UniverseSyncDep,
    _user: CurrentUserDep,
    exchange: Annotated[
        str | None,
        Query(max_length=20, description="MIC or exchange name. Defaults to configured."),
    ] = None,
) -> UniverseSyncResult:
    """Trigger a sync and report what it did.

    Authenticated because it spends provider quota, and a route that costs an
    API budget should not be anonymously triggerable.
    """
    report = await sync.sync(exchange)
    return UniverseSyncResult(
        fetched=report.fetched,
        written=report.written,
        deactivated=report.deactivated,
        succeeded=report.succeeded,
        error=report.error,
    )
