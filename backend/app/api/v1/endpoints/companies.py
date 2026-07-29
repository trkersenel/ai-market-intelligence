"""Company and ticker endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status

from app.api.deps import CompanyRepoDep, TickerRepoDep
from app.core.exceptions import NotFoundError
from app.models.enums import AssetType, EcosystemTag
from app.schemas.market import CompanyDetail, CompanySummary, TickerSummary

router = APIRouter(tags=["companies"])


@router.get(
    "",
    response_model=list[CompanySummary],
    summary="List tracked companies",
    description="Optionally filtered by ecosystem segment or free-text search.",
)
async def list_companies(
    companies: CompanyRepoDep,
    tags: Annotated[
        list[EcosystemTag] | None,
        Query(description="Return companies exposed to any of these segments."),
    ] = None,
    match_all: Annotated[
        bool, Query(description="Require every tag rather than any of them.")
    ] = False,
    search: Annotated[
        str | None, Query(min_length=2, max_length=80, description="Name or slug substring.")
    ] = None,
) -> list[CompanySummary]:
    """Return the tracked universe, narrowed by the supplied filters."""
    if search:
        found = await companies.search(search)
    elif tags:
        found = await companies.list_by_tags(tags, match_all=match_all)
    else:
        found = await companies.list_tracked()

    return [CompanySummary.model_validate(company) for company in found]


@router.get(
    "/{slug}",
    response_model=CompanyDetail,
    summary="Get one company with its listings",
    responses={status.HTTP_404_NOT_FOUND: {"description": "No such company."}},
)
async def get_company(slug: str, companies: CompanyRepoDep) -> CompanyDetail:
    """Return one company and every listing of it.

    Raises:
        NotFoundError: If no company has this slug.
    """
    company = await companies.get_by_slug(slug)
    if company is None:
        msg = f"Company {slug!r} is not tracked."
        raise NotFoundError(msg, details={"slug": slug})

    # Re-read with the relationship loaded: `tickers` is `lazy="raise_on_sql"`,
    # so serialising the instance above would raise rather than silently N+1.
    detailed = await companies.get_with_tickers(company.id)
    if detailed is None:  # pragma: no cover - impossible within one transaction
        msg = f"Company {slug!r} disappeared mid-request."
        raise NotFoundError(msg)
    return CompanyDetail.from_model(detailed)


@router.get(
    "/{slug}/tickers",
    response_model=list[TickerSummary],
    summary="List a company's listings",
)
async def list_company_tickers(slug: str, companies: CompanyRepoDep) -> list[TickerSummary]:
    """Return every tradable listing of one company.

    Raises:
        NotFoundError: If no company has this slug.
    """
    company = await companies.get_by_slug(slug)
    if company is None:
        msg = f"Company {slug!r} is not tracked."
        raise NotFoundError(msg, details={"slug": slug})

    detailed = await companies.get_with_tickers(company.id)
    assert detailed is not None  # noqa: S101 - same transaction as the read above
    return [TickerSummary.model_validate(ticker) for ticker in detailed.tickers]


tickers_router = APIRouter(tags=["tickers"])


@tickers_router.get(
    "",
    response_model=list[TickerSummary],
    summary="List tradable listings",
)
async def list_tickers(
    tickers: TickerRepoDep,
    asset_type: Annotated[
        AssetType | None, Query(description="Restrict to equities, ETFs or indices.")
    ] = None,
) -> list[TickerSummary]:
    """Return every active listing the platform ingests."""
    found = await tickers.list_active(asset_type=asset_type)
    return [TickerSummary.model_validate(ticker) for ticker in found]


@tickers_router.get(
    "/{symbol}",
    response_model=TickerSummary,
    summary="Get one listing",
    responses={status.HTTP_404_NOT_FOUND: {"description": "No such ticker."}},
)
async def get_ticker(symbol: str, tickers: TickerRepoDep) -> TickerSummary:
    """Return one listing by symbol, case-insensitively.

    Raises:
        NotFoundError: If the symbol is not tracked.
    """
    ticker = await tickers.get_by_symbol(symbol)
    if ticker is None:
        msg = f"Ticker {symbol.upper()!r} is not tracked."
        raise NotFoundError(msg, details={"symbol": symbol.upper()})
    return TickerSummary.model_validate(ticker)
