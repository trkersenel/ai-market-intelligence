"""Watchlist and portfolio endpoints.

Every route resolves its resource through the authenticated user, so ownership
is enforced by construction rather than by a check each handler must remember to
write. A resource belonging to another account returns 404, not 403 -- see
:mod:`app.services.portfolio_service` for why.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, status

from app.api.deps import CurrentUserDep, PortfolioServiceDep, WatchlistServiceDep
from app.schemas.auth import (
    PortfolioCreateRequest,
    PortfolioDetail,
    PortfolioSummary,
    PositionRequest,
    WatchlistCreateRequest,
    WatchlistDetail,
    WatchlistItemRequest,
    WatchlistSummary,
)

router = APIRouter(tags=["watchlists"])

#: Shared OpenAPI response documentation. Typed explicitly because FastAPI
#: keys responses by `int | str`, and an inferred `dict[int, ...]` fails to
#: unpack into that.
_NOT_FOUND: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"description": "Not found, or not yours."}
}
_UNAUTHORISED: dict[int | str, dict[str, Any]] = {
    status.HTTP_401_UNAUTHORIZED: {"description": "Missing or invalid token."}
}


@router.get(
    "",
    response_model=list[WatchlistSummary],
    summary="List your watchlists",
    responses={**_UNAUTHORISED},
)
async def list_watchlists(
    current_user: CurrentUserDep, service: WatchlistServiceDep
) -> list[WatchlistSummary]:
    """Return the authenticated user's watchlists, default first."""
    found = await service.list_for(current_user.id)
    return [WatchlistSummary.model_validate(item) for item in found]


@router.post(
    "",
    response_model=WatchlistDetail,
    status_code=status.HTTP_201_CREATED,
    summary="Create a watchlist",
    responses={
        **_UNAUTHORISED,
        status.HTTP_409_CONFLICT: {"description": "You already have a list with that name."},
    },
)
async def create_watchlist(
    payload: WatchlistCreateRequest,
    current_user: CurrentUserDep,
    service: WatchlistServiceDep,
) -> WatchlistDetail:
    """Create a watchlist, optionally pre-populated with symbols.

    Raises:
        ConflictError: If the name is already used by this user.
        NotFoundError: If any supplied symbol is not tracked.
    """
    watchlist = await service.create(
        current_user.id,
        name=payload.name,
        description=payload.description,
        symbols=payload.symbols,
    )
    return WatchlistDetail.from_model(watchlist)


@router.get(
    "/{watchlist_id}",
    response_model=WatchlistDetail,
    summary="Get one watchlist with its tickers",
    responses={**_UNAUTHORISED, **_NOT_FOUND},
)
async def get_watchlist(
    watchlist_id: uuid.UUID,
    current_user: CurrentUserDep,
    service: WatchlistServiceDep,
) -> WatchlistDetail:
    """Return one of the user's watchlists.

    Raises:
        NotFoundError: If it does not exist or belongs to another account.
    """
    watchlist = await service.get_owned(current_user.id, watchlist_id)
    return WatchlistDetail.from_model(watchlist)


@router.delete(
    "/{watchlist_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a watchlist",
    responses={**_UNAUTHORISED, **_NOT_FOUND},
)
async def delete_watchlist(
    watchlist_id: uuid.UUID,
    current_user: CurrentUserDep,
    service: WatchlistServiceDep,
) -> None:
    """Delete one of the user's watchlists and its membership rows.

    Raises:
        NotFoundError: If it does not exist or belongs to another account.
    """
    await service.delete(current_user.id, watchlist_id)


@router.post(
    "/{watchlist_id}/tickers",
    response_model=WatchlistDetail,
    summary="Add a ticker to a watchlist",
    description="Adding a symbol already on the list is a no-op, not an error.",
    responses={**_UNAUTHORISED, **_NOT_FOUND},
)
async def add_ticker(
    watchlist_id: uuid.UUID,
    payload: WatchlistItemRequest,
    current_user: CurrentUserDep,
    service: WatchlistServiceDep,
) -> WatchlistDetail:
    """Add a symbol to one of the user's watchlists.

    Raises:
        NotFoundError: If the watchlist is not the user's, or the symbol is not
            tracked.
    """
    watchlist = await service.add_symbol(
        current_user.id, watchlist_id, symbol=payload.symbol, note=payload.note
    )
    return WatchlistDetail.from_model(watchlist)


@router.delete(
    "/{watchlist_id}/tickers/{symbol}",
    response_model=WatchlistDetail,
    summary="Remove a ticker from a watchlist",
    responses={**_UNAUTHORISED, **_NOT_FOUND},
)
async def remove_ticker(
    watchlist_id: uuid.UUID,
    symbol: str,
    current_user: CurrentUserDep,
    service: WatchlistServiceDep,
) -> WatchlistDetail:
    """Remove a symbol from one of the user's watchlists.

    Raises:
        NotFoundError: If the watchlist is not the user's, or the symbol is not
            tracked.
    """
    watchlist = await service.remove_symbol(current_user.id, watchlist_id, symbol=symbol)
    return WatchlistDetail.from_model(watchlist)


# --- Portfolios ------------------------------------------------------------

portfolios_router = APIRouter(tags=["portfolios"])


@portfolios_router.get(
    "",
    response_model=list[PortfolioSummary],
    summary="List your portfolios",
    responses={**_UNAUTHORISED},
)
async def list_portfolios(
    current_user: CurrentUserDep, service: PortfolioServiceDep
) -> list[PortfolioSummary]:
    """Return the authenticated user's portfolios."""
    found = await service.list_for(current_user.id)
    return [PortfolioSummary.model_validate(item) for item in found]


@portfolios_router.post(
    "",
    response_model=PortfolioDetail,
    status_code=status.HTTP_201_CREATED,
    summary="Create a portfolio",
    responses={
        **_UNAUTHORISED,
        status.HTTP_409_CONFLICT: {"description": "You already have one with that name."},
    },
)
async def create_portfolio(
    payload: PortfolioCreateRequest,
    current_user: CurrentUserDep,
    service: PortfolioServiceDep,
) -> PortfolioDetail:
    """Create an empty portfolio.

    Raises:
        ConflictError: If the name is already used by this user.
    """
    portfolio = await service.create(
        current_user.id,
        name=payload.name,
        description=payload.description,
        base_currency=payload.base_currency,
    )
    return PortfolioDetail.from_model(portfolio, {})


@portfolios_router.get(
    "/{portfolio_id}",
    response_model=PortfolioDetail,
    summary="Get one portfolio, valued at the last completed close",
    description=(
        "Positions are valued at the most recent *completed* session. A "
        "still-trading price would make the same holdings worth different "
        "amounts depending on when the page was loaded."
    ),
    responses={**_UNAUTHORISED, **_NOT_FOUND},
)
async def get_portfolio(
    portfolio_id: uuid.UUID,
    current_user: CurrentUserDep,
    service: PortfolioServiceDep,
) -> PortfolioDetail:
    """Return one of the user's portfolios with valuations.

    Raises:
        NotFoundError: If it does not exist or belongs to another account.
    """
    portfolio = await service.get_owned(current_user.id, portfolio_id)
    prices = await service.latest_prices(portfolio)
    return PortfolioDetail.from_model(portfolio, prices)


@portfolios_router.delete(
    "/{portfolio_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a portfolio",
    responses={**_UNAUTHORISED, **_NOT_FOUND},
)
async def delete_portfolio(
    portfolio_id: uuid.UUID,
    current_user: CurrentUserDep,
    service: PortfolioServiceDep,
) -> None:
    """Delete one of the user's portfolios and its positions.

    Raises:
        NotFoundError: If it does not exist or belongs to another account.
    """
    await service.delete(current_user.id, portfolio_id)


@portfolios_router.put(
    "/{portfolio_id}/positions",
    response_model=PortfolioDetail,
    summary="Add or replace a holding",
    description=(
        "Upsert, not insert: a portfolio holds one line per instrument, so "
        "sending the same symbol twice updates the position rather than "
        "creating an ambiguous second row."
    ),
    responses={**_UNAUTHORISED, **_NOT_FOUND},
)
async def upsert_position(
    portfolio_id: uuid.UUID,
    payload: PositionRequest,
    current_user: CurrentUserDep,
    service: PortfolioServiceDep,
) -> PortfolioDetail:
    """Add a holding, or replace it if already held.

    Raises:
        NotFoundError: If the portfolio is not the user's, or the symbol is not
            tracked.
    """
    portfolio = await service.upsert_position(
        current_user.id,
        portfolio_id,
        symbol=payload.symbol,
        quantity=payload.quantity,
        average_cost=payload.average_cost,
        opened_at=payload.opened_at,
        note=payload.note,
    )
    prices = await service.latest_prices(portfolio)
    return PortfolioDetail.from_model(portfolio, prices)


@portfolios_router.delete(
    "/{portfolio_id}/positions/{symbol}",
    response_model=PortfolioDetail,
    summary="Remove a holding",
    responses={**_UNAUTHORISED, **_NOT_FOUND},
)
async def remove_position(
    portfolio_id: uuid.UUID,
    symbol: str,
    current_user: CurrentUserDep,
    service: PortfolioServiceDep,
) -> PortfolioDetail:
    """Remove a holding from one of the user's portfolios.

    Raises:
        NotFoundError: If the portfolio is not the user's, the symbol is not
            tracked, or it is not held.
    """
    portfolio = await service.remove_position(current_user.id, portfolio_id, symbol=symbol)
    prices = await service.latest_prices(portfolio)
    return PortfolioDetail.from_model(portfolio, prices)
