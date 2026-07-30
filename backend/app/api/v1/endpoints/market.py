"""Provider-backed market data endpoints.

Everything here reads through :class:`~app.marketdata.service.MarketDataService`
and therefore through its cache, so a page that renders six panels for one
symbol does not spend six times the quota -- and two users opening the same
symbol at once spend one request between them, not two.

These routes never touch PostgreSQL. That is the point of the split: the stored
tables hold what the platform analyses, and these hold what a provider says
right now about anything on the exchange.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Path, Query, status

from app.api.deps import MarketDataDep
from app.marketdata.domain import Interval
from app.schemas.marketdata import (
    CandleSeriesResponse,
    CapabilitiesResponse,
    EarningsResponse,
    InsiderTransactionResponse,
    MetricsResponse,
    ProfileResponse,
    QuoteResponse,
    RatingResponse,
)

router = APIRouter(tags=["market"])

SymbolPath = Annotated[str, Path(min_length=1, max_length=20, description="Exchange ticker.")]

#: Chart ranges offered to the UI, as (calendar days, bar interval).
#:
#: The interval steps down as the window widens because a chart cannot usefully
#: draw more points than it has pixels: five years of one-minute bars is half a
#: million points rendered into roughly a thousand columns, which costs a great
#: deal to produce and looks identical to the daily series.
_RANGES: dict[str, tuple[int, Interval]] = {
    "1D": (1, Interval.MINUTE_5),
    "5D": (5, Interval.MINUTE_30),
    "1M": (31, Interval.DAY_1),
    "3M": (93, Interval.DAY_1),
    "6M": (186, Interval.DAY_1),
    "1Y": (366, Interval.DAY_1),
    "5Y": (1827, Interval.WEEK_1),
    "MAX": (7305, Interval.MONTH_1),
}

RangeQuery = Annotated[
    str, Query(description="Chart window.", pattern="^(1D|5D|1M|3M|6M|1Y|5Y|MAX)$")
]

#: Documented on every provider-backed route. Both are ordinary outcomes on a
#: free tier rather than faults, and saying so in the schema lets a client
#: branch on them instead of treating every non-200 as breakage.
_UPSTREAM: dict[int | str, dict[str, Any]] = {
    status.HTTP_501_NOT_IMPLEMENTED: {"description": "No configured provider serves this."},
    status.HTTP_429_TOO_MANY_REQUESTS: {"description": "The provider's quota is spent."},
}


@router.get(
    "/capabilities",
    response_model=CapabilitiesResponse,
    summary="What the configured providers can serve",
)
async def get_capabilities(market: MarketDataDep) -> CapabilitiesResponse:
    """Return each provider's capabilities.

    Declared before ``/{symbol}/...`` so the literal path is matched first;
    otherwise "capabilities" would be read as a ticker.
    """
    providers = market.capabilities
    return CapabilitiesResponse(
        providers=providers,
        capabilities=sorted({name for values in providers.values() for name in values}),
    )


@router.get(
    "/{symbol}/quote",
    response_model=QuoteResponse,
    summary="Latest price snapshot",
    responses=_UPSTREAM,
)
async def get_quote(symbol: SymbolPath, market: MarketDataDep) -> QuoteResponse:
    """Return the current price with its move from the previous close."""
    return QuoteResponse.from_domain(await market.get_quote(symbol))


@router.get(
    "/{symbol}/candles",
    response_model=CandleSeriesResponse,
    summary="OHLCV bars over a window",
    responses=_UPSTREAM,
)
async def get_candles(
    symbol: SymbolPath,
    market: MarketDataDep,
    chart_range: RangeQuery = "1Y",
) -> CandleSeriesResponse:
    """Return bars for a named range.

    The range is a closed vocabulary rather than free start/end dates: it keeps
    the cache key small enough that everyone viewing "1Y" for a symbol shares
    one cached series, where arbitrary dates would give each visitor their own.
    """
    days, interval = _RANGES[chart_range]
    end = datetime.now(UTC).date()
    return CandleSeriesResponse.from_domain(
        await market.get_candles(
            symbol,
            interval=interval,
            start=end - timedelta(days=days),
            end=end,
        )
    )


@router.get(
    "/{symbol}/profile",
    response_model=ProfileResponse,
    summary="Company profile and logo",
    responses=_UPSTREAM,
)
async def get_profile(symbol: SymbolPath, market: MarketDataDep) -> ProfileResponse:
    """Return descriptive facts about the issuer."""
    return ProfileResponse.from_domain(await market.get_profile(symbol))


@router.get(
    "/{symbol}/metrics",
    response_model=MetricsResponse,
    summary="Valuation and profitability ratios",
    responses=_UPSTREAM,
)
async def get_metrics(symbol: SymbolPath, market: MarketDataDep) -> MetricsResponse:
    """Return the key ratios."""
    return MetricsResponse.from_domain(await market.get_metrics(symbol))


@router.get(
    "/{symbol}/ratings",
    response_model=list[RatingResponse],
    summary="Analyst recommendations by period",
    responses=_UPSTREAM,
)
async def get_ratings(symbol: SymbolPath, market: MarketDataDep) -> list[RatingResponse]:
    """Return aggregated sell-side ratings, most recent first."""
    ratings = await market.get_analyst_ratings(symbol)
    return [RatingResponse.from_domain(rating) for rating in ratings]


@router.get(
    "/{symbol}/insiders",
    response_model=list[InsiderTransactionResponse],
    summary="Reported insider transactions",
    responses=_UPSTREAM,
)
async def get_insiders(
    symbol: SymbolPath,
    market: MarketDataDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> list[InsiderTransactionResponse]:
    """Return recent insider trades, most recent first."""
    transactions = await market.get_insider_transactions(symbol)
    return [InsiderTransactionResponse.from_domain(item) for item in transactions[:limit]]


@router.get(
    "/{symbol}/earnings",
    response_model=list[EarningsResponse],
    summary="Reported earnings against estimates",
    responses=_UPSTREAM,
)
async def get_earnings(
    symbol: SymbolPath,
    market: MarketDataDep,
    limit: Annotated[int, Query(ge=1, le=40)] = 12,
) -> list[EarningsResponse]:
    """Return recent quarters with their EPS surprise."""
    quarters = await market.get_earnings(symbol)
    return [EarningsResponse.from_domain(quarter) for quarter in quarters[:limit]]


@router.get(
    "/{symbol}/news",
    summary="Company news from the provider",
    responses=_UPSTREAM,
)
async def get_provider_news(
    symbol: SymbolPath,
    market: MarketDataDep,
    days: Annotated[int, Query(ge=1, le=90)] = 14,
) -> list[dict[str, object]]:
    """Return provider news for one symbol.

    Distinct from ``/news``, which serves the platform's own ingested and
    sentiment-scored corpus. This one covers any symbol on the exchange,
    including the thousands nothing is stored for -- at the cost of carrying no
    sentiment, because nothing has scored it.
    """
    end = datetime.now(UTC).date()
    items = await market.get_news(symbol, start=end - timedelta(days=days), end=end)
    return [
        {
            "headline": item.headline,
            "url": item.url,
            "published_at": item.published_at.isoformat(),
            "source": item.source,
            "summary": item.summary,
            "image_url": item.image_url,
        }
        for item in items
    ]
