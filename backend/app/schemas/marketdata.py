"""Response schemas for the provider-backed market data endpoints.

Separate from :mod:`app.schemas.market`, which serialises the platform's *own*
stored tables. These serialise what a provider returned moments ago and never
touched PostgreSQL -- a distinction worth keeping in the type system, because
the two have different freshness, different failure modes, and only one of them
is guaranteed to exist for a given symbol.

Every optional field is optional because the free tier genuinely may not serve
it. They are ``None``, never ``0``: a UI that receives zero renders zero, and a
reader believes it.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from app.marketdata.domain import (
    AnalystRating,
    Candle,
    CandleSeries,
    CompanyProfile,
    Earnings,
    InsiderTransaction,
    KeyMetrics,
    Quote,
)


class QuoteResponse(BaseModel):
    """The latest price snapshot."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str
    timestamp: datetime
    #: Optional because the domain type is: a provider can answer with a symbol
    #: it knows and no last trade -- a halted issue, or one that has not traded
    #: today. Declaring it required here would turn that into a 500 on our side
    #: rather than a panel that says "no price".
    price: Decimal | None = None
    open: Decimal | None = None
    high: Decimal | None = None
    low: Decimal | None = None
    previous_close: Decimal | None = None
    volume: int | None = None
    average_volume: int | None = None
    week_52_high: Decimal | None = None
    week_52_low: Decimal | None = None
    session: str
    currency: str

    #: Computed rather than stored, because the two inputs come from the same
    #: response and deriving it in three different clients invites three
    #: different answers about what "change" means.
    change: Decimal | None = Field(default=None, description="Absolute move from previous close.")
    change_percent: float | None = Field(default=None, description="Percentage move.")

    @classmethod
    def from_domain(cls, quote: Quote) -> QuoteResponse:
        """Build from the provider-agnostic domain type."""
        change: Decimal | None = None
        change_percent: float | None = None
        previous = quote.previous_close
        if quote.price is not None and previous is not None and previous != 0:
            change = quote.price - previous
            change_percent = float(change / previous * 100)

        return cls(
            symbol=quote.symbol,
            timestamp=quote.timestamp,
            price=quote.price,
            open=quote.open,
            high=quote.high,
            low=quote.low,
            previous_close=quote.previous_close,
            volume=quote.volume,
            average_volume=quote.average_volume,
            week_52_high=quote.week_52_high,
            week_52_low=quote.week_52_low,
            session=quote.session.value,
            currency=quote.currency,
            change=change,
            change_percent=change_percent,
        )


class CandleResponse(BaseModel):
    """One OHLCV bar."""

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int

    @classmethod
    def from_domain(cls, candle: Candle) -> CandleResponse:
        """Build from the domain type."""
        return cls(
            timestamp=candle.timestamp,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
        )


class CandleSeriesResponse(BaseModel):
    """Bars over a window."""

    symbol: str
    interval: str
    adjusted: bool
    candles: list[CandleResponse]

    @classmethod
    def from_domain(cls, series: CandleSeries) -> CandleSeriesResponse:
        """Build from the domain type."""
        return cls(
            symbol=series.symbol,
            interval=series.interval.value,
            adjusted=series.adjusted,
            candles=[CandleResponse.from_domain(candle) for candle in series.candles],
        )


class ProfileResponse(BaseModel):
    """Descriptive facts about an issuer."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str
    name: str
    logo_url: str | None = None
    exchange: str | None = None
    industry: str | None = None
    sector: str | None = None
    country: str | None = None
    website: str | None = None
    ipo_date: date | None = None
    market_cap: Decimal | None = None
    shares_outstanding: Decimal | None = None
    description: str | None = None
    currency: str | None = None

    @classmethod
    def from_domain(cls, profile: CompanyProfile) -> ProfileResponse:
        """Build from the domain type."""
        return cls.model_validate(profile)


class MetricsResponse(BaseModel):
    """Valuation, profitability, growth and balance-sheet ratios."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str
    pe_ratio: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None
    price_to_book: float | None = None
    price_to_sales: float | None = None
    ev_to_ebitda: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    return_on_equity: float | None = None
    return_on_assets: float | None = None
    revenue_growth_yoy: float | None = None
    eps_growth_yoy: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    eps: float | None = None
    dividend_yield: float | None = None
    beta: float | None = None
    week_52_change: float | None = None

    @classmethod
    def from_domain(cls, metrics: KeyMetrics) -> MetricsResponse:
        """Build from the domain type."""
        return cls.model_validate(metrics)


class RatingResponse(BaseModel):
    """Aggregated sell-side recommendations for one period."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str
    period: date
    strong_buy: int
    buy: int
    hold: int
    sell: int
    strong_sell: int
    total: int
    consensus: str | None = None

    @classmethod
    def from_domain(cls, rating: AnalystRating) -> RatingResponse:
        """Build from the domain type, carrying its derived consensus."""
        return cls(
            symbol=rating.symbol,
            period=rating.period,
            strong_buy=rating.strong_buy,
            buy=rating.buy,
            hold=rating.hold,
            sell=rating.sell,
            strong_sell=rating.strong_sell,
            total=rating.total,
            consensus=rating.consensus,
        )


class InsiderTransactionResponse(BaseModel):
    """One reported insider trade."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str
    name: str
    transaction_date: date | None = None
    shares: int | None = None
    price: Decimal | None = None
    change: int | None = None
    transaction_code: str | None = None
    filing_date: date | None = None

    @classmethod
    def from_domain(cls, transaction: InsiderTransaction) -> InsiderTransactionResponse:
        """Build from the domain type."""
        return cls.model_validate(transaction)


class EarningsResponse(BaseModel):
    """One reported quarter against its estimate."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str
    fiscal_date: date
    eps_actual: Decimal | None = None
    eps_estimate: Decimal | None = None
    revenue_actual: Decimal | None = None
    revenue_estimate: Decimal | None = None
    report_date: date | None = None

    #: Derived here so every client agrees on the sign. A beat is positive.
    surprise_percent: float | None = None

    @classmethod
    def from_domain(cls, earnings: Earnings) -> EarningsResponse:
        """Build from the domain type, computing the EPS surprise."""
        surprise: float | None = None
        if (
            earnings.eps_actual is not None
            and earnings.eps_estimate is not None
            and earnings.eps_estimate != 0
        ):
            # abs() on the denominator: with a negative estimate, a smaller loss
            # than expected is a beat, and dividing by the signed value would
            # report it as a miss.
            surprise = float(
                (earnings.eps_actual - earnings.eps_estimate) / abs(earnings.eps_estimate) * 100
            )

        return cls(
            symbol=earnings.symbol,
            fiscal_date=earnings.fiscal_date,
            eps_actual=earnings.eps_actual,
            eps_estimate=earnings.eps_estimate,
            revenue_actual=earnings.revenue_actual,
            revenue_estimate=earnings.revenue_estimate,
            report_date=earnings.report_date,
            surprise_percent=surprise,
        )


class CapabilitiesResponse(BaseModel):
    """What each configured provider serves.

    Exposed so the frontend can decide what to render *before* requesting it. A
    page that knows charts are unavailable omits the section entirely rather
    than showing a panel that resolves to an error a moment later.
    """

    providers: dict[str, list[str]]
    capabilities: list[str]
