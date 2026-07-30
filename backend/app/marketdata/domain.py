"""Provider-agnostic market data types.

These are the platform's own vocabulary. No field here is named after a vendor's
JSON, and no vendor's quirks leak through: Finnhub returns ``c``/``h``/``l``/``o``
for a quote, Twelve Data returns ``close``/``high``, Polygon returns yet another
shape, and every one of them becomes a :class:`Quote`.

That is the whole mechanism behind "swap the provider without touching the UI".
The UI depends on these classes; an adapter's only job is to produce them. A new
provider means one new adapter file and a config change -- never a change to a
component, a hook, or a database column.

Two conventions run through the module:

**Money is ``Decimal``.** A price that round-trips through an IEEE double is no
longer the price the exchange printed, and the errors compound through returns
and portfolio valuations. Only ratios and statistics -- which are approximate by
nature -- are floats.

**Absent is ``None``, never zero.** A missing bid is not a bid of zero, and an
unknown market cap is not a market cap of nothing. Providers disagree about
which fields they populate, and collapsing "unknown" into a real value is how a
screener ends up ranking companies by which vendor happened to have the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum


class Interval(StrEnum):
    """Candle resolution, in the platform's own vocabulary.

    Providers spell these differently -- ``1``/``5``/``D`` for Finnhub,
    ``1min``/``1day`` for Twelve Data -- and each adapter maps to its own.
    """

    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1mo"

    @property
    def is_intraday(self) -> bool:
        """Whether the resolution is finer than one session.

        Determines cache lifetime and whether the extended-hours session is
        meaningful, so it is worth asking once here rather than at every call
        site.
        """
        return self in {
            Interval.MINUTE_1,
            Interval.MINUTE_5,
            Interval.MINUTE_15,
            Interval.MINUTE_30,
            Interval.HOUR_1,
        }


class MarketSession(StrEnum):
    """Which trading session a price belongs to."""

    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"


class StatementPeriod(StrEnum):
    """Reporting period of a financial statement."""

    QUARTERLY = "quarterly"
    ANNUAL = "annual"


@dataclass(frozen=True, slots=True)
class Listing:
    """One tradable security in the tracked universe.

    Deliberately thin. The universe is thousands of rows synced on a schedule,
    and everything expensive -- profile, logo, fundamentals -- is fetched per
    symbol on demand and cached separately.
    """

    symbol: str
    name: str
    exchange: str
    currency: str = "USD"
    #: Common stock, ETF, ADR. Providers disagree on the vocabulary, so adapters
    #: normalise to a small set and pass anything unrecognised through verbatim
    #: rather than guessing.
    security_type: str = "common"
    #: Provider-specific identifier, kept when one exists. FIGI where available,
    #: because it survives ticker changes and CUSIP is licensed.
    figi: str | None = None


@dataclass(frozen=True, slots=True)
class CompanyProfile:
    """Descriptive and structural facts about an issuer."""

    symbol: str
    name: str
    #: Absolute URL to the official logo. Never a data URI and never a
    #: placeholder: the frontend decides what to render when this is None, so a
    #: missing logo is visibly missing rather than silently wrong.
    logo_url: str | None = None
    exchange: str | None = None
    industry: str | None = None
    sector: str | None = None
    country: str | None = None
    headquarters: str | None = None
    website: str | None = None
    ceo: str | None = None
    ipo_date: date | None = None
    employees: int | None = None
    market_cap: Decimal | None = None
    enterprise_value: Decimal | None = None
    shares_outstanding: Decimal | None = None
    description: str | None = None
    currency: str = "USD"
    phone: str | None = None

    @property
    def has_logo(self) -> bool:
        """Whether a logo URL is available to render."""
        return bool(self.logo_url)


@dataclass(frozen=True, slots=True)
class Quote:
    """A point-in-time price snapshot.

    Every field beyond ``symbol`` and ``timestamp`` is optional because
    providers differ sharply in what they populate. Bid and ask in particular
    require a quote feed rather than a trade feed, and most free tiers have only
    the latter -- so the UI must be built to render a quote that has no spread,
    not to assume one.
    """

    symbol: str
    timestamp: datetime
    price: Decimal | None = None
    open: Decimal | None = None
    high: Decimal | None = None
    low: Decimal | None = None
    previous_close: Decimal | None = None
    volume: int | None = None
    average_volume: int | None = None
    bid: Decimal | None = None
    ask: Decimal | None = None
    bid_size: int | None = None
    ask_size: int | None = None
    vwap: Decimal | None = None
    week_52_high: Decimal | None = None
    week_52_low: Decimal | None = None
    #: Extended-hours prints, when the provider separates them. A pre-market
    #: price is not the same thing as an early regular-session price, and
    #: merging them silently misstates the day's open.
    pre_market_price: Decimal | None = None
    post_market_price: Decimal | None = None
    session: MarketSession = MarketSession.REGULAR
    currency: str = "USD"

    @property
    def change(self) -> Decimal | None:
        """Absolute change against the previous close."""
        if self.price is None or self.previous_close is None:
            return None
        return self.price - self.previous_close

    @property
    def change_percent(self) -> Decimal | None:
        """Percentage change against the previous close.

        ``None`` rather than zero when the previous close is missing or zero:
        a stock whose prior close is unknown has an unknown change, and showing
        0.00% would state something false.
        """
        if self.price is None or not self.previous_close:
            return None
        return (self.price - self.previous_close) / self.previous_close * 100

    @property
    def spread(self) -> Decimal | None:
        """Bid-ask spread, when both sides are known."""
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid


@dataclass(frozen=True, slots=True)
class Candle:
    """One OHLCV bar."""

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    #: Present only where the provider computes it; never derived here, because
    #: a VWAP reconstructed from a single bar is just the typical price wearing
    #: a more authoritative name.
    vwap: Decimal | None = None


@dataclass(frozen=True, slots=True)
class CandleSeries:
    """An ordered run of bars for one symbol at one resolution."""

    symbol: str
    interval: Interval
    candles: tuple[Candle, ...]
    #: Whether prices are adjusted for splits and dividends. Carried explicitly
    #: because mixing adjusted and raw series in one chart draws a cliff at
    #: every corporate action, and the two are indistinguishable by inspection.
    adjusted: bool = True

    @property
    def is_empty(self) -> bool:
        """Whether the series has no bars."""
        return not self.candles


@dataclass(frozen=True, slots=True)
class FinancialStatement:
    """One reporting period of one statement.

    Line items are a mapping rather than named fields on purpose. Statements
    differ by industry -- a bank has no cost of goods sold, an insurer has no
    inventory -- and a fixed schema would either be mostly null or would quietly
    drop the lines that matter for the company being viewed.
    """

    symbol: str
    period: StatementPeriod
    fiscal_date: date
    fiscal_year: int
    fiscal_quarter: int | None = None
    currency: str = "USD"
    line_items: dict[str, Decimal | None] = field(default_factory=dict)
    filing_date: date | None = None
    #: SEC filing URL where the provider supplies one, so a figure on screen can
    #: be traced to the document it came from.
    source_url: str | None = None

    def get(self, item: str) -> Decimal | None:
        """Return one line item, or ``None`` if this filing does not report it."""
        return self.line_items.get(item)


@dataclass(frozen=True, slots=True)
class Financials:
    """The three statements for one company, across periods."""

    symbol: str
    income_statements: tuple[FinancialStatement, ...] = ()
    balance_sheets: tuple[FinancialStatement, ...] = ()
    cash_flows: tuple[FinancialStatement, ...] = ()


@dataclass(frozen=True, slots=True)
class KeyMetrics:
    """Valuation, growth and profitability ratios.

    Floats, unlike prices: these are ratios and estimates, and implying exact
    decimal precision on a forward P/E would be false confidence.
    """

    symbol: str
    as_of: date | None = None

    # Valuation
    pe_ratio: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None
    price_to_book: float | None = None
    price_to_sales: float | None = None
    ev_to_ebitda: float | None = None
    ev_to_revenue: float | None = None

    # Profitability
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    return_on_equity: float | None = None
    return_on_assets: float | None = None
    return_on_invested_capital: float | None = None

    # Growth
    revenue_growth_yoy: float | None = None
    earnings_growth_yoy: float | None = None
    revenue_growth_3y: float | None = None
    eps_growth_yoy: float | None = None

    # Financial health
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    quick_ratio: float | None = None
    interest_coverage: float | None = None

    # Per share and yield
    eps: float | None = None
    book_value_per_share: float | None = None
    dividend_yield: float | None = None
    payout_ratio: float | None = None
    beta: float | None = None

    # Performance
    week_52_change: float | None = None


#: Consensus bands on the sell-side's own 1-5 scale, where 1 is strong buy.
#: The scale is the industry's, not ours, so the label matches what an
#: analyst reading the page expects the number to mean.
_CONSENSUS_BANDS: tuple[tuple[float, str], ...] = (
    (1.5, "Strong Buy"),
    (2.5, "Buy"),
    (3.5, "Hold"),
    (4.5, "Sell"),
)


@dataclass(frozen=True, slots=True)
class AnalystRating:
    """Aggregated sell-side recommendation for one period."""

    symbol: str
    period: date
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0

    @property
    def total(self) -> int:
        """Number of analysts covering the name."""
        return self.strong_buy + self.buy + self.hold + self.sell + self.strong_sell

    @property
    def consensus(self) -> str | None:
        """A one-word consensus, or ``None`` when nobody covers the name.

        Weighted 1 (strong buy) to 5 (strong sell), which is the convention the
        sell-side itself uses, so the label matches what an analyst expects.
        """
        if self.total == 0:
            return None
        weighted = (
            self.strong_buy * 1
            + self.buy * 2
            + self.hold * 3
            + self.sell * 4
            + self.strong_sell * 5
        ) / self.total
        for threshold, label in _CONSENSUS_BANDS:
            if weighted <= threshold:
                return label
        return "Strong Sell"


@dataclass(frozen=True, slots=True)
class PriceTarget:
    """Sell-side price targets."""

    symbol: str
    high: Decimal | None = None
    low: Decimal | None = None
    median: Decimal | None = None
    mean: Decimal | None = None
    analyst_count: int | None = None
    updated_at: date | None = None


@dataclass(frozen=True, slots=True)
class InsiderTransaction:
    """One reported insider trade."""

    symbol: str
    name: str
    transaction_date: date
    shares: Decimal
    price: Decimal | None = None
    #: Positive for acquisitions, negative for disposals. Providers encode this
    #: with letter codes that differ between them, so adapters normalise to a
    #: sign and keep the raw code for display.
    change: Decimal | None = None
    transaction_code: str | None = None
    filing_date: date | None = None

    @property
    def is_purchase(self) -> bool:
        """Whether the insider increased their position."""
        return self.change is not None and self.change > 0


@dataclass(frozen=True, slots=True)
class InstitutionalHolder:
    """One institution's reported position."""

    symbol: str
    holder: str
    shares: Decimal
    value: Decimal | None = None
    percent_of_shares: float | None = None
    report_date: date | None = None
    change: Decimal | None = None


@dataclass(frozen=True, slots=True)
class Dividend:
    """One dividend payment."""

    symbol: str
    ex_date: date
    amount: Decimal
    payment_date: date | None = None
    record_date: date | None = None
    declaration_date: date | None = None
    currency: str = "USD"


@dataclass(frozen=True, slots=True)
class Earnings:
    """One reported or scheduled earnings event."""

    symbol: str
    fiscal_date: date
    eps_actual: Decimal | None = None
    eps_estimate: Decimal | None = None
    revenue_actual: Decimal | None = None
    revenue_estimate: Decimal | None = None
    report_date: date | None = None
    #: Before open / after close, where reported. It changes which session a
    #: move belongs to, which is exactly what the correlation engine needs.
    timing: str | None = None

    @property
    def eps_surprise_percent(self) -> float | None:
        """Percentage by which actual EPS beat or missed the estimate."""
        if self.eps_actual is None or not self.eps_estimate:
            return None
        return float((self.eps_actual - self.eps_estimate) / abs(self.eps_estimate) * 100)


@dataclass(frozen=True, slots=True)
class NewsItem:
    """One news article from a market data provider."""

    headline: str
    url: str
    published_at: datetime
    source: str | None = None
    summary: str | None = None
    image_url: str | None = None
    symbols: tuple[str, ...] = ()
    category: str | None = None
    provider_id: str | None = None


@dataclass(frozen=True, slots=True)
class TradeTick:
    """A single trade print from a streaming feed."""

    symbol: str
    price: Decimal
    volume: int
    timestamp: datetime
    #: Exchange condition codes, passed through unmodified. They determine
    #: whether a print is eligible for the last price and for VWAP, and
    #: discarding them would make an odd-lot or out-of-sequence trade
    #: indistinguishable from a real one.
    conditions: tuple[str, ...] = ()
