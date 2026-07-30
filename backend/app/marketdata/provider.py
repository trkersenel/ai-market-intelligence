"""The market data provider contract.

One protocol, plus a capability declaration. Together they are what lets the
application switch vendors without the UI noticing.

The capability part matters more than it looks. A protocol alone forces every
adapter to implement every method, so a provider that has no bid/ask feed must
either raise or return something false. Neither is acceptable: raising turns a
missing feature into an error page, and returning zeros puts a fabricated spread
on screen. Instead an adapter *declares* what it supports, callers ask before
requesting, and an unsupported call raises a specific, catchable error that the
UI renders as "not available from this provider" rather than as a failure.

That distinction -- unsupported versus broken -- is the difference between a
degraded feature and an outage, and no free-tier provider covers this whole
surface. Building as if one did would mean the application only works on a plan
nobody is paying for.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import date
from enum import StrEnum
from typing import Protocol, runtime_checkable

from app.core.exceptions import AppError
from app.marketdata.domain import (
    AnalystRating,
    CandleSeries,
    CompanyProfile,
    Dividend,
    Earnings,
    Financials,
    InsiderTransaction,
    InstitutionalHolder,
    Interval,
    KeyMetrics,
    Listing,
    NewsItem,
    PriceTarget,
    Quote,
    TradeTick,
)


class Capability(StrEnum):
    """A discrete thing a provider can do.

    Granular on purpose. "Fundamentals" as a single flag would be useless: a
    provider commonly has income statements but not institutional holders, and a
    page that renders one section per capability needs to know which.
    """

    UNIVERSE = "universe"
    PROFILE = "profile"
    LOGO = "logo"
    QUOTE = "quote"
    BATCH_QUOTE = "batch_quote"
    CANDLES_DAILY = "candles_daily"
    CANDLES_INTRADAY = "candles_intraday"
    EXTENDED_HOURS = "extended_hours"
    BID_ASK = "bid_ask"
    FINANCIALS = "financials"
    METRICS = "metrics"
    ANALYST_RATINGS = "analyst_ratings"
    PRICE_TARGETS = "price_targets"
    INSIDER_TRANSACTIONS = "insider_transactions"
    INSTITUTIONAL_HOLDERS = "institutional_holders"
    DIVIDENDS = "dividends"
    EARNINGS = "earnings"
    NEWS = "news"
    STREAMING = "streaming"


class CapabilityNotSupportedError(AppError):
    """The provider cannot serve this data.

    A 501, deliberately, not a 500 or a 404. The request was well formed and the
    resource may well exist -- this provider simply does not offer it, and the
    client should render the section as unavailable rather than retrying or
    showing an error. Retrying a capability gap forever is the failure this
    exists to prevent.
    """

    status_code = 501
    code = "capability_not_supported"


class ProviderQuotaExceededError(AppError):
    """The provider's rate limit or daily quota is exhausted.

    Distinct from a generic rate-limit error because the remedy differs: a
    per-minute limit clears on its own and is worth retrying, while a daily
    quota does not and the caller should fall back to cache or to another
    provider instead of hammering a door that stays shut until midnight.
    """

    status_code = 429
    code = "provider_quota_exceeded"


@runtime_checkable
class MarketDataProvider(Protocol):
    """Everything the application may ask a market data vendor for.

    Adapters implement only what they support and declare the rest through
    :attr:`capabilities`. The default implementations raise
    :class:`CapabilityNotSupportedError`, so an adapter that forgets a method fails
    loudly and specifically rather than returning ``None`` into a chart.
    """

    @property
    def name(self) -> str:
        """Identifier recorded with cached data and shown in provenance."""
        ...

    @property
    def capabilities(self) -> frozenset[Capability]:
        """What this provider can serve."""
        ...

    def supports(self, capability: Capability) -> bool:
        """Whether this provider serves ``capability``."""
        ...

    # --- Universe and reference data --------------------------------------

    async def list_universe(self, exchange: str) -> Sequence[Listing]:
        """Return every tradable listing on an exchange."""
        ...

    async def get_profile(self, symbol: str) -> CompanyProfile:
        """Return descriptive facts about one issuer."""
        ...

    # --- Prices ------------------------------------------------------------

    async def get_quote(self, symbol: str) -> Quote:
        """Return the latest price snapshot for one symbol."""
        ...

    async def get_quotes(self, symbols: Sequence[str]) -> dict[str, Quote]:
        """Return snapshots for several symbols.

        Separate from :meth:`get_quote` because a provider that batches natively
        turns a dashboard's fifty requests into one, and one that does not needs
        the fan-out bounded rather than left to the caller.
        """
        ...

    async def get_candles(
        self,
        symbol: str,
        *,
        interval: Interval,
        start: date,
        end: date,
        adjusted: bool = True,
    ) -> CandleSeries:
        """Return OHLCV bars for one symbol over a window."""
        ...

    # --- Fundamentals ------------------------------------------------------

    async def get_financials(self, symbol: str) -> Financials:
        """Return income statement, balance sheet and cash flow."""
        ...

    async def get_metrics(self, symbol: str) -> KeyMetrics:
        """Return valuation, growth and profitability ratios."""
        ...

    # --- Sell-side and ownership -------------------------------------------

    async def get_analyst_ratings(self, symbol: str) -> Sequence[AnalystRating]:
        """Return aggregated recommendations over recent periods."""
        ...

    async def get_price_target(self, symbol: str) -> PriceTarget:
        """Return sell-side price targets."""
        ...

    async def get_insider_transactions(self, symbol: str) -> Sequence[InsiderTransaction]:
        """Return recently reported insider trades."""
        ...

    async def get_institutional_holders(self, symbol: str) -> Sequence[InstitutionalHolder]:
        """Return reported institutional positions."""
        ...

    # --- Events ------------------------------------------------------------

    async def get_dividends(self, symbol: str) -> Sequence[Dividend]:
        """Return dividend history."""
        ...

    async def get_earnings(self, symbol: str) -> Sequence[Earnings]:
        """Return reported and scheduled earnings."""
        ...

    async def get_news(self, symbol: str, *, start: date, end: date) -> Sequence[NewsItem]:
        """Return company news over a window."""
        ...

    # --- Streaming ---------------------------------------------------------

    def stream_trades(self, symbols: Sequence[str]) -> AsyncIterator[TradeTick]:
        """Yield trade prints as they arrive.

        An async iterator rather than a callback, so a consumer can apply
        backpressure by simply reading more slowly. A callback API would force
        the adapter to decide what happens when the consumer falls behind, and
        the only answers available to it are "drop" or "grow without bound".
        """
        ...

    async def aclose(self) -> None:
        """Release transports and open sockets."""
        ...


class BaseProvider:
    """Default implementations that refuse politely.

    An adapter inherits this and overrides what it supports. Everything left
    alone raises :class:`CapabilityNotSupportedError` naming the provider and the
    capability, which is what lets the API answer "Finnhub does not serve
    institutional holders" instead of a stack trace.
    """

    #: Overridden by each adapter.
    capabilities: frozenset[Capability] = frozenset()

    @property
    def name(self) -> str:
        """Identifier recorded with cached data and shown in provenance."""
        return type(self).__name__.replace("Provider", "").lower()

    def supports(self, capability: Capability) -> bool:
        """Whether this provider serves ``capability``."""
        return capability in self.capabilities

    def _unsupported(self, capability: Capability) -> CapabilityNotSupportedError:
        """Build the refusal for an unsupported capability."""
        msg = f"{self.name} does not provide {capability.value.replace('_', ' ')}."
        return CapabilityNotSupportedError(
            msg, details={"provider": self.name, "capability": capability.value}
        )

    async def list_universe(self, exchange: str) -> Sequence[Listing]:
        """Refuse: this provider serves no universe listing."""
        raise self._unsupported(Capability.UNIVERSE)

    async def get_profile(self, symbol: str) -> CompanyProfile:
        """Refuse: this provider serves no company profile."""
        raise self._unsupported(Capability.PROFILE)

    async def get_quote(self, symbol: str) -> Quote:
        """Refuse: this provider serves no quotes."""
        raise self._unsupported(Capability.QUOTE)

    async def get_quotes(self, symbols: Sequence[str]) -> dict[str, Quote]:
        """Fan out to :meth:`get_quote` when the provider has no batch endpoint.

        A correct default rather than a refusal: batching is an optimisation,
        and a provider without it can still answer -- just with more calls. The
        caller's rate limiter bounds the cost.
        """
        if not self.supports(Capability.QUOTE):
            raise self._unsupported(Capability.BATCH_QUOTE)
        return {symbol: await self.get_quote(symbol) for symbol in symbols}

    async def get_candles(
        self,
        symbol: str,
        *,
        interval: Interval,
        start: date,
        end: date,
        adjusted: bool = True,
    ) -> CandleSeries:
        """Refuse: this provider serves no candles."""
        raise self._unsupported(
            Capability.CANDLES_INTRADAY if interval.is_intraday else Capability.CANDLES_DAILY
        )

    async def get_financials(self, symbol: str) -> Financials:
        """Refuse: this provider serves no financial statements."""
        raise self._unsupported(Capability.FINANCIALS)

    async def get_metrics(self, symbol: str) -> KeyMetrics:
        """Refuse: this provider serves no ratio metrics."""
        raise self._unsupported(Capability.METRICS)

    async def get_analyst_ratings(self, symbol: str) -> Sequence[AnalystRating]:
        """Refuse: this provider serves no analyst ratings."""
        raise self._unsupported(Capability.ANALYST_RATINGS)

    async def get_price_target(self, symbol: str) -> PriceTarget:
        """Refuse: this provider serves no price targets."""
        raise self._unsupported(Capability.PRICE_TARGETS)

    async def get_insider_transactions(self, symbol: str) -> Sequence[InsiderTransaction]:
        """Refuse: this provider serves no insider transactions."""
        raise self._unsupported(Capability.INSIDER_TRANSACTIONS)

    async def get_institutional_holders(self, symbol: str) -> Sequence[InstitutionalHolder]:
        """Refuse: this provider serves no institutional holdings."""
        raise self._unsupported(Capability.INSTITUTIONAL_HOLDERS)

    async def get_dividends(self, symbol: str) -> Sequence[Dividend]:
        """Refuse: this provider serves no dividend history."""
        raise self._unsupported(Capability.DIVIDENDS)

    async def get_earnings(self, symbol: str) -> Sequence[Earnings]:
        """Refuse: this provider serves no earnings data."""
        raise self._unsupported(Capability.EARNINGS)

    async def get_news(self, symbol: str, *, start: date, end: date) -> Sequence[NewsItem]:
        """Refuse: this provider serves no news."""
        raise self._unsupported(Capability.NEWS)

    def stream_trades(self, symbols: Sequence[str]) -> AsyncIterator[TradeTick]:
        """Refuse: this provider has no streaming feed."""
        raise self._unsupported(Capability.STREAMING)

    async def aclose(self) -> None:
        """Release transports. Overridden by adapters that hold one."""
        return None
