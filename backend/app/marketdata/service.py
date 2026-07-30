"""The market data facade.

Everything above this line -- endpoints, jobs, the AI report generator -- talks
only to this class. It never learns which vendor answered, whether the value came
from cache, or that a capability is missing until it asks.

Three responsibilities, deliberately kept together because they are one decision
each time: resolve the capability to a provider, apply the right cache lifetime
for that kind of data, and coalesce concurrent misses. Splitting them would mean
every call site repeats the same three steps and eventually one of them forgets
the TTL.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from app.core.config import MarketDataSettings
from app.core.logging import get_logger
from app.marketdata.cache import ResponseCache, cache_key
from app.marketdata.domain import (
    AnalystRating,
    CandleSeries,
    CompanyProfile,
    Earnings,
    Financials,
    InsiderTransaction,
    Interval,
    KeyMetrics,
    Listing,
    NewsItem,
    Quote,
)
from app.marketdata.provider import Capability
from app.marketdata.registry import ProviderRegistry

logger = get_logger(__name__)


class MarketDataService:
    """Cached, provider-agnostic access to market data."""

    def __init__(
        self,
        *,
        registry: ProviderRegistry,
        cache: ResponseCache,
        settings: MarketDataSettings,
    ) -> None:
        """Wire the facade to its collaborators."""
        self._registry = registry
        self._cache = cache
        self._settings = settings

    @property
    def capabilities(self) -> dict[str, list[str]]:
        """What each configured provider serves."""
        return self._registry.describe()

    def supports(self, capability: Capability) -> bool:
        """Whether any provider serves ``capability``."""
        return self._registry.supports(capability)

    # --- Universe ----------------------------------------------------------

    async def list_universe(self, exchange: str | None = None) -> Sequence[Listing]:
        """Return every listing on an exchange.

        Cached for a day: the NASDAQ constituent list changes with listings and
        delistings, not by the minute, and the response is several megabytes.
        """
        wanted = exchange or self._settings.universe_exchange
        provider = self._registry.resolve(Capability.UNIVERSE)
        return await self._cache.get_or_fetch(
            cache_key("universe", provider.name, wanted),
            self._settings.profile_ttl_seconds,
            lambda: provider.list_universe(wanted),
        )

    # --- Company -----------------------------------------------------------

    async def get_profile(self, symbol: str) -> CompanyProfile:
        """Return descriptive facts about one issuer."""
        provider = self._registry.resolve(Capability.PROFILE)
        upper = symbol.upper()
        return await self._cache.get_or_fetch(
            cache_key("profile", provider.name, upper),
            self._settings.profile_ttl_seconds,
            lambda: provider.get_profile(upper),
        )

    async def get_quote(self, symbol: str) -> Quote:
        """Return the latest price snapshot.

        The shortest TTL in the platform. Fifteen seconds is long enough that a
        dashboard refreshing every few seconds costs one upstream call rather
        than dozens, and short enough that nobody is looking at a stale price.
        """
        provider = self._registry.resolve(Capability.QUOTE)
        upper = symbol.upper()
        return await self._cache.get_or_fetch(
            cache_key("quote", provider.name, upper),
            self._settings.quote_ttl_seconds,
            lambda: provider.get_quote(upper),
        )

    async def get_quotes(self, symbols: Sequence[str]) -> dict[str, Quote]:
        """Return snapshots for several symbols.

        Routed through :meth:`get_quote` per symbol rather than the provider's
        batch call, precisely so each one is cached and coalesced individually.
        A batch endpoint would refetch the forty-nine symbols already cached in
        order to get the fiftieth -- and a dashboard's tiles overlap heavily
        between users, which is exactly where the per-symbol cache pays.

        A symbol the provider cannot price is omitted rather than raising: one
        delisted ticker must not blank an entire watchlist.
        """
        quotes: dict[str, Quote] = {}
        for symbol in symbols:
            try:
                quotes[symbol.upper()] = await self.get_quote(symbol)
            except Exception as exc:  # noqa: BLE001 - one bad symbol is not a failure
                logger.debug("quote_unavailable", symbol=symbol.upper(), error=str(exc))
        return quotes

    async def get_metrics(self, symbol: str) -> KeyMetrics:
        """Return valuation, growth and profitability ratios."""
        provider = self._registry.resolve(Capability.METRICS)
        upper = symbol.upper()
        return await self._cache.get_or_fetch(
            cache_key("metrics", provider.name, upper),
            self._settings.metrics_ttl_seconds,
            lambda: provider.get_metrics(upper),
        )

    async def get_financials(self, symbol: str) -> Financials:
        """Return income statement, balance sheet and cash flow.

        Cached for a day. Statements change when a company files, which is four
        times a year -- and the payload is the largest the platform fetches.
        """
        provider = self._registry.resolve(Capability.FINANCIALS)
        upper = symbol.upper()
        return await self._cache.get_or_fetch(
            cache_key("financials", provider.name, upper),
            self._settings.financials_ttl_seconds,
            lambda: provider.get_financials(upper),
        )

    async def get_candles(
        self,
        symbol: str,
        *,
        interval: Interval = Interval.DAY_1,
        start: date,
        end: date,
        adjusted: bool = True,
    ) -> CandleSeries:
        """Return OHLCV bars over a window.

        The cache key carries the interval, window and adjustment flag. Keying on
        the symbol alone would serve a one-minute intraday series to a caller
        asking for five years of daily bars -- a bug that presents as a
        mysteriously wrong chart rather than as an error.
        """
        capability = (
            Capability.CANDLES_INTRADAY if interval.is_intraday else Capability.CANDLES_DAILY
        )
        provider = self._registry.resolve(capability)
        upper = symbol.upper()
        return await self._cache.get_or_fetch(
            cache_key("candles", provider.name, upper, interval.value, start, end, adjusted),
            self._settings.candles_ttl_seconds,
            lambda: provider.get_candles(
                upper, interval=interval, start=start, end=end, adjusted=adjusted
            ),
        )

    # --- Sell-side, ownership, events --------------------------------------

    async def get_analyst_ratings(self, symbol: str) -> Sequence[AnalystRating]:
        """Return aggregated sell-side recommendations."""
        provider = self._registry.resolve(Capability.ANALYST_RATINGS)
        upper = symbol.upper()
        return await self._cache.get_or_fetch(
            cache_key("ratings", provider.name, upper),
            self._settings.metrics_ttl_seconds,
            lambda: provider.get_analyst_ratings(upper),
        )

    async def get_insider_transactions(self, symbol: str) -> Sequence[InsiderTransaction]:
        """Return recently reported insider trades."""
        provider = self._registry.resolve(Capability.INSIDER_TRANSACTIONS)
        upper = symbol.upper()
        return await self._cache.get_or_fetch(
            cache_key("insiders", provider.name, upper),
            self._settings.financials_ttl_seconds,
            lambda: provider.get_insider_transactions(upper),
        )

    async def get_earnings(self, symbol: str) -> Sequence[Earnings]:
        """Return reported earnings with estimates."""
        provider = self._registry.resolve(Capability.EARNINGS)
        upper = symbol.upper()
        return await self._cache.get_or_fetch(
            cache_key("earnings", provider.name, upper),
            self._settings.financials_ttl_seconds,
            lambda: provider.get_earnings(upper),
        )

    async def get_news(self, symbol: str, *, start: date, end: date) -> Sequence[NewsItem]:
        """Return company news over a window."""
        provider = self._registry.resolve(Capability.NEWS)
        upper = symbol.upper()
        return await self._cache.get_or_fetch(
            cache_key("news", provider.name, upper, start, end),
            self._settings.news_ttl_seconds,
            lambda: provider.get_news(upper, start=start, end=end),
        )

    # --- Diagnostics -------------------------------------------------------

    @property
    def cache_stats(self) -> dict[str, int]:
        """Cache hit, miss and coalesce counts."""
        return self._cache.stats

    async def aclose(self) -> None:
        """Release every provider's transport.

        Exposed here so callers never reach past the facade to the registry --
        the whole point of this class is that nothing above it knows a registry
        exists.
        """
        await self._registry.aclose()

    def invalidate_symbol(self, symbol: str) -> int:
        """Drop every cached entry for one symbol.

        Called after a sync rewrites that symbol's stored data, when the cached
        derivatives are known stale before their TTL would say so.
        """
        upper = symbol.upper()
        return sum(
            self._cache.invalidate_prefix(f"{kind}:{provider.name}:{upper}")
            for kind in ("profile", "quote", "metrics", "financials", "candles", "news")
            for provider in self._registry.providers
        )
