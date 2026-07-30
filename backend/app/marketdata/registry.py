"""Capability-based provider resolution.

The single place that knows which vendor serves what. Everything above it asks
for *data*, never for a provider by name, which is what makes "swap the provider
without changing the UI" true rather than aspirational.

Resolution walks the configured providers in order and returns the first that
declares the capability. That ordering is the whole configuration surface: put a
better provider first and it takes over, with no other change anywhere. It also
means the platform runs on whatever combination of keys happens to exist —
Finnhub alone gives everything but charts; adding Twelve Data adds charts; a paid
key later can pre-empt both by being listed first.

When nothing serves a capability, the caller gets a 501 naming what is missing
and which providers were consulted. That is a UI state — "charts need a data
provider that offers them" — not an error page.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.config import MarketDataSettings, Settings
from app.core.logging import get_logger
from app.marketdata.cache import ResponseCache
from app.marketdata.provider import (
    BaseProvider,
    Capability,
    CapabilityNotSupportedError,
    MarketDataProvider,
)

logger = get_logger(__name__)


class ProviderRegistry:
    """Holds the configured providers and resolves capabilities to them."""

    def __init__(self, providers: Sequence[MarketDataProvider]) -> None:
        """Register providers in priority order.

        Args:
            providers: Consulted first-to-last. The first declaring a capability
                serves it.
        """
        self._providers = list(providers)
        logger.info(
            "provider_registry_ready",
            providers=[p.name for p in self._providers],
            capabilities=sorted({c.value for p in self._providers for c in p.capabilities}),
        )

    @property
    def providers(self) -> Sequence[MarketDataProvider]:
        """Every registered provider, in priority order."""
        return tuple(self._providers)

    @property
    def capabilities(self) -> frozenset[Capability]:
        """Everything the registry can serve, across all providers."""
        return frozenset(c for provider in self._providers for c in provider.capabilities)

    def supports(self, capability: Capability) -> bool:
        """Whether any registered provider serves ``capability``."""
        return any(provider.supports(capability) for provider in self._providers)

    def resolve(self, capability: Capability) -> MarketDataProvider:
        """Return the highest-priority provider serving ``capability``.

        Raises:
            CapabilityNotSupportedError: When no provider offers it. The error
                names the capability and the providers consulted, so the
                message is actionable -- "no configured provider serves candles
                (tried: finnhub)" tells the reader exactly what to add.
        """
        for provider in self._providers:
            if provider.supports(capability):
                return provider

        configured = ", ".join(p.name for p in self._providers) or "none"
        msg = (
            f"No configured provider serves {capability.value.replace('_', ' ')} "
            f"(tried: {configured})."
        )
        raise CapabilityNotSupportedError(
            msg,
            details={
                "capability": capability.value,
                "providers": [p.name for p in self._providers],
            },
        )

    def describe(self) -> dict[str, list[str]]:
        """Return each provider's capabilities.

        Surfaced through the API so the frontend can decide what to render
        *before* requesting it. A page that knows charts are unavailable can
        omit the section entirely rather than showing a panel that resolves to
        an error a moment later.
        """
        return {
            provider.name: sorted(c.value for c in provider.capabilities)
            for provider in self._providers
        }

    async def aclose(self) -> None:
        """Release every provider's transport."""
        for provider in self._providers:
            await provider.aclose()


def build_registry(settings: Settings) -> ProviderRegistry:
    """Construct the registry from configuration.

    Providers with no credential are skipped with a log line naming what is
    consequently unavailable. A missing key degrades a feature; it never
    prevents startup, because a platform that refuses to boot without every
    optional credential is a platform nobody can run.
    """
    from app.marketdata.providers.finnhub import FinnhubProvider  # noqa: PLC0415

    market = settings.marketdata
    providers: list[MarketDataProvider] = []

    if FinnhubProvider.is_configured(market):
        providers.append(FinnhubProvider(market, settings.ingestion))
    else:
        logger.warning(
            "provider_unconfigured",
            provider="finnhub",
            missing="MARKETDATA_FINNHUB_API_KEY",
            consequence="no universe, profiles, logos, quotes or fundamentals",
        )

    _warn_about_gaps(market, providers)
    return ProviderRegistry(providers)


def _warn_about_gaps(settings: MarketDataSettings, providers: Sequence[MarketDataProvider]) -> None:
    """Log the capabilities no configured provider can serve.

    Stated at startup rather than discovered at request time, so the gap is
    visible in the boot log instead of appearing as an empty panel later.
    """
    available = {c for provider in providers for c in provider.capabilities}
    notable = {
        Capability.CANDLES_DAILY: "price charts",
        Capability.PROFILE: "company profiles",
        Capability.QUOTE: "live quotes",
        Capability.FINANCIALS: "financial statements",
    }
    missing = [label for capability, label in notable.items() if capability not in available]
    if missing:
        logger.warning(
            "capability_gap",
            unavailable=missing,
            hint=(
                "set MARKETDATA_TWELVEDATA_API_KEY for charts"
                if settings.twelvedata_api_key is None
                else "check provider configuration"
            ),
        )


class NullProvider(BaseProvider):
    """A provider that serves nothing.

    Used where a registry must exist but no credential does -- chiefly tests
    that exercise the "capability unavailable" path without reaching a network.
    """

    capabilities = frozenset()

    @property
    def name(self) -> str:
        """Identifier recorded with cached data."""
        return "null"


__all__ = ["NullProvider", "ProviderRegistry", "ResponseCache", "build_registry"]
