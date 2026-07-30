"""Syncing the exchange universe into PostgreSQL.

The point of storing the symbol file rather than querying the provider is that
search must be free. Typing into a search box fires a request per keystroke, and
the free tier allows sixty a minute -- so proxying search to the provider would
exhaust the quota during a single demo, and each keystroke would cost a network
round trip besides.

Synced once a day. New listings appear at IPO and disappear at delisting, so a
daily refresh is far more current than the data warrants.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from app.core.logging import get_logger
from app.marketdata.provider import Capability, CapabilityNotSupportedError
from app.marketdata.service import MarketDataService
from app.repositories.listing import ListingRepository

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class UniverseSyncReport:
    """What one sync did."""

    exchange: str
    fetched: int
    written: int
    deactivated: int
    succeeded: bool = True
    error: str | None = None


class UniverseSyncService:
    """Keeps the stored universe aligned with the provider's symbol file."""

    def __init__(
        self,
        *,
        market_data: MarketDataService,
        listings: ListingRepository,
    ) -> None:
        """Wire the service to its collaborators."""
        self._market_data = market_data
        self._listings = listings

    async def sync(self, exchange: str | None = None) -> UniverseSyncReport:
        """Fetch the universe and reconcile it with what is stored.

        A provider that cannot serve the universe is reported as an unsuccessful
        run rather than raised: the scheduler must keep running its other jobs,
        and the reason belongs in the report where the caller can log it once.

        Args:
            exchange: MIC or colloquial name. Defaults to the configured
                exchange.

        Returns:
            Counts for the run, or a failure report naming the cause.
        """
        try:
            listings = await self._market_data.list_universe(exchange)
        except CapabilityNotSupportedError as exc:
            logger.warning("universe_sync_unsupported", error=str(exc))
            return UniverseSyncReport(
                exchange=exchange or "",
                fetched=0,
                written=0,
                deactivated=0,
                succeeded=False,
                error=str(exc),
            )

        provider = self._provider_name()
        now = datetime.now(UTC)
        rows: list[dict[str, object]] = [
            {
                "symbol": listing.symbol.upper(),
                "name": listing.name,
                "exchange": listing.exchange,
                "currency": listing.currency,
                "security_type": listing.security_type,
                "figi": listing.figi,
                "is_active": True,
                "source": provider,
                "synced_at": now,
            }
            for listing in listings
        ]

        # Deduplicate before the upsert. PostgreSQL rejects an ON CONFLICT
        # statement whose own VALUES contain the conflict key twice -- "cannot
        # affect row a second time" -- and a provider file legitimately can,
        # since the same symbol may be listed under two MICs.
        unique: dict[str, dict[str, object]] = {}
        for row in rows:
            unique[str(row["symbol"])] = row

        written = await self._listings.upsert_many(list(unique.values()))
        deactivated = await self._listings.deactivate_missing(unique, source=provider)

        logger.info(
            "universe_synced",
            exchange=exchange or "default",
            provider=provider,
            fetched=len(rows),
            unique=len(unique),
            written=written,
            deactivated=deactivated,
        )
        return UniverseSyncReport(
            exchange=exchange or "",
            fetched=len(rows),
            written=written,
            deactivated=deactivated,
        )

    def _provider_name(self) -> str:
        """Return the name of whichever provider serves the universe."""
        for name, capabilities in self._market_data.capabilities.items():
            if Capability.UNIVERSE.value in capabilities:
                return name
        return "unknown"  # pragma: no cover - unreachable once resolution succeeded
