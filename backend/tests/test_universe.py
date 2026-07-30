"""Tests for the browsable exchange universe.

The sync service is exercised against a fake repository and a fake provider, so
these run without a database or a network. The parts worth protecting are the
reconciliation rules -- dedupe, deactivate, degrade -- because each one exists
because of a specific failure that is silent when it recurs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from app.marketdata.domain import Listing
from app.marketdata.provider import CapabilityNotSupportedError
from app.services.universe import UniverseSyncService


class FakeMarketData:
    """Returns scripted listings."""

    def __init__(
        self,
        listings: Sequence[Listing] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self._listings = list(listings or [])
        self._error = error
        self.requested: list[str | None] = []

    @property
    def capabilities(self) -> dict[str, list[str]]:
        return {"fake": ["universe", "quote"]}

    async def list_universe(self, exchange: str | None = None) -> Sequence[Listing]:
        self.requested.append(exchange)
        if self._error is not None:
            raise self._error
        return self._listings


class FakeListingRepository:
    """Records what a sync would have written."""

    def __init__(self) -> None:
        self.upserted: list[dict[str, Any]] = []
        self.deactivate_calls: list[tuple[set[str], str]] = []

    async def upsert_many(self, rows: Sequence[dict[str, Any]]) -> int:
        self.upserted.extend(rows)
        return len(rows)

    async def deactivate_missing(self, symbols: Any, *, source: str) -> int:
        self.deactivate_calls.append(({s.upper() for s in symbols}, source))
        return 0


def _listing(symbol: str, name: str = "Example Corp", exchange: str = "XNAS") -> Listing:
    return Listing(symbol=symbol, name=name, exchange=exchange)


class TestSync:
    """Reconciling the stored universe with the provider's file."""

    async def test_every_listing_is_written_with_its_provenance(self) -> None:
        """Source and timestamp make a stale or mixed universe diagnosable."""
        market = FakeMarketData([_listing("MU"), _listing("NVDA")])
        repository = FakeListingRepository()
        service = UniverseSyncService(market_data=market, listings=repository)  # type: ignore[arg-type]

        report = await service.sync()

        assert (report.fetched, report.written) == (2, 2)
        assert report.succeeded
        assert {row["symbol"] for row in repository.upserted} == {"MU", "NVDA"}
        assert all(row["source"] == "fake" for row in repository.upserted)
        assert all(row["synced_at"] is not None for row in repository.upserted)

    async def test_symbols_are_upper_cased(self) -> None:
        """Stored symbols are normalised to upper case.

        The join to ``ticker`` is on symbol, so a case mismatch would silently
        report every tracked company as untracked.
        """
        market = FakeMarketData([_listing("mu")])
        repository = FakeListingRepository()
        service = UniverseSyncService(market_data=market, listings=repository)  # type: ignore[arg-type]

        await service.sync()

        assert repository.upserted[0]["symbol"] == "MU"

    async def test_a_duplicated_symbol_is_collapsed_before_the_upsert(self) -> None:
        """PostgreSQL rejects an ON CONFLICT whose own VALUES repeat the key.

        The provider's file legitimately can: the same symbol may appear under
        two MICs. Without the dedupe the whole sync fails with "cannot affect
        row a second time" -- and it fails for the entire batch, not one row.
        """
        market = FakeMarketData(
            [
                _listing("MU", name="Micron", exchange="XNAS"),
                _listing("MU", name="Micron", exchange="ARCX"),
                _listing("NVDA"),
            ]
        )
        repository = FakeListingRepository()
        service = UniverseSyncService(market_data=market, listings=repository)  # type: ignore[arg-type]

        report = await service.sync()

        assert report.fetched == 3
        assert len(repository.upserted) == 2
        assert report.written == 2

    async def test_the_deactivation_set_is_what_was_actually_written(self) -> None:
        """Deactivation sees the deduplicated set, scoped to this provider.

        The scoping is what stops one provider's sync from deactivating rows
        another provider supplied -- which is what would happen the first time
        a second provider covering a different exchange was added.
        """
        market = FakeMarketData([_listing("MU"), _listing("MU"), _listing("NVDA")])
        repository = FakeListingRepository()
        service = UniverseSyncService(market_data=market, listings=repository)  # type: ignore[arg-type]

        await service.sync()

        present, source = repository.deactivate_calls[0]
        assert present == {"MU", "NVDA"}
        assert source == "fake"

    async def test_an_unsupported_provider_is_reported_not_raised(self) -> None:
        """A provider without the capability yields a report, not an exception.

        The scheduler runs six other jobs; one missing capability must not take
        them down, and the reason belongs in the report where it is logged once.
        """
        market = FakeMarketData(error=CapabilityNotSupportedError("no universe provider"))
        repository = FakeListingRepository()
        service = UniverseSyncService(market_data=market, listings=repository)  # type: ignore[arg-type]

        report = await service.sync()

        assert report.succeeded is False
        assert report.error is not None
        assert "universe" in report.error
        assert repository.upserted == []

    async def test_an_empty_universe_writes_nothing(self) -> None:
        """An empty response is a no-op, not a mass delisting.

        A provider outage returning an empty file must not deactivate the whole
        stored universe -- which is exactly what a naive reconcile would do.
        """
        market = FakeMarketData([])
        repository = FakeListingRepository()
        service = UniverseSyncService(market_data=market, listings=repository)  # type: ignore[arg-type]

        report = await service.sync()

        assert (report.fetched, report.written, report.deactivated) == (0, 0, 0)
        assert repository.upserted == []
        # The repository's own guard: an empty present-set is a no-op rather
        # than "deactivate everything".
        present, _ = repository.deactivate_calls[0]
        assert present == set()

    @pytest.mark.parametrize("exchange", [None, "XNAS", "NASDAQ"])
    async def test_the_requested_exchange_is_passed_through(self, exchange: str | None) -> None:
        """A second exchange must not require a code change."""
        market = FakeMarketData([_listing("MU")])
        service = UniverseSyncService(market_data=market, listings=FakeListingRepository())  # type: ignore[arg-type]

        await service.sync(exchange)

        assert market.requested == [exchange]
