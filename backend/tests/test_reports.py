"""Tests for the on-demand company briefing.

The value here is not that a model produces prose -- that is the model's job and
cannot be asserted. It is that the model is only ever handed facts this platform
fetched, that a missing capability subtracts evidence instead of failing, and
that an unreachable model degrades to something honest rather than to something
that merely sounds generated.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pytest

from app.marketdata.domain import (
    AnalystRating,
    CompanyProfile,
    Earnings,
    KeyMetrics,
    MarketSession,
    Quote,
)
from app.marketdata.provider import CapabilityNotSupportedError
from app.services.rag.llm import LlmResponse
from app.services.reports import CompanyReportService


class FakeMarketData:
    """Serves scripted facts, or raises for capabilities it does not have."""

    def __init__(self, **overrides: Any) -> None:
        self._data: dict[str, Any] = {
            "profile": CompanyProfile(symbol="MU", name="Micron Technology Inc", industry="Semis"),
            "quote": Quote(
                symbol="MU",
                timestamp=datetime(2026, 7, 30, tzinfo=UTC),
                price=Decimal("874.66"),
                previous_close=Decimal("739.00"),
                session=MarketSession.REGULAR,
            ),
            "metrics": KeyMetrics(symbol="MU", pe_ratio=18.99, gross_margin=72.57),
            "ratings": [
                AnalystRating(
                    symbol="MU",
                    period=date(2026, 7, 1),
                    strong_buy=18,
                    buy=33,
                    hold=4,
                    sell=1,
                    strong_sell=0,
                )
            ],
            "earnings": [
                Earnings(
                    symbol="MU",
                    fiscal_date=date(2026, 6, 30),
                    eps_actual=Decimal("25.11"),
                    eps_estimate=Decimal("21.40"),
                )
            ],
        }
        self._data.update(overrides)

    async def _serve(self, key: str) -> Any:
        value = self._data[key]
        if isinstance(value, Exception):
            raise value
        return value

    async def get_profile(self, symbol: str) -> Any:
        return await self._serve("profile")

    async def get_quote(self, symbol: str) -> Any:
        return await self._serve("quote")

    async def get_metrics(self, symbol: str) -> Any:
        return await self._serve("metrics")

    async def get_analyst_ratings(self, symbol: str) -> Any:
        return await self._serve("ratings")

    async def get_earnings(self, symbol: str) -> Any:
        return await self._serve("earnings")


class FakeLlm:
    """Records the context it was given."""

    def __init__(self, *, generative: bool = True) -> None:
        self._generative = generative
        self.context: str | None = None
        self.question: str | None = None

    @property
    def model_name(self) -> str:
        return "fake-model" if self._generative else "extractive-v1"

    @property
    def is_generative(self) -> bool:
        return self._generative

    async def complete(self, *, question: str, context: str) -> LlmResponse:
        self.question = question
        self.context = context
        return LlmResponse(text="A briefing.", model_name=self.model_name)


class FakeReportRepository:
    """An in-memory cache."""

    def __init__(self, stored: dict[str, Any] | None = None) -> None:
        self.stored = stored
        self.writes: list[dict[str, Any]] = []

    async def get_fresh(self, symbol: str, *, model: str, newer_than: datetime) -> Any:
        if self.stored and self.stored.get("model") == model:
            return self.stored
        return None

    async def store(self, report: dict[str, Any]) -> None:
        self.writes.append(report)


def _service(market: Any, llm: Any, repository: Any) -> CompanyReportService:
    return CompanyReportService(market_data=market, llm=llm, reports=repository)  # type: ignore[arg-type]


class TestEvidence:
    """What the model is allowed to see."""

    async def test_the_model_receives_numbered_facts_not_a_company_name(self) -> None:
        """The whole design in one assertion.

        Asked about Micron directly, a 3B model will report a market cap from
        its training data. Handed numbered facts, it interprets those facts.
        """
        llm = FakeLlm()
        await _service(FakeMarketData(), llm, FakeReportRepository()).get("MU")

        assert llm.context is not None
        assert "[1]" in llm.context
        assert "874.66" in llm.context
        assert "18.99" in llm.context

    async def test_the_change_is_computed_from_the_facts(self) -> None:
        """Derived in the evidence rather than left for the model to work out.

        A small model asked to divide two prices will produce a plausible
        number that is not the right one.
        """
        llm = FakeLlm()
        await _service(FakeMarketData(), llm, FakeReportRepository()).get("MU")

        assert llm.context is not None
        assert "18.36%" in llm.context

    async def test_margins_are_not_rescaled(self) -> None:
        """The provider already sends percentages.

        Multiplying by 100 would hand the model a 72.57% gross margin as
        7,257%, and it would dutifully explain what an extraordinary figure
        that is.
        """
        llm = FakeLlm()
        await _service(FakeMarketData(), llm, FakeReportRepository()).get("MU")

        assert llm.context is not None
        assert "72.57%" in llm.context
        assert "7,257" not in llm.context

    async def test_an_unsupported_capability_subtracts_evidence(self) -> None:
        """A free tier without ratings must not fail the whole briefing."""
        market = FakeMarketData(ratings=CapabilityNotSupportedError("no ratings provider"))
        llm = FakeLlm()

        report = await _service(market, llm, FakeReportRepository()).get("MU")

        assert report.generated is True
        assert llm.context is not None
        assert "Analyst" not in llm.context
        assert "874.66" in llm.context

    async def test_absent_fields_are_omitted_rather_than_labelled(self) -> None:
        """A field the provider omitted must not appear at all.

        A line reading "P/E: not available" invites the model to explain the
        absence, and explaining an absence means inventing a reason.
        """
        market = FakeMarketData(metrics=KeyMetrics(symbol="MU", pe_ratio=None, gross_margin=None))
        llm = FakeLlm()

        await _service(market, llm, FakeReportRepository()).get("MU")

        assert llm.context is not None
        assert "Price/earnings" not in llm.context
        assert "None" not in llm.context

    async def test_no_facts_at_all_yields_a_coverage_statement(self) -> None:
        """An unknown symbol is a coverage gap, not a verdict on the company."""
        market = FakeMarketData(
            profile=CapabilityNotSupportedError("x"),
            quote=CapabilityNotSupportedError("x"),
            metrics=CapabilityNotSupportedError("x"),
            ratings=CapabilityNotSupportedError("x"),
            earnings=CapabilityNotSupportedError("x"),
        )
        llm = FakeLlm()

        report = await _service(market, llm, FakeReportRepository()).get("MU")

        assert report.generated is False
        assert report.evidence == ()
        assert llm.context is None, "the model must not be asked to write about nothing"


class TestDegradation:
    """What happens with no generative model."""

    async def test_the_evidence_is_returned_as_a_digest(self) -> None:
        """Not an error and not a placeholder.

        The extractive answerer selects sentences from prose; there is no prose
        here, only figures, so it has nothing to select. Returning the figures
        is the truthful thing to return.
        """
        report = await _service(
            FakeMarketData(), FakeLlm(generative=False), FakeReportRepository()
        ).get("MU")

        assert report.generated is False
        assert any("874.66" in line for line in report.evidence)
        assert "ollama serve" in report.summary


class TestCache:
    """Serving a previously written briefing."""

    async def test_a_fresh_report_is_served_from_the_cache(self) -> None:
        stored = {
            "summary": "Cached briefing.",
            "evidence": ["Last price: 874.66"],
            "model": "fake-model",
            "generated_at": datetime(2026, 7, 30, tzinfo=UTC),
            "generated": True,
        }
        llm = FakeLlm()

        report = await _service(FakeMarketData(), llm, FakeReportRepository(stored)).get("MU")

        assert report.cached is True
        assert report.summary == "Cached briefing."
        assert llm.context is None, "a cache hit must not run inference"

    async def test_a_report_from_another_model_is_not_reused(self) -> None:
        """A briefing written by a different model is not reused.

        Provenance is the one field a reader uses to judge how much weight the
        analysis deserves, so it must never be wrong.
        """
        stored = {
            "summary": "Written by something else.",
            "evidence": [],
            "model": "some-other-model",
            "generated_at": datetime(2026, 7, 30, tzinfo=UTC),
            "generated": True,
        }

        report = await _service(FakeMarketData(), FakeLlm(), FakeReportRepository(stored)).get("MU")

        assert report.cached is False
        assert report.model == "fake-model"

    async def test_refresh_bypasses_the_cache(self) -> None:
        stored = {
            "summary": "Cached briefing.",
            "evidence": [],
            "model": "fake-model",
            "generated_at": datetime(2026, 7, 30, tzinfo=UTC),
            "generated": True,
        }
        llm = FakeLlm()

        report = await _service(FakeMarketData(), llm, FakeReportRepository(stored)).get(
            "MU", refresh=True
        )

        assert report.cached is False
        assert llm.context is not None

    async def test_a_generated_report_is_written_back(self) -> None:
        repository = FakeReportRepository()

        await _service(FakeMarketData(), FakeLlm(), repository).get("MU")

        assert len(repository.writes) == 1
        assert repository.writes[0]["symbol"] == "MU"
        assert repository.writes[0]["model"] == "fake-model"

    @pytest.mark.parametrize("symbol", ["mu", "Mu", "MU"])
    async def test_the_symbol_is_normalised(self, symbol: str) -> None:
        """The cache key is the symbol; three spellings must not be three rows."""
        repository = FakeReportRepository()

        report = await _service(FakeMarketData(), FakeLlm(), repository).get(symbol)

        assert report.symbol == "MU"
        assert repository.writes[0]["symbol"] == "MU"


class TestMonetaryScale:
    """Large money figures given to the model."""

    @pytest.mark.parametrize(
        ("amount", "expected"),
        [
            (Decimal("4656563865378.91"), "4.66 trillion USD"),
            (Decimal("918433735531.99"), "918.43 billion USD"),
            (Decimal("45000000"), "45.00 million USD"),
            (Decimal("4500"), "4,500.00 USD"),
        ],
        ids=["trillions", "billions", "millions", "small"],
    )
    async def test_market_cap_is_scaled_and_correctly_labelled(
        self, amount: Decimal, expected: str
    ) -> None:
        """Market cap is scaled to a human unit and labelled with it.

        The original bug: the line read "(USD millions)" over a figure the
        adapter had already converted to absolute dollars. A model trusting the
        label would report NVIDIA's market cap as $4.6 quintillion. And a raw
        fifteen-digit number gets reproduced digit for digit into prose no
        analyst would write.
        """
        market = FakeMarketData(
            profile=CompanyProfile(symbol="MU", name="Example", market_cap=amount)
        )
        llm = FakeLlm()

        await _service(market, llm, FakeReportRepository()).get("MU")

        assert llm.context is not None
        assert f"Market capitalisation: {expected}" in llm.context
        assert "millions)" not in llm.context
