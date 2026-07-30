"""On-demand AI company reports.

Generated when a company page opens and cached, so the second visit is instant
and the model runs once per symbol per day rather than once per page view.

The design point worth stating: **the model is never asked what it knows about
a company.** It is handed a numbered block of facts this platform just fetched
from a provider, and asked to explain what they mean together. Every number in
the output exists in that block, and the prompt requires each claim to cite the
line it came from. Ask ``llama3.2:3b`` about Micron directly and it will
cheerfully report a market cap from its training data; ask it to interpret
sixteen numbered facts and it interprets those facts.

That is also why a missing capability is not a failure here. If the free tier
serves no analyst ratings, the ratings lines are simply absent from the evidence
and the report does not discuss coverage. It never says "no data available for
ratings" in the middle of a paragraph, and it never invents one.

When no generative model is reachable the service does not error and does not
degrade to something that sounds generated. It returns the same evidence as a
plain factual digest, marked as such. A reader gets the numbers with no
narrative rather than a narrative with no grounding.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from app.core.logging import get_logger
from app.marketdata.service import MarketDataService
from app.repositories.documents import AiReportRepository
from app.services.rag.llm import LlmClient

logger = get_logger(__name__)

#: What the model is asked to produce. Phrased as an analyst's brief rather than
#: "summarise", because "summarise" reliably yields a restatement of the input
#: list, which the page already shows above the panel.
REPORT_QUESTION = (
    "Write a short analyst briefing on this company. Cover, in this order: "
    "(1) what the recent price action and valuation say together, "
    "(2) what the profitability and growth figures imply about the business, "
    "(3) what the sell-side and any recent news add or contradict. "
    "Lead with the most consequential fact. Where two facts point in opposite "
    "directions, say so plainly rather than averaging them. Three short "
    "paragraphs, no headings, no bullet points."
)


@dataclass(frozen=True, slots=True)
class CompanyReport:
    """A generated briefing with the evidence it was built from."""

    symbol: str
    summary: str
    evidence: tuple[str, ...]
    model: str
    generated_at: datetime
    #: False when no generative model was reachable and the evidence is being
    #: returned as a plain digest. The API surfaces it so the UI can label the
    #: panel honestly instead of presenting a list as an analysis.
    generated: bool = True
    #: True when served from the cache rather than produced now. Diagnostic, and
    #: it explains why a report can mention a price that has since moved.
    cached: bool = False


@dataclass
class _Evidence:
    """Numbered facts, in the order they will be shown to the model."""

    lines: list[str] = field(default_factory=list)

    def add_money(self, label: str, value: Decimal | float | None) -> None:
        """Record a large monetary figure at human scale.

        Two problems solved at once. The provider adapter already converts
        Finnhub's millions to absolute dollars, so a label saying "millions"
        would misstate the unit by a factor of a million -- and a model that
        trusts the label reports NVIDIA at $4.6 quintillion. And a raw
        "4,656,563,865,378.91" is a figure the model reproduces digit for digit
        into prose no analyst would write.
        """
        if value is None:
            return
        amount = float(value)
        for threshold, suffix in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
            if abs(amount) >= threshold:
                self.lines.append(f"{label}: {amount / threshold:,.2f} {suffix} USD")
                return
        self.lines.append(f"{label}: {amount:,.2f} USD")

    def add(self, label: str, value: object, *, unit: str = "") -> None:
        """Record one fact, skipping anything the provider did not supply.

        The skip is the important half. A line reading "P/E: not available"
        invites the model to discuss the absence, and a small model asked to
        discuss an absence tends to explain it -- which means inventing a
        reason.
        """
        if value is None or value == "":
            return
        rendered = _render(value)
        self.lines.append(f"{label}: {rendered}{unit}")

    def numbered(self) -> str:
        """Render as the numbered passages the system prompt refers to."""
        return "\n".join(f"[{index}] {line}" for index, line in enumerate(self.lines, start=1))


def _render(value: object) -> str:
    """Format a value for the evidence block.

    Rounded on the way in rather than passed at full precision: a model shown
    "70.55000000000001" will faithfully reproduce every digit, and a briefing
    quoting ROE to twelve decimal places reads as machine output.
    """
    if isinstance(value, Decimal):
        return f"{value:,.2f}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


class CompanyReportService:
    """Builds, caches and serves the per-company briefing."""

    def __init__(
        self,
        *,
        market_data: MarketDataService,
        llm: LlmClient,
        reports: AiReportRepository,
        ttl_hours: int = 12,
    ) -> None:
        """Wire the service to its collaborators.

        Args:
            market_data: Source of every fact in the evidence block.
            llm: Generator. May be non-generative, which is handled.
            reports: Cache.
            ttl_hours: How long a report stays fresh. Twelve hours means a
                report generated before the open is regenerated after the
                close, which is when the facts have actually changed.
        """
        self._market_data = market_data
        self._llm = llm
        self._reports = reports
        self._ttl = timedelta(hours=ttl_hours)

    async def get(self, symbol: str, *, refresh: bool = False) -> CompanyReport:
        """Return a briefing, generating one if none is fresh.

        Args:
            symbol: Ticker.
            refresh: Bypass the cache and regenerate.

        Returns:
            The report, flagged with whether it was cached and whether a model
            actually wrote it.
        """
        upper = symbol.upper()

        if not refresh:
            cached = await self._reports.get_fresh(
                upper, model=self._llm.model_name, newer_than=datetime.now(UTC) - self._ttl
            )
            if cached is not None:
                return CompanyReport(
                    symbol=upper,
                    summary=str(cached["summary"]),
                    evidence=tuple(cached.get("evidence", [])),
                    model=str(cached["model"]),
                    generated_at=cached["generated_at"],
                    generated=bool(cached.get("generated", True)),
                    cached=True,
                )

        evidence = await self._gather(upper)
        if not evidence.lines:
            msg = "no facts available"
            logger.info("report_skipped", symbol=upper, reason=msg)
            return CompanyReport(
                symbol=upper,
                summary=(
                    f"No provider currently serves data for {upper}, so there is nothing to "
                    "analyse. This is a coverage gap, not a judgement about the company."
                ),
                evidence=(),
                model=self._llm.model_name,
                generated_at=datetime.now(UTC),
                generated=False,
            )

        report = await self._write(upper, evidence)
        await self._reports.store(
            {
                "symbol": report.symbol,
                "summary": report.summary,
                "evidence": list(report.evidence),
                "model": report.model,
                "generated": report.generated,
                "generated_at": report.generated_at,
            }
        )
        return report

    async def _write(self, symbol: str, evidence: _Evidence) -> CompanyReport:
        """Produce the prose, or the honest fallback."""
        now = datetime.now(UTC)

        if not self._llm.is_generative:
            # Not an error path and not a placeholder. The extractive answerer
            # works by selecting sentences from prose passages; there is no
            # prose here, only figures, so it has nothing to select. Returning
            # the figures as a digest is the truthful thing to return.
            logger.info("report_not_generated", symbol=symbol, reason="no generative model")
            return CompanyReport(
                symbol=symbol,
                summary=(
                    "No local or hosted language model is currently reachable, so this is the "
                    "underlying evidence rather than a written analysis. Start Ollama "
                    "(`ollama serve`) or configure an API key to have it interpreted."
                ),
                evidence=tuple(evidence.lines),
                model=self._llm.model_name,
                generated_at=now,
                generated=False,
            )

        response = await self._llm.complete(
            question=REPORT_QUESTION,
            context=evidence.numbered(),
        )
        logger.info(
            "report_generated",
            symbol=symbol,
            model=response.model_name,
            facts=len(evidence.lines),
            completion_tokens=response.completion_tokens,
        )
        return CompanyReport(
            symbol=symbol,
            summary=response.text,
            evidence=tuple(evidence.lines),
            model=response.model_name,
            generated_at=now,
        )

    async def _gather(self, symbol: str) -> _Evidence:
        """Collect every fact a configured provider will serve.

        The five fetches run concurrently and independently: each is cached by
        the market data layer, and a capability the free tier does not serve
        must subtract its lines from the evidence rather than fail the report.
        """
        profile, quote, metrics, ratings, earnings = await asyncio.gather(
            self._market_data.get_profile(symbol),
            self._market_data.get_quote(symbol),
            self._market_data.get_metrics(symbol),
            self._market_data.get_analyst_ratings(symbol),
            self._market_data.get_earnings(symbol),
            return_exceptions=True,
        )

        evidence = _Evidence()
        _add_profile(evidence, profile)
        _add_quote(evidence, quote)
        _add_metrics(evidence, metrics)
        _add_ratings(evidence, ratings)
        _add_earnings(evidence, earnings)
        return evidence


def _usable(result: Any) -> bool:
    """Whether a gathered result is data rather than a raised exception."""
    if isinstance(result, BaseException):
        logger.debug("report_evidence_unavailable", error=str(result))
        return False
    return result is not None


def _add_profile(evidence: _Evidence, profile: Any) -> None:
    """Add identity, so the model knows what it is describing."""
    if not _usable(profile):
        return
    evidence.add("Company", profile.name)
    evidence.add("Industry", profile.industry)
    evidence.add("Country", profile.country)
    evidence.add_money("Market capitalisation", profile.market_cap)


def _add_quote(evidence: _Evidence, quote: Any) -> None:
    """Add the price and its move."""
    if not _usable(quote):
        return
    evidence.add("Last price", quote.price)
    evidence.add("Previous close", quote.previous_close)
    if quote.price is not None and quote.previous_close not in (None, 0):
        change = (quote.price - quote.previous_close) / quote.previous_close * 100
        evidence.add("Change since previous close", float(change), unit="%")
    evidence.add("52-week high", quote.week_52_high)
    evidence.add("52-week low", quote.week_52_low)


def _add_metrics(evidence: _Evidence, metrics: Any) -> None:
    """Add valuation, profitability and growth.

    Margins and growth rates arrive from the provider already as percentages,
    so the unit is appended and the number is never rescaled -- rescaling would
    hand the model a 72% gross margin as 7,257% and it would dutifully explain
    what an extraordinary figure that is.
    """
    if not _usable(metrics):
        return
    evidence.add("Price/earnings ratio", metrics.pe_ratio)
    evidence.add("Forward price/earnings ratio", metrics.forward_pe)
    evidence.add("Price/book ratio", metrics.price_to_book)
    evidence.add("EV/EBITDA", metrics.ev_to_ebitda)
    evidence.add("Gross margin", metrics.gross_margin, unit="%")
    evidence.add("Operating margin", metrics.operating_margin, unit="%")
    evidence.add("Net margin", metrics.net_margin, unit="%")
    evidence.add("Return on equity", metrics.return_on_equity, unit="%")
    evidence.add("Revenue growth year over year", metrics.revenue_growth_yoy, unit="%")
    evidence.add("EPS growth year over year", metrics.eps_growth_yoy, unit="%")
    evidence.add("Debt/equity", metrics.debt_to_equity)
    evidence.add("Beta", metrics.beta)


def _add_ratings(evidence: _Evidence, ratings: Any) -> None:
    """Add the most recent sell-side tally."""
    if not _usable(ratings) or not ratings:
        return
    latest = ratings[0]
    evidence.add(
        "Analyst ratings",
        f"{latest.total} analysts covering as of {latest.period.isoformat()}; "
        f"{latest.strong_buy} strong buy, {latest.buy} buy, {latest.hold} hold, "
        f"{latest.sell} sell, {latest.strong_sell} strong sell",
    )
    evidence.add("Analyst consensus", latest.consensus)


def _add_earnings(evidence: _Evidence, earnings: Any) -> None:
    """Add the last four quarters against estimate.

    Four rather than all of them: a small model given twenty quarters starts
    describing the list instead of the trend, and the recent ones are what a
    briefing turns on.
    """
    if not _usable(earnings) or not earnings:
        return
    for quarter in earnings[:4]:
        if quarter.eps_actual is None:
            continue
        estimate = (
            f" against an estimate of {quarter.eps_estimate:,.2f}"
            if quarter.eps_estimate is not None
            else ""
        )
        evidence.add(
            f"Earnings for quarter ending {quarter.fiscal_date.isoformat()}",
            f"EPS {quarter.eps_actual:,.2f}{estimate}",
        )
