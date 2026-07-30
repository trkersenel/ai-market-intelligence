"""Finnhub adapter.

Written against real responses from the free tier, not against the docs, because
the two disagree in ways that matter. What the free tier actually serves, probed
directly:

    universe, profile (with logo), quote, metrics,
    news, analyst ratings, insider transactions,
    financial statements, earnings          -> 200
    candles, price targets, dividends       -> 403

So this adapter declares no charting capability. That is not a limitation to
work around: the platform asks before it requests, and a page renders the
sections this provider can fill. Historical bars come from a second adapter,
which is the entire point of the provider abstraction.

Three unit conventions in the payloads will silently corrupt every derived
number if missed, and all three are handled at the boundary here:

- ``marketCapitalization`` is in **millions** of the reporting currency. Apple
  returns 4880167.96, meaning $4.88T. Passing it through unscaled understates
  every company by six orders of magnitude and quietly reorders any screen
  sorted by size.
- ``shareOutstanding`` and ``10DayAverageTradingVolume`` are likewise in
  millions.
- Margins and returns are already percentages, not fractions: ``grossMarginTTM``
  of 47.86 means 47.86%. Multiplying by 100 again produces a 4,786% gross margin
  that looks obviously wrong -- which is the lucky case. Dividing produces
  0.4786, which looks entirely plausible and is wrong by a factor of 100.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from app.clients.http import HttpClient
from app.core.config import IngestionSettings, MarketDataSettings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.marketdata.domain import (
    AnalystRating,
    CompanyProfile,
    Earnings,
    Financials,
    FinancialStatement,
    InsiderTransaction,
    KeyMetrics,
    Listing,
    MarketSession,
    NewsItem,
    Quote,
    StatementPeriod,
)
from app.marketdata.provider import BaseProvider, Capability, ProviderQuotaExceededError

logger = get_logger(__name__)

#: Payload fields denominated in millions.
_MILLIONS = Decimal(1_000_000)

#: MIC code identifying NASDAQ within the combined US symbol list. The list
#: returns roughly 31,000 US securities across every venue; filtering by MIC is
#: what narrows it to the ~5,600 NASDAQ listings.
NASDAQ_MIC = "XNAS"

#: Maps the platform's statement names onto Finnhub's report sections.
#: Provider responses that mean something specific rather than a generic
#: failure: a bad key must not be retried, and a quota must not be treated
#: as an outage.
_HTTP_UNAUTHORISED = 401
_HTTP_TOO_MANY_REQUESTS = 429

_STATEMENT_SECTIONS = {"income_statement": "ic", "balance_sheet": "bs", "cash_flow": "cf"}


class FinnhubProvider(BaseProvider):
    """Reference data, quotes, fundamentals and news from Finnhub."""

    capabilities = frozenset(
        {
            Capability.UNIVERSE,
            Capability.PROFILE,
            Capability.LOGO,
            Capability.QUOTE,
            Capability.METRICS,
            Capability.FINANCIALS,
            Capability.ANALYST_RATINGS,
            Capability.INSIDER_TRANSACTIONS,
            Capability.EARNINGS,
            Capability.NEWS,
        }
    )

    def __init__(
        self,
        settings: MarketDataSettings,
        ingestion: IngestionSettings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the adapter.

        Raises:
            ExternalServiceError: If no API key is configured.
        """
        if settings.finnhub_api_key is None:
            msg = "Finnhub API key is not configured"
            raise ExternalServiceError(msg)

        self._settings = settings
        self._token = settings.finnhub_api_key.get_secret_value()
        self._http = HttpClient(
            settings=ingestion,
            base_url=settings.finnhub_base_url,
            # The free tier allows 60 calls a minute. Held a little under it so
            # a burst cannot trip the limit, since a 429 costs more than the
            # fraction of a second the margin gives up.
            rate_limit=settings.finnhub_rate_limit,
            provider="finnhub",
            client=client,
        )

    @staticmethod
    def is_configured(settings: MarketDataSettings) -> bool:
        """Return whether a credential is available."""
        return settings.finnhub_api_key is not None

    @property
    def name(self) -> str:
        """Identifier recorded with cached data."""
        return "finnhub"

    async def _get(self, path: str, **params: Any) -> Any:
        """Issue a request with the token attached.

        The token goes in the query string because Finnhub requires it there;
        it is never logged, since the shared HTTP client logs the path and the
        provider name rather than the full URL.
        """
        try:
            return await self._http.get_json(path, params={**params, "token": self._token})
        except ExternalServiceError as exc:
            status = exc.details.get("status_code")
            if status == _HTTP_UNAUTHORISED:
                msg = "Finnhub rejected the API key"
                raise ExternalServiceError(msg, details={"provider": "finnhub"}) from exc
            if status == _HTTP_TOO_MANY_REQUESTS:
                msg = "Finnhub rate limit exceeded"
                raise ProviderQuotaExceededError(msg, details={"provider": "finnhub"}) from exc
            raise

    # --- Universe ----------------------------------------------------------

    async def list_universe(self, exchange: str) -> Sequence[Listing]:
        """Return every listing on an exchange.

        Finnhub serves one combined US file and redirects to static storage to
        do it, so the shared client's ``follow_redirects`` is load-bearing here.
        Filtering happens client-side because the API has no MIC parameter.
        """
        payload = await self._get("/stock/symbol", exchange="US")
        if not isinstance(payload, list):
            msg = "Finnhub returned a malformed symbol list"
            raise ExternalServiceError(msg, details={"provider": "finnhub"})

        wanted = NASDAQ_MIC if exchange.upper() in {"NASDAQ", "XNAS"} else exchange.upper()
        listings = [
            Listing(
                symbol=str(row["symbol"]),
                name=str(row.get("description") or row["symbol"]),
                exchange=str(row.get("mic") or ""),
                currency=str(row.get("currency") or "USD"),
                security_type=str(row.get("type") or "common"),
                figi=row.get("figi") or None,
            )
            for row in payload
            if row.get("symbol") and (wanted == "US" or row.get("mic") == wanted)
        ]
        logger.info("finnhub_universe_fetched", exchange=exchange, listings=len(listings))
        return listings

    # --- Profile -----------------------------------------------------------

    async def get_profile(self, symbol: str) -> CompanyProfile:
        """Return descriptive facts about one issuer.

        Notes:
            The free profile carries no CEO, employee count, sector or business
            description -- only ``finnhubIndustry``. Those fields stay ``None``
            rather than being guessed at, so a page renders them as unavailable
            instead of inventing a plausible value.
        """
        payload = await self._get("/stock/profile2", symbol=symbol.upper())
        if not isinstance(payload, dict) or not payload.get("ticker"):
            msg = f"Finnhub has no profile for {symbol.upper()}"
            raise ExternalServiceError(msg, details={"symbol": symbol.upper()})

        return CompanyProfile(
            symbol=str(payload["ticker"]),
            name=str(payload.get("name") or symbol.upper()),
            logo_url=payload.get("logo") or None,
            exchange=payload.get("exchange") or None,
            industry=payload.get("finnhubIndustry") or None,
            country=payload.get("country") or None,
            website=payload.get("weburl") or None,
            phone=payload.get("phone") or None,
            ipo_date=_to_date(payload.get("ipo")),
            # Both arrive in millions.
            market_cap=_scale(payload.get("marketCapitalization"), _MILLIONS),
            shares_outstanding=_scale(payload.get("shareOutstanding"), _MILLIONS),
            currency=str(payload.get("currency") or "USD"),
        )

    # --- Quote -------------------------------------------------------------

    async def get_quote(self, symbol: str) -> Quote:
        """Return the latest price snapshot.

        Notes:
            This is a trade snapshot, not a quote feed: there is no bid, ask or
            extended-hours print. Those capabilities are undeclared, so nothing
            downstream asks for them.
        """
        upper = symbol.upper()
        payload = await self._get("/quote", symbol=upper)
        if not isinstance(payload, dict):
            msg = f"Finnhub returned a malformed quote for {upper}"
            raise ExternalServiceError(msg, details={"symbol": upper})

        price = _to_decimal(payload.get("c"))
        # Finnhub answers 200 with every field zeroed for an unknown symbol
        # rather than 404, so an all-zero payload is the not-found signal.
        if price is None or (price == 0 and not payload.get("pc")):
            msg = f"Finnhub has no quote for {upper}"
            raise ExternalServiceError(msg, details={"symbol": upper})

        timestamp = payload.get("t")
        return Quote(
            symbol=upper,
            timestamp=(
                datetime.fromtimestamp(int(timestamp), tz=UTC) if timestamp else datetime.now(UTC)
            ),
            price=price,
            open=_to_decimal(payload.get("o")),
            high=_to_decimal(payload.get("h")),
            low=_to_decimal(payload.get("l")),
            previous_close=_to_decimal(payload.get("pc")),
            session=MarketSession.REGULAR,
        )

    # --- Metrics -----------------------------------------------------------

    async def get_metrics(self, symbol: str) -> KeyMetrics:
        """Return valuation, growth and profitability ratios.

        Notes:
            Margins and returns arrive as percentages already. They are stored
            as percentages, and the naming keeps that explicit so no consumer
            multiplies by 100 a second time.
        """
        upper = symbol.upper()
        payload = await self._get("/stock/metric", symbol=upper, metric="all")
        metric = payload.get("metric") if isinstance(payload, dict) else None
        if not isinstance(metric, dict):
            msg = f"Finnhub returned no metrics for {upper}"
            raise ExternalServiceError(msg, details={"symbol": upper})

        return KeyMetrics(
            symbol=upper,
            pe_ratio=_to_float(metric.get("peTTM")),
            forward_pe=_to_float(metric.get("peBasicExclExtraTTM")),
            price_to_book=_to_float(metric.get("pbAnnual")),
            price_to_sales=_to_float(metric.get("psTTM")),
            ev_to_ebitda=_to_float(metric.get("currentEv/freeCashFlowTTM")),
            gross_margin=_to_float(metric.get("grossMarginTTM")),
            operating_margin=_to_float(metric.get("operatingMarginTTM")),
            net_margin=_to_float(metric.get("netProfitMarginTTM")),
            return_on_equity=_to_float(metric.get("roeTTM")),
            return_on_assets=_to_float(metric.get("roaTTM")),
            return_on_invested_capital=_to_float(metric.get("roiTTM")),
            revenue_growth_yoy=_to_float(metric.get("revenueGrowthTTMYoy")),
            earnings_growth_yoy=_to_float(metric.get("netIncomeGrowthTTMYoy")),
            revenue_growth_3y=_to_float(metric.get("revenueGrowth3Y")),
            eps_growth_yoy=_to_float(metric.get("epsGrowthTTMYoy")),
            debt_to_equity=_to_float(metric.get("totalDebt/totalEquityAnnual")),
            current_ratio=_to_float(metric.get("currentRatioAnnual")),
            quick_ratio=_to_float(metric.get("quickRatioAnnual")),
            eps=_to_float(metric.get("epsTTM")),
            book_value_per_share=_to_float(metric.get("bookValuePerShareAnnual")),
            dividend_yield=_to_float(metric.get("dividendYieldIndicatedAnnual")),
            payout_ratio=_to_float(metric.get("payoutRatioTTM")),
            beta=_to_float(metric.get("beta")),
            week_52_change=_to_float(metric.get("52WeekPriceReturnDaily")),
        )

    async def get_52_week_range(self, symbol: str) -> tuple[Decimal | None, Decimal | None]:
        """Return the 52-week high and low.

        Lives on the metrics endpoint rather than the quote, so a caller that
        wants a complete quote must merge the two. Exposed separately to make
        that second call explicit rather than hidden inside ``get_quote``,
        where it would double the request cost of every dashboard tile.
        """
        payload = await self._get("/stock/metric", symbol=symbol.upper(), metric="all")
        metric = payload.get("metric", {}) if isinstance(payload, dict) else {}
        return (_to_decimal(metric.get("52WeekHigh")), _to_decimal(metric.get("52WeekLow")))

    # --- Fundamentals ------------------------------------------------------

    async def get_financials(self, symbol: str) -> Financials:
        """Return the three statements as filed.

        Notes:
            Finnhub serves the SEC filing's own line items, so the labels differ
            between filers and between industries. They are kept verbatim in
            ``line_items`` rather than mapped onto a fixed schema, because a
            fixed schema would drop exactly the lines that distinguish a bank
            from a manufacturer.
        """
        upper = symbol.upper()
        payload = await self._get("/stock/financials-reported", symbol=upper, freq="quarterly")
        reports = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(reports, list):
            return Financials(symbol=upper)

        income: list[FinancialStatement] = []
        balance: list[FinancialStatement] = []
        cash: list[FinancialStatement] = []
        buckets = {"income_statement": income, "balance_sheet": balance, "cash_flow": cash}

        for report in reports:
            filing = report.get("report") or {}
            fiscal = _to_date(report.get("endDate")) or _to_date(report.get("filedDate"))
            if fiscal is None:
                continue
            for statement_name, section in _STATEMENT_SECTIONS.items():
                rows = filing.get(section)
                if not isinstance(rows, list):
                    continue
                buckets[statement_name].append(
                    FinancialStatement(
                        symbol=upper,
                        period=StatementPeriod.QUARTERLY,
                        fiscal_date=fiscal,
                        fiscal_year=int(report.get("year") or fiscal.year),
                        fiscal_quarter=_to_int(report.get("quarter")),
                        currency=str(report.get("currency") or "USD"),
                        filing_date=_to_date(report.get("filedDate")),
                        line_items={
                            str(row.get("label") or row.get("concept")): _to_decimal(
                                row.get("value")
                            )
                            for row in rows
                            if row.get("label") or row.get("concept")
                        },
                    )
                )

        return Financials(
            symbol=upper,
            income_statements=tuple(income),
            balance_sheets=tuple(balance),
            cash_flows=tuple(cash),
        )

    # --- Sell-side and ownership -------------------------------------------

    async def get_analyst_ratings(self, symbol: str) -> Sequence[AnalystRating]:
        """Return aggregated recommendations by period."""
        upper = symbol.upper()
        payload = await self._get("/stock/recommendation", symbol=upper)
        if not isinstance(payload, list):
            return []

        ratings = []
        for row in payload:
            period = _to_date(row.get("period"))
            if period is None:
                continue
            ratings.append(
                AnalystRating(
                    symbol=upper,
                    period=period,
                    strong_buy=_to_int(row.get("strongBuy")) or 0,
                    buy=_to_int(row.get("buy")) or 0,
                    hold=_to_int(row.get("hold")) or 0,
                    sell=_to_int(row.get("sell")) or 0,
                    strong_sell=_to_int(row.get("strongSell")) or 0,
                )
            )
        return sorted(ratings, key=lambda rating: rating.period, reverse=True)

    async def get_insider_transactions(self, symbol: str) -> Sequence[InsiderTransaction]:
        """Return recently reported insider trades."""
        upper = symbol.upper()
        payload = await self._get("/stock/insider-transactions", symbol=upper)
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return []

        transactions = []
        for row in rows:
            transaction_date = _to_date(row.get("transactionDate")) or _to_date(
                row.get("filingDate")
            )
            shares = _to_decimal(row.get("share"))
            if transaction_date is None or shares is None:
                continue
            transactions.append(
                InsiderTransaction(
                    symbol=upper,
                    name=str(row.get("name") or "Unknown"),
                    transaction_date=transaction_date,
                    shares=shares,
                    price=_to_decimal(row.get("transactionPrice")),
                    change=_to_decimal(row.get("change")),
                    transaction_code=row.get("transactionCode") or None,
                    filing_date=_to_date(row.get("filingDate")),
                )
            )
        return sorted(transactions, key=lambda item: item.transaction_date, reverse=True)

    # --- Events ------------------------------------------------------------

    async def get_earnings(self, symbol: str) -> Sequence[Earnings]:
        """Return reported earnings with estimates."""
        upper = symbol.upper()
        payload = await self._get("/stock/earnings", symbol=upper)
        if not isinstance(payload, list):
            return []

        results = []
        for row in payload:
            fiscal = _to_date(row.get("period"))
            if fiscal is None:
                continue
            results.append(
                Earnings(
                    symbol=upper,
                    fiscal_date=fiscal,
                    eps_actual=_to_decimal(row.get("actual")),
                    eps_estimate=_to_decimal(row.get("estimate")),
                    timing=row.get("hour") or None,
                )
            )
        return sorted(results, key=lambda item: item.fiscal_date, reverse=True)

    async def get_news(self, symbol: str, *, start: date, end: date) -> Sequence[NewsItem]:
        """Return company news over a window."""
        upper = symbol.upper()
        payload = await self._get(
            "/company-news",
            symbol=upper,
            **{"from": start.isoformat(), "to": end.isoformat()},
        )
        if not isinstance(payload, list):
            return []

        items = []
        for row in payload:
            published = row.get("datetime")
            headline = row.get("headline")
            url = row.get("url")
            if not published or not headline or not url:
                continue
            items.append(
                NewsItem(
                    headline=str(headline),
                    url=str(url),
                    published_at=datetime.fromtimestamp(int(published), tz=UTC),
                    source=row.get("source") or None,
                    summary=row.get("summary") or None,
                    image_url=row.get("image") or None,
                    symbols=(upper,),
                    category=row.get("category") or None,
                    provider_id=str(row.get("id")) if row.get("id") else None,
                )
            )
        return items

    async def aclose(self) -> None:
        """Release the transport."""
        await self._http.aclose()


# --- Coercion --------------------------------------------------------------
#
# Every provider field passes through one of these. They exist because vendor
# JSON is not typed: a number may arrive as a string, a missing value as null,
# an empty string, a zero or the literal "N/A", and each has to become None
# rather than a plausible-looking figure.


def _to_decimal(value: Any) -> Decimal | None:
    """Convert to an exact Decimal, or None when unusable.

    Routed through ``str`` deliberately: ``Decimal(0.1)`` captures the binary
    expansion, while ``Decimal("0.1")`` is the number the vendor reported.
    """
    if value is None or value == "":
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return result if result.is_finite() else None


def _scale(value: Any, factor: Decimal) -> Decimal | None:
    """Convert and rescale a field reported in millions."""
    parsed = _to_decimal(value)
    return None if parsed is None else parsed * factor


def _to_float(value: Any) -> float | None:
    """Convert to float, or None when unusable."""
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def _to_int(value: Any) -> int | None:
    """Convert to int, or None when unusable."""
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def _to_date(value: Any) -> date | None:
    """Parse an ISO date, tolerating a datetime suffix."""
    if not value:
        return None
    text = str(value)
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None
