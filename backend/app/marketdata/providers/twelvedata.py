"""Twelve Data adapter.

Fills the gap Finnhub's free tier leaves: historical and intraday bars, which it
refuses with a 403. Probed against the real free plan, which serves:

    time_series (1min through 1month), quote, eod   -> 200

The quote is richer than Finnhub's -- it carries average volume and the 52-week
range, which Finnhub keeps on a separate metrics endpoint. Even so this adapter
is registered *after* Finnhub for quotes, because the free plan allows 8 requests
a minute against Finnhub's 60. Bars are cached for fifteen minutes and are worth
the budget; a quote refreshing every few seconds is not.

One detail worth keeping: every numeric field arrives as a **string**
(``"193.375"``, not ``193.375``). That is a gift rather than a nuisance. It means
a price can reach ``Decimal`` without ever passing through a float, so the value
stored is exactly the one the exchange printed -- no binary rounding anywhere in
the path.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from app.clients.http import HttpClient
from app.core.config import IngestionSettings, MarketDataSettings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.marketdata.domain import (
    Candle,
    CandleSeries,
    Interval,
    MarketSession,
    Quote,
)
from app.marketdata.provider import BaseProvider, Capability, ProviderQuotaExceededError

logger = get_logger(__name__)

#: The platform's intervals mapped onto Twelve Data's vocabulary.
_INTERVALS: dict[Interval, str] = {
    Interval.MINUTE_1: "1min",
    Interval.MINUTE_5: "5min",
    Interval.MINUTE_15: "15min",
    Interval.MINUTE_30: "30min",
    Interval.HOUR_1: "1h",
    Interval.DAY_1: "1day",
    Interval.WEEK_1: "1week",
    Interval.MONTH_1: "1month",
}

#: Bars per request. The free plan caps output at 5000, which is about twenty
#: years of daily bars -- more than the longest chart range needs.
_MAX_OUTPUT_SIZE = 5000

#: Twelve Data signals errors in the body with its own code while returning
#: HTTP 200, so the status alone never reveals a failure.
_CODE_QUOTA_EXCEEDED = 429
_CODE_NOT_FOUND = 404


class TwelveDataProvider(BaseProvider):
    """Historical and intraday bars, plus quotes, from Twelve Data."""

    capabilities = frozenset(
        {
            Capability.CANDLES_DAILY,
            Capability.CANDLES_INTRADAY,
            Capability.QUOTE,
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
        if settings.twelvedata_api_key is None:
            msg = "Twelve Data API key is not configured"
            raise ExternalServiceError(msg)

        self._settings = settings
        self._token = settings.twelvedata_api_key.get_secret_value()
        self._http = HttpClient(
            settings=ingestion,
            base_url=settings.twelvedata_base_url,
            # 8 requests a minute on the free plan, so roughly one every eight
            # seconds. The limiter's burst lets a page load several series
            # before throttling begins.
            rate_limit=settings.twelvedata_rate_limit,
            provider="twelvedata",
            client=client,
        )

    @staticmethod
    def is_configured(settings: MarketDataSettings) -> bool:
        """Return whether a credential is available."""
        return settings.twelvedata_api_key is not None

    @property
    def name(self) -> str:
        """Identifier recorded with cached data."""
        return "twelvedata"

    async def _get(self, path: str, **params: Any) -> dict[str, Any]:
        """Issue a request and surface in-body errors.

        Twelve Data answers HTTP 200 and puts the failure in the payload, so a
        response that looks successful at the transport layer can still be a
        quota rejection. Every call has to inspect the body.
        """
        payload = await self._http.get_json(path, params={**params, "apikey": self._token})
        if not isinstance(payload, dict):
            msg = "Twelve Data returned an unexpected payload"
            raise ExternalServiceError(msg, details={"provider": "twelvedata"})

        if payload.get("status") == "error" or "code" in payload:
            code = payload.get("code")
            message = str(payload.get("message") or "Twelve Data rejected the request")
            if code == _CODE_QUOTA_EXCEEDED:
                raise ProviderQuotaExceededError(
                    message, details={"provider": "twelvedata", "code": code}
                )
            raise ExternalServiceError(message, details={"provider": "twelvedata", "code": code})
        return payload

    # --- Candles -----------------------------------------------------------

    async def get_candles(
        self,
        symbol: str,
        *,
        interval: Interval,
        start: date,
        end: date,
        adjusted: bool = True,
    ) -> CandleSeries:
        """Return OHLCV bars over a window.

        Args:
            symbol: Ticker.
            interval: Bar resolution.
            start: Inclusive first session.
            end: Inclusive last session.
            adjusted: Requested, but see the note below.

        Notes:
            The free plan returns split-adjusted prices and offers no switch, so
            ``adjusted=False`` cannot be honoured. Rather than silently ignoring
            the argument, the returned series records ``adjusted=True`` -- which
            is what the data actually is. A consumer that must know can check
            the flag instead of assuming it got what it asked for.
        """
        upper = symbol.upper()
        resolution = _INTERVALS.get(interval)
        if resolution is None:  # pragma: no cover - the mapping is exhaustive
            msg = f"Twelve Data does not support the {interval.value} interval"
            raise ExternalServiceError(msg, details={"interval": interval.value})

        payload = await self._get(
            "/time_series",
            symbol=upper,
            interval=resolution,
            start_date=start.isoformat(),
            # Exclusive, verified against the API: end_date=2026-07-30
            # returns bars through 07-29 only. Passing the caller's date
            # straight through drops the most recent session from every
            # chart -- which presents as "today's data hasn't arrived yet"
            # rather than as a bug, so it would have survived a long time.
            end_date=(end + timedelta(days=1)).isoformat(),
            outputsize=_MAX_OUTPUT_SIZE,
            order="ASC",
        )

        rows = payload.get("values")
        if not isinstance(rows, list):
            return CandleSeries(symbol=upper, interval=interval, candles=())

        candles = [candle for row in rows if (candle := _to_candle(row)) is not None]
        candles.sort(key=lambda item: item.timestamp)
        logger.info(
            "twelvedata_candles_fetched",
            symbol=upper,
            interval=interval.value,
            candles=len(candles),
        )
        return CandleSeries(
            symbol=upper,
            interval=interval,
            candles=tuple(candles),
            adjusted=True,
        )

    # --- Quote -------------------------------------------------------------

    async def get_quote(self, symbol: str) -> Quote:
        """Return the latest price snapshot.

        Richer than Finnhub's: this one carries average volume and the 52-week
        range in the same response, where Finnhub keeps them on a separate
        metrics endpoint.
        """
        upper = symbol.upper()
        payload = await self._get("/quote", symbol=upper)

        price = _to_decimal(payload.get("close"))
        if price is None:
            msg = f"Twelve Data has no quote for {upper}"
            raise ExternalServiceError(msg, details={"symbol": upper})

        week_52 = payload.get("fifty_two_week")
        week_52 = week_52 if isinstance(week_52, dict) else {}
        timestamp = payload.get("timestamp")

        return Quote(
            symbol=upper,
            timestamp=(
                datetime.fromtimestamp(int(timestamp), tz=UTC) if timestamp else datetime.now(UTC)
            ),
            price=price,
            open=_to_decimal(payload.get("open")),
            high=_to_decimal(payload.get("high")),
            low=_to_decimal(payload.get("low")),
            previous_close=_to_decimal(payload.get("previous_close")),
            volume=_to_int(payload.get("volume")),
            average_volume=_to_int(payload.get("average_volume")),
            week_52_high=_to_decimal(week_52.get("high")),
            week_52_low=_to_decimal(week_52.get("low")),
            # The payload states plainly whether the session is open, so the
            # session is read rather than inferred from a clock -- which would
            # be wrong on every holiday and half-day.
            session=(
                MarketSession.REGULAR if payload.get("is_market_open") else MarketSession.CLOSED
            ),
            currency=str(payload.get("currency") or "USD"),
        )

    async def aclose(self) -> None:
        """Release the transport."""
        await self._http.aclose()


def _to_candle(row: Any) -> Candle | None:
    """Convert one row into a bar, or ``None`` when it is unusable.

    A single malformed row is dropped rather than failing the series: a gap in
    a five-year chart is a missing point, while an exception is a blank page.
    """
    if not isinstance(row, dict):
        return None

    timestamp = _to_datetime(row.get("datetime"))
    open_ = _to_decimal(row.get("open"))
    high = _to_decimal(row.get("high"))
    low = _to_decimal(row.get("low"))
    close = _to_decimal(row.get("close"))
    if timestamp is None or None in (open_, high, low, close):
        return None

    return Candle(
        timestamp=timestamp,
        open=open_,  # type: ignore[arg-type]
        high=high,  # type: ignore[arg-type]
        low=low,  # type: ignore[arg-type]
        close=close,  # type: ignore[arg-type]
        volume=_to_int(row.get("volume")) or 0,
    )


def _to_datetime(value: Any) -> datetime | None:
    """Parse a bar timestamp.

    Daily bars carry a date ("2026-07-30"); intraday bars carry a datetime
    ("2026-07-30 15:59:00"). Both are exchange-local and naive, so a timezone is
    attached rather than left off -- a naive timestamp compared against an
    aware one raises, and comparing bars to news timestamps is exactly what the
    correlation engine does.
    """
    if not value:
        return None
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.combine(date.fromisoformat(text[:10]), datetime.min.time())
        except ValueError:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _to_decimal(value: Any) -> Decimal | None:
    """Convert to an exact Decimal, or None when unusable.

    Twelve Data sends numbers as strings, so this is a direct string-to-Decimal
    construction -- the value never becomes a float, and nothing is rounded.
    """
    if value is None or value == "":
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return result if result.is_finite() else None


def _to_int(value: Any) -> int | None:
    """Convert to int, or None when unusable."""
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None
