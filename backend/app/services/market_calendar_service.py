"""Derive each exchange's trading calendar from the prices already ingested.

Without a calendar, a holiday is indistinguishable from missing data. That
ambiguity is not academic: a four-day weekend looks like a collapse in trading
activity, and the anomaly detectors would dutifully report it. The tracked
universe spans NASDAQ, NYSE, TWSE and KRX, which do not share holidays.

The calendar is *derived from observed prices* rather than fetched from a
calendar library or hardcoded. That keeps the platform dependency-free here and,
more importantly, self-consistent: the calendar describes the sessions the
platform actually has data for, which is exactly the question its consumers ask.

The tradeoff is stated plainly because it matters. A date on which every listing
of an exchange failed to ingest is indistinguishable from a holiday. Two things
bound the damage: a single successful listing is enough to mark a session, and
only the span between the first and last observed date is filled in, so nothing
is asserted about dates the platform has never seen.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

from app.core.logging import get_logger
from app.repositories.market import MarketCalendarRepository
from app.repositories.price import DailyPriceRepository

logger = get_logger(__name__)

#: Weekday numbers ISO-8601 assigns to Saturday and Sunday.
_WEEKEND = frozenset({6, 7})


@dataclass(frozen=True)
class CalendarRebuildReport:
    """Outcome of rebuilding the calendar."""

    exchanges: int
    trading_days: int
    non_trading_days: int
    rows_written: int


class MarketCalendarService:
    """Builds and refreshes the per-exchange session calendar."""

    def __init__(
        self,
        *,
        prices: DailyPriceRepository,
        calendar: MarketCalendarRepository,
    ) -> None:
        """Wire the service to its repositories.

        Args:
            prices: Source of the observed sessions.
            calendar: Destination repository.
        """
        self._prices = prices
        self._calendar = calendar

    async def rebuild(self) -> CalendarRebuildReport:
        """Rebuild every exchange's calendar from observed price data.

        Returns:
            Counts of what was written.

        Notes:
            Idempotent: the repository upserts on ``(exchange, session_date)``,
            so a rebuild converges rather than duplicating. Weekends are recorded
            explicitly as non-trading rather than omitted, so a consumer can
            distinguish "known closed" from "unknown".
        """
        observed = await self._prices.get_sessions_by_exchange()
        if not observed:
            logger.warning("calendar_rebuild_skipped", reason="no prices ingested yet")
            return CalendarRebuildReport(0, 0, 0, 0)

        sessions_by_exchange: dict[str, set[date]] = defaultdict(set)
        for exchange, session_date in observed:
            sessions_by_exchange[exchange].add(session_date)

        rows: list[dict[str, object]] = []
        trading = 0
        non_trading = 0

        for exchange, trading_days in sorted(sessions_by_exchange.items()):
            for session_date, is_trading, note in self._span(exchange, trading_days):
                rows.append(
                    {
                        "exchange": exchange,
                        "session_date": session_date,
                        "is_trading_day": is_trading,
                        "note": note,
                    }
                )
                if is_trading:
                    trading += 1
                else:
                    non_trading += 1

        written = await self._calendar.bulk_upsert(rows)
        report = CalendarRebuildReport(
            exchanges=len(sessions_by_exchange),
            trading_days=trading,
            non_trading_days=non_trading,
            rows_written=written,
        )
        logger.info(
            "calendar_rebuilt",
            exchanges=report.exchanges,
            trading_days=report.trading_days,
            non_trading_days=report.non_trading_days,
        )
        return report

    @staticmethod
    def _span(exchange: str, trading_days: set[date]) -> list[tuple[date, bool, str | None]]:
        """Classify every calendar day between an exchange's first and last session.

        Args:
            exchange: Exchange code, used only in the note text.
            trading_days: Dates on which at least one listing traded.

        Returns:
            One ``(date, is_trading, note)`` triple per day in the span.

        Notes:
            A weekday with no observed session is annotated as a *probable*
            holiday, not asserted as one. The platform cannot distinguish a
            genuine closure from a day every provider dropped, and a note that
            says so is more useful than a confident guess -- particularly for
            2026-07-28, a Tuesday absent from all twelve US listings.
        """
        first, last = min(trading_days), max(trading_days)
        classified: list[tuple[date, bool, str | None]] = []

        current = first
        while current <= last:
            if current in trading_days:
                classified.append((current, True, None))
            elif current.isoweekday() in _WEEKEND:
                classified.append((current, False, "weekend"))
            else:
                classified.append(
                    (current, False, f"no session observed on {exchange}; probable holiday")
                )
            current += timedelta(days=1)

        return classified
