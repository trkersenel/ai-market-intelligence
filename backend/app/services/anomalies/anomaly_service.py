"""Anomaly detection orchestration.

Owns the policy: which sessions to score, how much baseline to establish, which
detectors to run, and what to do when a listing has too little history. The
statistics live in :mod:`app.services.anomalies.detectors` and persistence in the
repository.

The calendar is consulted before anything else. A session absent from an
exchange's calendar is not a session with zero activity -- it is a day the market
was shut, and reporting it as a volume anomaly is the single most obvious way for
this feature to lose an analyst's trust.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.models.price import TechnicalIndicator
from app.repositories.anomaly import AnomalyRepository
from app.repositories.company import TickerRepository
from app.repositories.market import MarketCalendarRepository
from app.repositories.price import TechnicalIndicatorRepository
from app.services.anomalies.detectors import (
    MIN_HISTORY_SESSIONS,
    Detection,
    IsolationForestDetector,
    Observation,
    ZScoreDetector,
)

logger = get_logger(__name__)

#: Sessions of history loaded to establish a baseline. Two years: long enough
#: that a single volatile quarter cannot define "normal", short enough that a
#: regime from three years ago does not suppress today's genuine outliers.
BASELINE_SESSIONS = 500

#: Sessions reported on a routine run. Older anomalies are already stored, and
#: re-running the detectors over them would only rewrite identical rows.
DEFAULT_LOOKBACK_SESSIONS = 30


@dataclass(frozen=True)
class TickerAnomalyResult:
    """Outcome of running the detectors over one listing."""

    symbol: str
    detections: int
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether detection completed without an error."""
        return self.error is None


@dataclass(frozen=True)
class AnomalyRunReport:
    """Aggregate outcome of one detection run."""

    started_at: datetime
    finished_at: datetime
    results: tuple[TickerAnomalyResult, ...]

    @property
    def detections(self) -> int:
        """Total anomalies written across every listing."""
        return sum(result.detections for result in self.results)

    @property
    def failures(self) -> tuple[TickerAnomalyResult, ...]:
        """Listings whose detection failed."""
        return tuple(result for result in self.results if not result.succeeded)

    @property
    def duration_seconds(self) -> float:
        """Wall-clock duration of the run."""
        return (self.finished_at - self.started_at).total_seconds()


class AnomalyDetectionService:
    """Runs both detectors across the tracked universe and stores the results."""

    def __init__(
        self,
        *,
        tickers: TickerRepository,
        indicators: TechnicalIndicatorRepository,
        anomalies: AnomalyRepository,
        calendar: MarketCalendarRepository,
        z_score: ZScoreDetector | None = None,
        isolation_forest: IsolationForestDetector | None = None,
        lookback_sessions: int = DEFAULT_LOOKBACK_SESSIONS,
    ) -> None:
        """Wire the service to its collaborators.

        Args:
            tickers: Listing repository, read for the work queue.
            indicators: Source of the features the detectors consume.
            anomalies: Destination repository.
            calendar: Consulted to skip non-trading sessions.
            z_score: Univariate detector. Constructed with defaults if omitted.
            isolation_forest: Multivariate detector. Constructed if omitted.
            lookback_sessions: How many recent sessions each run reports on.
        """
        self._tickers = tickers
        self._indicators = indicators
        self._anomalies = anomalies
        self._calendar = calendar
        self._z_score = z_score or ZScoreDetector()
        self._isolation_forest = isolation_forest or IsolationForestDetector()
        self._lookback = lookback_sessions

    async def detect_all(self, *, lookback_sessions: int | None = None) -> AnomalyRunReport:
        """Run both detectors over every active listing.

        Args:
            lookback_sessions: Override the reporting window. Pass a large value
                to re-scan history after changing a threshold.

        Returns:
            A report naming every listing processed and every failure.
        """
        started = datetime.now(UTC)
        listings = await self._tickers.list_active()

        results: list[TickerAnomalyResult] = []
        for listing in listings:
            results.append(
                await self._detect_one(
                    ticker_id=listing.id,
                    symbol=listing.symbol,
                    exchange=listing.exchange,
                    lookback=lookback_sessions or self._lookback,
                )
            )

        report = AnomalyRunReport(
            started_at=started, finished_at=datetime.now(UTC), results=tuple(results)
        )
        logger.info(
            "anomaly_run_complete",
            listings=len(listings),
            detections=report.detections,
            failures=len(report.failures),
            duration_seconds=round(report.duration_seconds, 2),
        )
        return report

    async def detect_symbol(
        self, symbol: str, *, lookback_sessions: int | None = None
    ) -> TickerAnomalyResult:
        """Run both detectors over one listing.

        Raises:
            NotFoundError: If the symbol is not tracked.
        """
        listing = await self._tickers.get_by_symbol(symbol)
        if listing is None:
            msg = f"Ticker {symbol.upper()!r} is not tracked."
            raise NotFoundError(msg, details={"symbol": symbol.upper()})

        return await self._detect_one(
            ticker_id=listing.id,
            symbol=listing.symbol,
            exchange=listing.exchange,
            lookback=lookback_sessions or self._lookback,
        )

    async def _detect_one(
        self, *, ticker_id: int, symbol: str, exchange: str | None, lookback: int
    ) -> TickerAnomalyResult:
        """Load features, run both detectors, and persist what they found."""
        rows = await self._indicators.get_recent(ticker_id, sessions=BASELINE_SESSIONS)
        if len(rows) < MIN_HISTORY_SESSIONS:
            logger.info("anomaly_detection_skipped", symbol=symbol, sessions=len(rows))
            return TickerAnomalyResult(symbol=symbol, detections=0)

        trading_days = await self._trading_days(exchange, rows)
        observations = [
            self._to_observation(row)
            for row in rows
            if trading_days is None or row.trade_date in trading_days
        ]
        if len(observations) < MIN_HISTORY_SESSIONS:
            return TickerAnomalyResult(symbol=symbol, detections=0)

        since = observations[-min(lookback, len(observations))].trade_date
        detections = [
            *self._z_score.detect(observations, since=since),
            *self._isolation_forest.detect(observations, since=since),
        ]
        if not detections:
            return TickerAnomalyResult(symbol=symbol, detections=0)

        written = await self._anomalies.bulk_upsert(
            [self._to_row(ticker_id, detection) for detection in detections]
        )
        return TickerAnomalyResult(symbol=symbol, detections=written)

    async def _trading_days(
        self, exchange: str | None, rows: Sequence[TechnicalIndicator]
    ) -> set[date] | None:
        """Return the exchange's trading sessions over the loaded window.

        Returns ``None`` when no calendar exists for the exchange, in which case
        every session is used. Degrading to "trust the data" is right here: an
        unpopulated calendar must not silently suppress all detection.
        """
        if exchange is None or not rows:
            return None

        sessions = await self._calendar.list_trading_days(
            exchange, start=rows[0].trade_date, end=rows[-1].trade_date
        )
        if not sessions:
            logger.debug("calendar_absent", exchange=exchange)
            return None
        return set(sessions)

    @staticmethod
    def _to_observation(row: TechnicalIndicator) -> Observation:
        """Convert a stored feature row into the detectors' input shape."""
        return Observation(
            trade_date=row.trade_date,
            daily_return=_as_float(row.daily_return),
            volume_ratio=_as_float(row.volume_ratio),
            volatility_20=_as_float(row.volatility_20),
            relative_strength=_as_float(row.relative_strength_smh),
        )

    @staticmethod
    def _to_row(ticker_id: int, detection: Detection) -> dict[str, Any]:
        """Flatten a detection into the mapping the repository upserts."""
        return {
            "ticker_id": ticker_id,
            "trade_date": detection.trade_date,
            "anomaly_type": detection.anomaly_type,
            "method": detection.method,
            "direction": detection.direction,
            "severity": detection.severity,
            "score": detection.score,
            "confidence": detection.confidence,
            "observed_value": _as_decimal(detection.observed_value),
            "expected_value": _as_decimal(detection.expected_value),
            "deviation": _as_decimal(detection.deviation),
            "context": detection.context,
            "detected_at": datetime.now(UTC),
        }


def _as_float(value: Decimal | None) -> float | None:
    """Convert a stored NUMERIC to the float the detectors work in."""
    return None if value is None else float(value)


def _as_decimal(value: float | None) -> Decimal | None:
    """Convert a detector float back to the exact type the column stores."""
    if value is None:
        return None
    converted = Decimal(str(value))
    return converted.quantize(Decimal("0.000001")) if converted.is_finite() else None


def default_since(sessions: int) -> date:
    """Return a conservative calendar-date floor for ``sessions`` trading days.

    Trading days are roughly 5 in 7, so the calendar span is padded rather than
    computed exactly -- a floor that is slightly too early costs one redundant
    upsert, while one that is too late silently drops sessions.
    """
    return datetime.now(UTC).date() - timedelta(days=int(sessions * 1.6) + 7)
