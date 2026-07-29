"""Repository for detected anomalies."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

from sqlalchemy import Row, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models.anomaly import Anomaly
from app.models.company import Ticker
from app.models.enums import AnomalyType, Severity
from app.repositories.base import BaseRepository


class AnomalyRepository(BaseRepository[Anomaly, int]):
    """Reads and idempotent writes over detected anomalies."""

    model = Anomaly

    #: Refreshed on conflict. Re-running a detector over a window must replace
    #: its previous verdict -- including clearing an explanation that a later
    #: correlation pass will rewrite -- not append a second row.
    _UPSERT_COLUMNS = (
        "direction",
        "severity",
        "score",
        "confidence",
        "observed_value",
        "expected_value",
        "deviation",
        "explanation",
        "related_document_ids",
        "context",
        "detected_at",
    )

    async def bulk_upsert(self, rows: Sequence[dict[str, Any]]) -> int:
        """Insert anomalies, refreshing any already recorded.

        Args:
            rows: Mappings keyed by ``ticker_id``, ``trade_date``,
                ``anomaly_type`` and ``method``.

        Returns:
            The number of rows inserted or updated.
        """
        if not rows:
            return 0

        statement = pg_insert(Anomaly).values(list(rows))
        statement = statement.on_conflict_do_update(
            index_elements=[
                Anomaly.ticker_id,
                Anomaly.trade_date,
                Anomaly.anomaly_type,
                Anomaly.method,
            ],
            set_={column: statement.excluded[column] for column in self._UPSERT_COLUMNS},
        )
        return await self._execute_dml(statement)

    async def list_recent(
        self,
        *,
        start: date,
        end: date,
        min_severity: Severity | None = None,
        anomaly_type: AnomalyType | None = None,
        limit: int = 100,
    ) -> Sequence[Row[tuple[Anomaly, str]]]:
        """Return anomalies across all tickers in a window, most recent first.

        Powers the anomalies feed. The symbol is joined in rather than loaded
        per row, so rendering the feed costs one query regardless of its length.

        Args:
            start: Inclusive start of the window.
            end: Inclusive end of the window.
            min_severity: Drop anomalies below this severity.
            anomaly_type: Restrict to one kind of anomaly.
            limit: Maximum rows returned.
        """
        statement = (
            select(Anomaly, Ticker.symbol)
            .join(Ticker, Ticker.id == Anomaly.ticker_id)
            .where(Anomaly.trade_date >= start, Anomaly.trade_date <= end)
        )
        if min_severity is not None:
            statement = statement.where(Anomaly.severity.in_(self._at_least(min_severity)))
        if anomaly_type is not None:
            statement = statement.where(Anomaly.anomaly_type == anomaly_type)

        statement = statement.order_by(Anomaly.trade_date.desc(), Anomaly.confidence.desc()).limit(
            limit
        )
        result = await self._session.execute(statement)
        return result.all()

    async def list_for_ticker(self, ticker_id: int, *, start: date, end: date) -> Sequence[Anomaly]:
        """Return one ticker's anomalies in a window, most recent first."""
        result = await self._session.execute(
            select(Anomaly)
            .where(
                Anomaly.ticker_id == ticker_id,
                Anomaly.trade_date >= start,
                Anomaly.trade_date <= end,
            )
            .order_by(Anomaly.trade_date.desc())
        )
        return result.scalars().all()

    async def list_unexplained(self, *, limit: int = 50) -> Sequence[Anomaly]:
        """Return anomalies the correlation engine has not yet written up.

        The work queue for the news-correlation stage of the pipeline.
        """
        result = await self._session.execute(
            select(Anomaly)
            .where(Anomaly.explanation.is_(None))
            .order_by(Anomaly.trade_date.desc(), Anomaly.confidence.desc())
            .limit(limit)
        )
        return result.scalars().all()

    async def attach_explanation(
        self,
        anomaly_id: int,
        *,
        explanation: str,
        document_ids: Sequence[str],
    ) -> None:
        """Record the correlation engine's verdict for one anomaly.

        Args:
            anomaly_id: Anomaly to annotate.
            explanation: Human-readable narrative citing the documents.
            document_ids: MongoDB ``_id`` values of the cited articles.
        """
        anomaly = await self.get_or_raise(anomaly_id)
        anomaly.explanation = explanation
        anomaly.related_document_ids = list(document_ids)

    @staticmethod
    def _at_least(severity: Severity) -> tuple[Severity, ...]:
        """Return every severity at or above ``severity``.

        The enum is stored as a PostgreSQL ENUM, whose declaration order gives
        it a native ordering -- but relying on that would couple the query to
        the order members happen to be declared in. An explicit ladder keeps the
        comparison meaningful if a member is ever inserted in the middle.
        """
        ladder = (Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.EXTREME)
        return ladder[ladder.index(severity) :]
