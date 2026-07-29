"""Score stored news articles and write the results back.

Scoring happens *after* storage, not during ingestion. Two reasons, and both are
about failure isolation: FinBERT inference is orders of magnitude slower than
fetching an RSS feed, and coupling them would let a slow model stall news
ingestion entirely. Separating them also means a model failure loses sentiment
for a batch, never the articles themselves -- which are the irreplaceable part.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.core.logging import get_logger
from app.repositories.documents import NewsRepository
from app.services.sentiment import SentimentAnalyzer

logger = get_logger(__name__)

#: Articles fetched and scored per pass. Bounded so one run cannot pull an
#: unbounded backlog into memory, and so a batch stays small enough that a
#: crash mid-run loses little work.
DEFAULT_BATCH_LIMIT = 200

#: How far back an unscored article is worth scoring. Beyond this the article is
#: unlikely to be cited by the correlation engine, and the inference cost buys
#: nothing.
DEFAULT_MAX_AGE_DAYS = 30


@dataclass(frozen=True)
class SentimentRunReport:
    """Outcome of one scoring pass."""

    started_at: datetime
    finished_at: datetime
    scored: int
    model_name: str
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the pass completed without an error."""
        return self.error is None

    @property
    def duration_seconds(self) -> float:
        """Wall-clock duration of the pass."""
        return (self.finished_at - self.started_at).total_seconds()


class SentimentScoringService:
    """Finds unscored articles, scores them, and stores the verdicts."""

    def __init__(
        self,
        *,
        analyzer: SentimentAnalyzer,
        news: NewsRepository,
        batch_limit: int = DEFAULT_BATCH_LIMIT,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    ) -> None:
        """Wire the service to its collaborators.

        Args:
            analyzer: Anything satisfying :class:`SentimentAnalyzer`.
            news: Document repository holding the articles.
            batch_limit: Articles scored per pass.
            max_age_days: Oldest article worth scoring.
        """
        self._analyzer = analyzer
        self._news = news
        self._batch_limit = batch_limit
        self._max_age = timedelta(days=max_age_days)

    async def score_pending(self, *, limit: int | None = None) -> SentimentRunReport:
        """Score every recent article that has no sentiment yet.

        Returns:
            A report of how many were scored and by which model.

        Notes:
            Only articles *without* a score are fetched, which is what makes the
            pass resumable: a run that dies halfway leaves the articles it
            already scored alone, and the next run picks up exactly where it
            stopped. No cursor or checkpoint is needed -- the absence of a score
            is the work queue.
        """
        started = datetime.now(UTC)
        since = started - self._max_age

        try:
            pending = await self._news.list_unscored(since=since, limit=limit or self._batch_limit)
            if not pending:
                return SentimentRunReport(
                    started_at=started,
                    finished_at=datetime.now(UTC),
                    scored=0,
                    model_name=self._analyzer.model_name,
                )

            texts = [_scoring_text(article.title, article.summary) for article in pending]
            scores = await self._analyzer.score_many(texts)

            updates = [
                (article.url_hash, score) for article, score in zip(pending, scores, strict=True)
            ]
            written = await self._news.set_sentiments(updates)

            report = SentimentRunReport(
                started_at=started,
                finished_at=datetime.now(UTC),
                scored=written,
                model_name=self._analyzer.model_name,
            )
            logger.info(
                "sentiment_run_complete",
                scored=written,
                model=report.model_name,
                duration_seconds=round(report.duration_seconds, 2),
            )
            return report

        except Exception as exc:  # reported in the run summary, never propagated
            logger.exception("sentiment_run_failed")
            return SentimentRunReport(
                started_at=started,
                finished_at=datetime.now(UTC),
                scored=0,
                model_name=self._analyzer.model_name,
                error=f"{type(exc).__name__}: {exc}",
            )


def _scoring_text(title: str, summary: str | None) -> str:
    """Build the passage handed to the analyser.

    Title plus summary, deliberately not the full body. A financial story states
    its thesis in the headline and lede; the remainder is context, quotes and
    boilerplate that dilutes the signal -- and BERT would truncate it at 512
    tokens anyway, so including it mostly means paying to discard it.
    """
    return f"{title}. {summary}" if summary else title
