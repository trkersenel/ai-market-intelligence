"""Explain detected anomalies by correlating them with news.

This is the feature the whole platform exists to deliver: a price move on its
own is a number, and the product is the sentence that says why.

Two design choices carry most of the weight.

**The time window is asymmetric.** News published before or during a session can
have caused the move; news published after it is mostly reaction to the move --
"NVIDIA falls 8%" written at 21:00 explains nothing about why it fell. Ranking
them equally would let the platform confidently cite an article that was
literally written *because* of the thing it claims to explain. The lookback is
therefore days and the lookahead hours, kept only because publication timestamps
are unreliable and a genuinely causal story can be stamped slightly late.

**Correlation is never called causation.** The output is ranked candidate
explanations with their evidence, phrased as possibilities. The platform cannot
establish causation from timing and text overlap, and language that implies it
would be the most damaging kind of wrong -- confident, plausible, and unfounded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

from app.core.logging import get_logger
from app.models.anomaly import Anomaly
from app.models.enums import Direction, Sentiment
from app.repositories.anomaly import AnomalyRepository
from app.repositories.company import TickerRepository
from app.repositories.documents import NewsRepository
from app.schemas.documents import NewsArticle

logger = get_logger(__name__)

#: Weights for the ranking signals.
#:
#: Attribution is split into two signals because they are not equally strong,
#: which a live run made obvious: Yahoo's per-ticker feed also carries general
#: market news, so an article can arrive stamped "000660.KS" while being about
#: US crypto regulation. Treating the feed stamp as proof produced explanations
#: citing the CLARITY Act as a possible cause of SK Hynix volume.
#:
#: The stamp is still evidence -- the source thought it relevant to that ticker --
#: but text that actually names the company is much stronger, and the two are
#: reported separately so a reader can tell which they are looking at.
_WEIGHT_SOURCE_ATTRIBUTION = 0.20
_WEIGHT_NAMED_IN_TEXT = 0.30
_WEIGHT_RECENCY = 0.25
_WEIGHT_SENTIMENT = 0.20
_WEIGHT_TAG_OVERLAP = 0.05

#: Candidate articles considered per anomaly before ranking.
DEFAULT_CANDIDATES = 40

#: Explanations kept. More than a handful stops being an explanation and starts
#: being a reading list.
DEFAULT_EXPLANATIONS = 3

#: Shortest company-name token treated as distinctive. Two-letter fragments
#: match too much prose to be evidence of anything.
_MIN_NAME_TOKEN_CHARS = 3


@dataclass(frozen=True)
class CandidateExplanation:
    """One article proposed as a possible cause, with its evidence."""

    article: NewsArticle
    score: float
    #: The source filed this article under the ticker. Suggestive, not proof:
    #: per-ticker feeds also carry general market news.
    source_attributed: bool
    #: The article text actually names the company. Much stronger.
    named_in_text: bool
    sentiment_agrees: bool
    hours_before: float

    @property
    def ticker_match(self) -> bool:
        """Whether either form of attribution applies."""
        return self.source_attributed or self.named_in_text

    @property
    def headline(self) -> str:
        """The article's title."""
        return self.article.title


@dataclass(frozen=True)
class CorrelationResult:
    """The explanation attached to one anomaly."""

    anomaly_id: int
    symbol: str
    explanation: str
    candidates: tuple[CandidateExplanation, ...]

    @property
    def document_ids(self) -> tuple[str, ...]:
        """MongoDB ids of the cited articles."""
        return tuple(
            candidate.article.id
            for candidate in self.candidates
            if candidate.article.id is not None
        )


class CorrelationEngine:
    """Ranks news against an anomaly and writes a human-readable explanation."""

    def __init__(
        self,
        *,
        news: NewsRepository,
        anomalies: AnomalyRepository,
        tickers: TickerRepository,
        lookback_hours: int = 72,
        lookahead_hours: int = 24,
        max_explanations: int = DEFAULT_EXPLANATIONS,
    ) -> None:
        """Wire the engine to its collaborators.

        Args:
            news: Source of candidate articles.
            anomalies: Read for the work queue and written with explanations.
            tickers: Resolves an anomaly's ticker to its symbol.
            lookback_hours: How far before the session news may have caused it.
            lookahead_hours: How far after, kept small -- see the module
                docstring on why the window is asymmetric.
            max_explanations: Candidates retained per anomaly.
        """
        self._news = news
        self._anomalies = anomalies
        self._tickers = tickers
        self._lookback = timedelta(hours=lookback_hours)
        self._lookahead = timedelta(hours=lookahead_hours)
        self._max_explanations = max_explanations

    async def explain_pending(self, *, limit: int = 50) -> list[CorrelationResult]:
        """Explain anomalies that have no narrative yet.

        The work queue is the absence of an explanation, so a run that fails
        part-way simply leaves less for the next one.
        """
        pending = await self._anomalies.list_unexplained(limit=limit)
        results: list[CorrelationResult] = []

        for anomaly in pending:
            result = await self.explain(anomaly)
            if result is not None:
                results.append(result)

        logger.info("correlation_run_complete", pending=len(pending), explained=len(results))
        return results

    async def explain(self, anomaly: Anomaly) -> CorrelationResult | None:
        """Rank news against one anomaly and store the explanation.

        Returns ``None`` when no candidate article is plausible enough to cite.
        Writing "no explanation found" would be worse than writing nothing: the
        absence of an explanation is information, and a placeholder would keep
        the anomaly out of the queue on every subsequent run, so it would never
        be revisited once the news arrived.
        """
        ticker = await self._tickers.get(anomaly.ticker_id)
        if ticker is None:  # pragma: no cover - foreign key guarantees this
            return None

        session_close = self._session_close(anomaly.trade_date)
        window_start, window_end = self._window(anomaly.trade_date)
        articles = await self._news.list_recent(
            since=window_start, until=window_end, limit=DEFAULT_CANDIDATES
        )
        if not articles:
            return None

        # Anchored to the close, NOT to the window end. Measuring from the
        # window end would shift every article by the lookahead, making a
        # reaction piece published after the close look closer to the event than
        # the news that actually preceded it.
        candidates = self._rank(
            articles, anomaly, ticker.symbol, ticker.display_name, session_close
        )
        if not candidates:
            return None

        explanation = self._compose(anomaly, ticker.symbol, candidates)
        await self._anomalies.attach_explanation(
            anomaly.id,
            explanation=explanation,
            document_ids=[
                candidate.article.id for candidate in candidates if candidate.article.id is not None
            ],
        )
        return CorrelationResult(
            anomaly_id=anomaly.id,
            symbol=ticker.symbol,
            explanation=explanation,
            candidates=tuple(candidates),
        )

    @staticmethod
    def _session_close(trade_date: date) -> datetime:
        """Return the moment the session closed.

        21:00 UTC, the US close. Approximate for the Asian listings, but the
        lookback is measured in days, so a few hours of drift cannot change
        which articles fall inside the window.
        """
        return datetime.combine(trade_date, time(21, 0), tzinfo=UTC)

    def _window(self, trade_date: date) -> tuple[datetime, datetime]:
        """Return the asymmetric time window around a session."""
        close = self._session_close(trade_date)
        return (close - self._lookback, close + self._lookahead)

    def _rank(
        self,
        articles: list[NewsArticle],
        anomaly: Anomaly,
        symbol: str,
        display_name: str,
        session_close: datetime,
    ) -> list[CandidateExplanation]:
        """Score and order candidate articles."""
        scored: list[CandidateExplanation] = []

        name_tokens = self._name_tokens(display_name)

        for article in articles:
            source_attributed = symbol.upper() in {t.upper() for t in article.tickers}
            named_in_text = self._names_company(article, symbol, name_tokens)
            hours_before = (session_close - article.published_at).total_seconds() / 3600
            sentiment_agrees = self._sentiment_agrees(article, anomaly.direction)

            score = (
                _WEIGHT_SOURCE_ATTRIBUTION * (1.0 if source_attributed else 0.0)
                + _WEIGHT_NAMED_IN_TEXT * (1.0 if named_in_text else 0.0)
                + _WEIGHT_RECENCY * self._recency(hours_before)
                + _WEIGHT_SENTIMENT * (1.0 if sentiment_agrees else 0.0)
                + _WEIGHT_TAG_OVERLAP * (1.0 if article.tags else 0.0)
            )

            # An article about neither this company nor this sector is not
            # evidence, however well its wording happens to score.
            if not source_attributed and not named_in_text and not article.tags:
                continue

            scored.append(
                CandidateExplanation(
                    article=article,
                    score=round(score, 4),
                    source_attributed=source_attributed,
                    named_in_text=named_in_text,
                    sentiment_agrees=sentiment_agrees,
                    hours_before=round(hours_before, 1),
                )
            )

        scored.sort(key=lambda candidate: candidate.score, reverse=True)
        return scored[: self._max_explanations]

    def _recency(self, hours_before: float) -> float:
        """Score proximity to the session, penalising news published after it.

        Decays linearly across the lookback window. News from *after* the close
        scores at most half, because it is far more likely to be reporting the
        move than to have caused it.
        """
        lookback_hours = self._lookback.total_seconds() / 3600
        if hours_before < 0:
            return 0.5 * max(0.0, 1.0 + hours_before / max(lookback_hours, 1))
        return max(0.0, 1.0 - hours_before / max(lookback_hours, 1))

    @staticmethod
    def _sentiment_agrees(article: NewsArticle, direction: Direction) -> bool:
        """Whether the article's sentiment corroborates the move's direction.

        Agreement is corroborating evidence, not a requirement: a stock can fall
        on good news, and treating disagreement as disqualifying would discard
        the most interesting cases the platform could surface.
        """
        if article.sentiment is None:
            return False
        if direction is Direction.UP:
            return article.sentiment.label is Sentiment.BULLISH
        if direction is Direction.DOWN:
            return article.sentiment.label is Sentiment.BEARISH
        return False

    @staticmethod
    def _name_tokens(display_name: str) -> frozenset[str]:
        """Distinctive words from a listing's name, for matching against prose.

        Corporate suffixes are dropped: "Inc" and "Co" appear in half the
        financial press and would match everything.
        """
        noise = {"inc", "inc.", "corp", "corp.", "co", "co.", "ltd", "ltd.",
                 "plc", "nv", "n.v.", "sa", "ag", "holding", "holdings",
                 "company", "technology", "technologies", "electronics",
                 "adr", "the"}  # fmt: skip
        words = re.findall(r"[a-z0-9]+", display_name.lower())
        return frozenset(
            word for word in words if word not in noise and len(word) >= _MIN_NAME_TOKEN_CHARS
        )

    @staticmethod
    def _names_company(article: NewsArticle, symbol: str, name_tokens: frozenset[str]) -> bool:
        """Whether the article text actually names the company or its symbol.

        The distinction that matters: a Korean listing's symbol ("000660.KS")
        never appears in English prose, so matching only on the symbol would
        mark every Asian company as unnamed and leave the feed stamp as the sole
        evidence -- which is exactly how a crypto-regulation story ended up
        cited as a cause of SK Hynix volume.
        """
        haystack = f"{article.title} {article.summary or ''}".lower()
        if symbol.lower() in haystack:
            return True
        return any(token in haystack for token in name_tokens)

    def _compose(
        self, anomaly: Anomaly, symbol: str, candidates: list[CandidateExplanation]
    ) -> str:
        """Write the human-readable explanation.

        Deliberately hedged. "Possible contributing factors" rather than
        "caused by": the engine has established temporal proximity and topical
        overlap, which is not causation, and prose implying otherwise would be
        confidently wrong in exactly the way that destroys trust.
        """
        move = _describe_move(anomaly)
        lines = [f"{symbol} {move} on {anomaly.trade_date.isoformat()}."]

        # When nothing in the corpus predates the close, every candidate is
        # coverage of the move rather than a possible cause. Presenting reaction
        # as explanation is the specific failure this engine is built to avoid,
        # so the framing changes rather than the ranking being quietly trusted.
        preceding = [c for c in candidates if c.hours_before >= 0]
        if preceding:
            lines.append("Possible contributing factors, ranked by relevance:")
        else:
            lines.append(
                "No news published before the close was found in the corpus. "
                "The following covered the move afterwards and may describe, "
                "but cannot explain, it:"
            )

        for candidate in candidates:
            markers: list[str] = []
            if candidate.named_in_text:
                markers.append("names this company")
            elif candidate.source_attributed:
                # Deliberately hedged: the feed filed it under this ticker, which
                # is not the same as the article being about the company.
                markers.append("filed under this ticker by the source")
            if candidate.sentiment_agrees:
                markers.append("sentiment agrees with the move")
            timing = (
                f"{candidate.hours_before:.0f}h before the close"
                if candidate.hours_before >= 0
                else f"{abs(candidate.hours_before):.0f}h after the close"
            )
            markers.append(timing)
            lines.append(f"  - {candidate.headline} ({'; '.join(markers)})")

        if preceding:
            # Omitted in the reaction-only case, where "cannot explain" above is
            # the stronger statement and this would read as a softer retraction
            # of it.
            lines.append("These are correlations in time and topic, not established causes.")
        return "\n".join(lines)


def _describe_move(anomaly: Anomaly) -> str:
    """Phrase the anomaly in words an analyst would use."""
    magnitude = (
        f" of {float(anomaly.observed_value) * 100:.1f}%"
        if anomaly.observed_value is not None and anomaly.anomaly_type.value == "return"
        else ""
    )
    if anomaly.anomaly_type.value == "volume":
        return "traded on unusual volume"
    if anomaly.anomaly_type.value == "volatility":
        return "showed unusual volatility"
    if anomaly.direction is Direction.UP:
        return f"rose{magnitude}"
    if anomaly.direction is Direction.DOWN:
        return f"fell{magnitude}"
    return "moved unusually"
