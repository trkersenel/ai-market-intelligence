"""Tests for sentiment scoring.

The lexicon analyser is tested on its actual behaviour, including where it is
wrong. Pretending a keyword matcher understands "beats estimates but guides
lower" would make the tests a fiction; asserting that it *doesn't* documents the
limitation the FinBERT path exists to fix.

FinBERT itself is not exercised here -- it would pull 2 GB of torch into CI for
no additional confidence in our code, which is the adapter, not the model.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from app.models.enums import DataSource, Sentiment
from app.schemas.documents import NewsArticle, SentimentScore
from app.services.sentiment import (
    FinBertSentimentAnalyzer,
    LexiconSentimentAnalyzer,
    SentimentAnalyzer,
    build_analyzer,
)
from app.services.sentiment_service import SentimentScoringService

NOW = datetime(2026, 7, 29, tzinfo=UTC)


@pytest.fixture
def analyzer() -> LexiconSentimentAnalyzer:
    """The dependency-free analyser."""
    return LexiconSentimentAnalyzer()


class TestLexiconAnalyzer:
    """Keyword scoring with negation handling."""

    async def test_clearly_bullish_text(self, analyzer: LexiconSentimentAnalyzer) -> None:
        score = await analyzer.score(
            "Micron beats estimates and raises guidance on record HBM demand"
        )

        assert score.label is Sentiment.BULLISH
        assert score.polarity > 0

    async def test_clearly_bearish_text(self, analyzer: LexiconSentimentAnalyzer) -> None:
        score = await analyzer.score(
            "Intel warns of weak demand; shares plunged after the downgrade"
        )

        assert score.label is Sentiment.BEARISH
        assert score.polarity < 0

    async def test_text_with_no_signal_is_neutral(self, analyzer: LexiconSentimentAnalyzer) -> None:
        score = await analyzer.score("TSMC will hold its annual shareholder meeting in June")

        assert score.label is Sentiment.NEUTRAL
        assert score.polarity == 0.0

    async def test_negation_flips_polarity(self, analyzer: LexiconSentimentAnalyzer) -> None:
        """Without this, "demand is not weak" scores bearish."""
        plain = await analyzer.score("DRAM demand is weak this quarter")
        negated = await analyzer.score("DRAM demand is not weak this quarter")

        assert plain.label is Sentiment.BEARISH
        assert negated.polarity > plain.polarity

    async def test_confidence_grows_with_evidence(self, analyzer: LexiconSentimentAnalyzer) -> None:
        """Two matched terms out of two is weaker than twenty out of twenty."""
        thin = await analyzer.score("Micron beat")
        thick = await analyzer.score(
            "Micron beat and raised guidance on record demand; a breakthrough "
            "quarter as shipments surged and profit exceeded every estimate"
        )

        assert thick.confidence > thin.confidence

    async def test_confidence_is_bounded(self, analyzer: LexiconSentimentAnalyzer) -> None:
        score = await analyzer.score("surge " * 100)

        assert 0.0 <= score.confidence <= 1.0

    async def test_polarity_is_bounded(self, analyzer: LexiconSentimentAnalyzer) -> None:
        score = await analyzer.score("plunge slump downgrade warning " * 20)

        assert -1.0 <= score.polarity <= 1.0

    async def test_multi_word_terms_are_matched(self, analyzer: LexiconSentimentAnalyzer) -> None:
        score = await analyzer.score("HBM capacity is sold out through next year")

        assert score.label is Sentiment.BULLISH

    async def test_it_cannot_read_a_contrastive_clause(
        self, analyzer: LexiconSentimentAnalyzer
    ) -> None:
        """The documented limitation the FinBERT path exists to address.

        "Beats estimates but guides lower" is a bearish story. A keyword matcher
        has no syntax, sees one bullish and one bearish term, and calls it a
        wash. Asserting the real behaviour keeps the tradeoff visible instead of
        implying a capability the analyser does not have.
        """
        score = await analyzer.score("Micron beats estimates but cuts guidance")

        assert score.label is Sentiment.NEUTRAL

    async def test_batch_scoring_matches_individual_scoring(
        self, analyzer: LexiconSentimentAnalyzer
    ) -> None:
        texts = ["Micron beats estimates", "Intel warns of weak demand", "A neutral headline"]

        batch = await analyzer.score_many(texts)
        individual = [await analyzer.score(text) for text in texts]

        assert batch == individual

    async def test_the_model_name_is_recorded(self, analyzer: LexiconSentimentAnalyzer) -> None:
        """Provenance: a stored score must say what produced it."""
        score = await analyzer.score("Micron beats estimates")

        assert score.model_name == "lexicon-v1" == analyzer.model_name


class TestAnalyzerSelection:
    """Choosing between the two implementations."""

    def test_lexicon_is_returned_when_finbert_is_not_requested(self) -> None:
        assert isinstance(build_analyzer(prefer_finbert=False), LexiconSentimentAnalyzer)

    def test_a_missing_ml_stack_degrades_rather_than_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 2GB dependency must never be the reason the platform will not start."""
        monkeypatch.setattr(FinBertSentimentAnalyzer, "is_available", staticmethod(lambda: False))

        assert isinstance(build_analyzer(prefer_finbert=True), LexiconSentimentAnalyzer)

    def test_finbert_is_selected_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(FinBertSentimentAnalyzer, "is_available", staticmethod(lambda: True))

        assert isinstance(build_analyzer(prefer_finbert=True), FinBertSentimentAnalyzer)

    def test_both_implementations_satisfy_the_protocol(self) -> None:
        assert isinstance(LexiconSentimentAnalyzer(), SentimentAnalyzer)
        assert isinstance(FinBertSentimentAnalyzer(), SentimentAnalyzer)

    def test_finbert_construction_loads_nothing(self) -> None:
        """Lazy by design: constructing must not pull 440MB of weights."""
        instance = FinBertSentimentAnalyzer()

        assert instance._model is None
        assert instance._tokenizer is None


# --- Scoring service -------------------------------------------------------


class FakeNewsRepository:
    """In-memory stand-in tracking which articles were scored."""

    def __init__(self, articles: Sequence[NewsArticle]) -> None:
        self.articles = list(articles)
        self.stored: dict[str, SentimentScore] = {}

    async def list_unscored(self, *, since: datetime, limit: int = 200) -> list[NewsArticle]:
        return [
            article
            for article in self.articles
            if article.sentiment is None
            and article.url_hash not in self.stored
            and article.published_at >= since
        ][:limit]

    async def set_sentiments(self, updates: Sequence[tuple[str, SentimentScore]]) -> int:
        for url_hash, score in updates:
            self.stored[url_hash] = score
        return len(updates)


class FailingAnalyzer:
    """Analyser that always fails, to exercise the error path."""

    @property
    def model_name(self) -> str:
        return "failing"

    async def score(self, text: str) -> SentimentScore:
        raise RuntimeError(text)

    async def score_many(self, texts: Sequence[str]) -> list[SentimentScore]:
        msg = "model unavailable"
        raise RuntimeError(msg)


def _article(title: str, *, published_at: datetime | None = None) -> NewsArticle:
    return NewsArticle(
        url_hash=f"hash-{abs(hash(title))}",
        url=f"https://news.test/{abs(hash(title))}",
        title=title,
        source=DataSource.RSS,
        published_at=published_at or NOW - timedelta(hours=2),
        ingested_at=NOW - timedelta(hours=1),
    )


class TestSentimentScoringService:
    """Batch scoring policy."""

    async def test_unscored_articles_are_scored_and_stored(self) -> None:
        news = FakeNewsRepository(
            [_article("Micron beats estimates"), _article("Intel warns of weak demand")]
        )
        service = SentimentScoringService(
            analyzer=LexiconSentimentAnalyzer(),
            news=news,  # type: ignore[arg-type]
        )

        report = await service.score_pending()

        assert report.succeeded
        assert report.scored == 2
        assert len(news.stored) == 2

    async def test_the_pass_is_resumable(self) -> None:
        """Absence of a score is the queue: a second run finds nothing to do."""
        news = FakeNewsRepository([_article("Micron beats estimates")])
        service = SentimentScoringService(
            analyzer=LexiconSentimentAnalyzer(),
            news=news,  # type: ignore[arg-type]
        )

        first = await service.score_pending()
        second = await service.score_pending()

        assert first.scored == 1
        assert second.scored == 0

    async def test_articles_older_than_the_window_are_skipped(self) -> None:
        news = FakeNewsRepository(
            [_article("Ancient news", published_at=NOW - timedelta(days=400))]
        )
        service = SentimentScoringService(
            analyzer=LexiconSentimentAnalyzer(),
            news=news,  # type: ignore[arg-type]
            max_age_days=30,
        )

        report = await service.score_pending()

        assert report.scored == 0

    async def test_the_batch_limit_is_respected(self) -> None:
        news = FakeNewsRepository([_article(f"Headline {index}") for index in range(50)])
        service = SentimentScoringService(
            analyzer=LexiconSentimentAnalyzer(),
            news=news,  # type: ignore[arg-type]
            batch_limit=10,
        )

        report = await service.score_pending()

        assert report.scored == 10

    async def test_a_model_failure_is_reported_not_raised(self) -> None:
        """A failing model must cost sentiment for a batch, never the articles."""
        news = FakeNewsRepository([_article("Micron beats estimates")])
        service = SentimentScoringService(
            analyzer=FailingAnalyzer(),  # type: ignore[arg-type]
            news=news,  # type: ignore[arg-type]
        )

        report = await service.score_pending()

        assert not report.succeeded
        assert report.error is not None
        assert "model unavailable" in report.error
        assert news.stored == {}

    async def test_the_model_name_is_carried_into_the_report(self) -> None:
        news = FakeNewsRepository([_article("Micron beats estimates")])
        service = SentimentScoringService(
            analyzer=LexiconSentimentAnalyzer(),
            news=news,  # type: ignore[arg-type]
        )

        report = await service.score_pending()

        assert report.model_name == "lexicon-v1"
