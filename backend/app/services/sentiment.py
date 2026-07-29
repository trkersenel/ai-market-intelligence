"""Sentiment scoring for financial text.

Two implementations behind one protocol, and the choice between them is a real
engineering tradeoff rather than a preference.

**FinBERT** is BERT fine-tuned on financial text, and it is right for this
domain: general-purpose sentiment models read "Micron beats estimates, shares
fall on guidance" as positive, because they weight "beats" and miss that the
market reaction is the story. FinBERT costs roughly 2 GB of PyTorch and a 440 MB
checkpoint, and inference is CPU-bound.

**The lexicon** analyser costs nothing, loads instantly, and is meaningfully
worse. It exists so the platform runs, and its test suite passes, on a machine
with no ML stack -- and so a torch installation failing in CI degrades one
column rather than breaking the build.

The default is the lexicon precisely because a system that quietly requires a
2 GB download to start is a system that will not start. FinBERT is selected
explicitly, in configuration, and only the worker needs it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import ClassVar, Protocol, runtime_checkable

from app.core.logging import get_logger
from app.models.enums import Sentiment
from app.schemas.documents import SentimentScore

logger = get_logger(__name__)

#: Terms that carry directional meaning in market commentary. Deliberately
#: domain-specific: "beat", "guidance" and "sold out" mean something here that
#: they do not mean in general English, and a general-purpose word list scores
#: financial text close to randomly.
#:
#: Bare nouns that need a modifier to mean anything -- "demand", "profit",
#: "guidance" -- are deliberately absent. A unigram lexicon has no way to tell
#: "strong demand" from "weak demand", and including them let "DRAM demand is
#: weak" net out to neutral, which is worse than having no opinion at all.
_BULLISH_TERMS: dict[str, float] = {
    "beat": 1.0, "beats": 1.0, "raised": 1.0, "raises": 1.0, "upgrade": 1.2,
    "upgraded": 1.2, "record": 0.9, "surge": 1.2, "surged": 1.2, "rally": 1.0,
    "rallied": 1.0, "jumped": 1.1, "soared": 1.3, "strong": 0.8, "growth": 0.7,
    "expansion": 0.7, "sold out": 1.1, "shortage": 0.8,
    "outperform": 1.1, "accelerating": 0.9, "wins": 0.9,
    "partnership": 0.6, "breakthrough": 1.0, "exceeded": 1.0, "optimistic": 0.8,
}  # fmt: skip

_BEARISH_TERMS: dict[str, float] = {
    "miss": 1.0, "missed": 1.0, "misses": 1.0, "cut": 1.0, "cuts": 1.0,
    "downgrade": 1.2, "downgraded": 1.2, "plunge": 1.3, "plunged": 1.3,
    "slump": 1.1, "slumped": 1.1, "fell": 0.9, "falls": 0.9, "decline": 0.9,
    "declined": 0.9, "weak": 0.8, "weakness": 0.9, "glut": 1.0, "oversupply": 1.1,
    "delay": 0.8, "delayed": 0.8, "shortfall": 1.0, "warning": 1.0, "warns": 1.0,
    "underperform": 1.1, "loss": 0.9, "losses": 0.9, "layoffs": 1.0,
    "investigation": 0.9, "recall": 1.0, "halted": 1.0, "slowdown": 1.0,
}  # fmt: skip

#: Terms that invert the polarity of a nearby sentiment word. Without these,
#: "demand is not weak" scores bearish.
_NEGATIONS = frozenset({"not", "no", "never", "without", "fails", "failed", "unlikely"})

#: Words either side of a sentiment term that a negation can reach across.
_NEGATION_WINDOW = 3

#: Below this, the verdict is NEUTRAL regardless of sign. A single matched word
#: in a long article is not evidence of a direction.
_NEUTRAL_BAND = 0.15


@runtime_checkable
class SentimentAnalyzer(Protocol):
    """Scores financial text as bullish, bearish or neutral."""

    @property
    def model_name(self) -> str:
        """Identifier stored with each score, for provenance and reproducibility."""
        ...

    async def score(self, text: str) -> SentimentScore:
        """Return the sentiment of one passage."""
        ...

    async def score_many(self, texts: Sequence[str]) -> list[SentimentScore]:
        """Return sentiments for a batch, which transformer models do far faster."""
        ...


class LexiconSentimentAnalyzer:
    """Weighted keyword scoring with negation handling.

    Fast, dependency-free and genuinely limited: it cannot read "beats estimates
    but guides lower" as the bearish story it is, because it has no syntax. It is
    a floor, not a solution -- but a floor that always works.
    """

    def __init__(self) -> None:
        """Compile the term patterns once."""
        self._bullish = _compile_terms(_BULLISH_TERMS)
        self._bearish = _compile_terms(_BEARISH_TERMS)

    @property
    def model_name(self) -> str:
        """Identifier stored with each score."""
        return "lexicon-v1"

    async def score(self, text: str) -> SentimentScore:
        """Score one passage."""
        lowered = text.lower()
        tokens = re.findall(r"[a-z']+", lowered)

        bullish = self._weigh(lowered, tokens, self._bullish)
        bearish = self._weigh(lowered, tokens, self._bearish)
        total = bullish + bearish

        if total == 0:
            return SentimentScore(
                label=Sentiment.NEUTRAL,
                confidence=0.35,
                polarity=0.0,
                model_name=self.model_name,
            )

        polarity = (bullish - bearish) / total
        label = (
            Sentiment.NEUTRAL
            if abs(polarity) < _NEUTRAL_BAND
            else (Sentiment.BULLISH if polarity > 0 else Sentiment.BEARISH)
        )
        # Confidence grows with both the margin and the amount of evidence: two
        # matched terms out of two is weaker than twenty out of twenty.
        evidence = min(total / 6.0, 1.0)
        confidence = min(0.95, 0.4 + 0.55 * abs(polarity) * evidence)

        return SentimentScore(
            label=label,
            confidence=round(confidence, 4),
            polarity=round(polarity, 4),
            model_name=self.model_name,
        )

    async def score_many(self, texts: Sequence[str]) -> list[SentimentScore]:
        """Score a batch. No batching advantage here; the loop is the batch."""
        return [await self.score(text) for text in texts]

    def _weigh(self, lowered: str, tokens: Sequence[str], patterns: dict[str, float]) -> float:
        """Sum matched term weights, flipping any that a negation reaches."""
        total = 0.0
        for term, weight in patterns.items():
            if " " in term:
                # Multi-word terms are matched on the raw string; negation
                # handling for phrases is not attempted rather than done badly.
                total += lowered.count(term) * weight
                continue
            for index, token in enumerate(tokens):
                if token != term:
                    continue
                window = tokens[max(0, index - _NEGATION_WINDOW) : index]
                total += -weight if any(w in _NEGATIONS for w in window) else weight
        return max(total, 0.0)


class FinBertSentimentAnalyzer:
    """FinBERT (``ProsusAI/finbert``) inference over financial text.

    Loaded lazily and pinned to CPU. Inference is genuinely CPU-bound, so every
    call runs in a worker thread: doing it inline would block the event loop for
    hundreds of milliseconds per batch, stalling the scheduler and every other
    job sharing the loop.
    """

    #: FinBERT's label order. Hardcoding index-to-label would break silently if
    #: the checkpoint's config ever reordered them, so the model's own mapping is
    #: read at load time and this is only the fallback.
    _FALLBACK_LABELS = ("positive", "negative", "neutral")

    _LABEL_MAP: ClassVar[dict[str, Sentiment]] = {
        "positive": Sentiment.BULLISH,
        "negative": Sentiment.BEARISH,
        "neutral": Sentiment.NEUTRAL,
    }

    def __init__(
        self,
        *,
        model_id: str = "ProsusAI/finbert",
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        """Configure the analyser without loading anything.

        Args:
            model_id: Hugging Face checkpoint.
            max_length: Token cap. BERT's hard limit is 512; longer articles are
                truncated, which is acceptable because a financial story states
                its thesis in the opening paragraphs.
            batch_size: Passages per forward pass.
        """
        self._model_id = model_id
        self._max_length = max_length
        self._batch_size = batch_size
        self._model: object | None = None
        self._tokenizer: object | None = None
        self._labels: tuple[str, ...] = self._FALLBACK_LABELS

    @property
    def model_name(self) -> str:
        """Identifier stored with each score."""
        return self._model_id

    @staticmethod
    def is_available() -> bool:
        """Return whether the ML stack is installed.

        Checked before construction so a missing dependency degrades to the
        lexicon with a log line, rather than raising at startup.
        """
        from importlib.util import find_spec  # noqa: PLC0415

        return find_spec("torch") is not None and find_spec("transformers") is not None

    async def score(self, text: str) -> SentimentScore:
        """Score one passage."""
        return (await self.score_many([text]))[0]

    async def score_many(self, texts: Sequence[str]) -> list[SentimentScore]:
        """Score a batch in a worker thread."""
        import asyncio  # noqa: PLC0415

        if not texts:
            return []
        return await asyncio.to_thread(self._score_sync, list(texts))

    def _load(self) -> tuple[object, object]:
        """Load the tokenizer and model once, on first use."""
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model

        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        logger.info("finbert_loading", model=self._model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        model = AutoModelForSequenceClassification.from_pretrained(self._model_id)
        model.eval()
        torch.set_num_threads(max(1, (torch.get_num_threads() or 2) // 2))
        self._model = model

        # Read the label order from the checkpoint rather than assuming it.
        config_labels = getattr(model.config, "id2label", None)
        if isinstance(config_labels, dict) and config_labels:
            self._labels = tuple(str(config_labels[key]).lower() for key in sorted(config_labels))
        logger.info("finbert_loaded", model=self._model_id, labels=list(self._labels))
        return self._tokenizer, self._model

    def _score_sync(self, texts: Sequence[str]) -> list[SentimentScore]:
        """Run inference synchronously; always called inside a worker thread."""
        import torch  # noqa: PLC0415

        tokenizer, model = self._load()
        scores: list[SentimentScore] = []

        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            encoded = tokenizer(  # type: ignore[operator]
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = model(**encoded).logits  # type: ignore[operator]
            probabilities = torch.softmax(logits, dim=-1).tolist()
            scores.extend(self._to_score(row) for row in probabilities)

        return scores

    def _to_score(self, probabilities: Sequence[float]) -> SentimentScore:
        """Convert a probability row into the platform's score shape."""
        by_label = dict(zip(self._labels, probabilities, strict=False))
        best_label = max(by_label, key=lambda label: by_label[label])

        bullish = by_label.get("positive", 0.0)
        bearish = by_label.get("negative", 0.0)

        return SentimentScore(
            label=self._LABEL_MAP.get(best_label, Sentiment.NEUTRAL),
            confidence=round(float(by_label[best_label]), 4),
            # Signed polarity from the two directional classes, so articles can
            # be ranked by conviction rather than only bucketed by label.
            polarity=round(float(bullish - bearish), 4),
            model_name=self.model_name,
        )


def build_analyzer(*, prefer_finbert: bool) -> SentimentAnalyzer:
    """Return the best available analyser.

    Args:
        prefer_finbert: Whether to use FinBERT when the ML stack is present.

    Returns:
        FinBERT if requested and importable, the lexicon otherwise. A missing
        dependency is a logged downgrade, never a startup failure.
    """
    if not prefer_finbert:
        return LexiconSentimentAnalyzer()
    if not FinBertSentimentAnalyzer.is_available():
        logger.warning(
            "finbert_unavailable",
            reason="torch or transformers is not installed",
            fallback="lexicon-v1",
        )
        return LexiconSentimentAnalyzer()
    return FinBertSentimentAnalyzer()


def _compile_terms(terms: dict[str, float]) -> dict[str, float]:
    """Normalise term keys to lower case."""
    return {term.lower(): weight for term, weight in terms.items()}
