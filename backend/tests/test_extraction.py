"""Tests for relationship extraction.

The extractor's job is not to be right -- a 3B model reading a headline will not
be. Its job is to let nothing through that has not been mechanically checked,
while still accepting evidence that genuinely states a relationship. Both halves
matter: a validator that rejects everything is as useless as one that accepts
everything, and only the second failure is obvious.

Every rejection case below is a real proposal this pipeline produced against the
live corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from app.models.enums import RelationKind
from app.services.extraction.detector import MentionDetector
from app.services.extraction.extractor import ExtractionReport, RelationshipExtractor
from app.services.rag.llm import LlmResponse


@dataclass
class FakeEntity:
    """Stands in for the ORM node."""

    slug: str
    name: str
    symbol: str | None = None
    aliases: list[str] = field(default_factory=list)


NVDA = FakeEntity("nvidia", "NVIDIA", "NVDA", ["Nvidia"])
TSMC = FakeEntity("tsmc", "TSMC", "TSM", ["Taiwan Semiconductor Manufacturing"])
MU = FakeEntity("micron", "Micron Technology", "MU", ["Micron"])
HYNIX = FakeEntity("sk-hynix", "SK Hynix", None, ["SK Hynix", "Hynix"])
SAMSUNG = FakeEntity("samsung", "Samsung Electronics", None, ["Samsung"])
CRWV = FakeEntity("coreweave", "CoreWeave", "CRWV", ["CoreWeave"])

ENTITIES = [NVDA, TSMC, MU, HYNIX, SAMSUNG, CRWV]


class ScriptedLlm:
    """Returns a fixed JSON reply."""

    def __init__(self, payload: object, *, generative: bool = True) -> None:
        self._payload = payload
        self._generative = generative
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "fake-model" if self._generative else "extractive-v1"

    @property
    def is_generative(self) -> bool:
        return self._generative

    async def complete(self, *, question: str, context: str) -> LlmResponse:
        self.calls += 1
        text = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
        return LlmResponse(text=text, model_name=self.model_name)


def _document(title: str, summary: str = "") -> dict[str, Any]:
    return {"_id": "doc-1", "title": title, "summary": summary}


async def _run(payload: object, document: dict[str, Any]) -> tuple[list[Any], ExtractionReport]:
    extractor = RelationshipExtractor(llm=ScriptedLlm(payload), entities=ENTITIES)  # type: ignore[arg-type]
    return await extractor.extract([document])


class TestMentionDetection:
    """Which articles reach the model at all."""

    def test_an_alias_counts_as_a_mention(self) -> None:
        """An article writing the full legal name still names TSMC.

        Without aliases the detector matches nothing and the article is dropped
        from extraction silently, which is worse than dropping it loudly.
        """
        mentions = MentionDetector(ENTITIES).detect(  # type: ignore[arg-type]
            "Taiwan Semiconductor Manufacturing raised its outlook, helping Nvidia suppliers."
        )

        assert {mention.entity.slug for mention in mentions} == {"tsmc", "nvidia"}

    def test_an_entity_named_repeatedly_is_one_mention(self) -> None:
        """Six mentions of NVIDIA is still one participant."""
        mentions = MentionDetector(ENTITIES).detect("NVIDIA. Nvidia. NVDA. NVIDIA again.")  # type: ignore[arg-type]

        assert len(mentions) == 1

    def test_a_short_ticker_does_not_match_prose(self) -> None:
        """Two-character symbols appear inside ordinary words and other tickers.

        "MU" would otherwise match constantly, filling the vocabulary with a
        company the article never mentions.
        """
        mentions = MentionDetector(ENTITIES).detect("The company must adapt to demand.")  # type: ignore[arg-type]

        assert mentions == []

    async def test_an_article_naming_one_entity_never_reaches_the_model(self) -> None:
        """Most of the corpus never reaches the model.

        Local inference is ~10s an article and most of the
        corpus cannot possibly yield an edge.
        """
        llm = ScriptedLlm([])
        extractor = RelationshipExtractor(llm=llm, entities=ENTITIES)  # type: ignore[arg-type]

        _, report = await extractor.extract([_document("NVIDIA rose 3% on Tuesday.")])

        assert llm.calls == 0
        assert report.documents_read == 0
        assert report.rejected["fewer_than_two_entities"] == 1


class TestAcceptance:
    """Evidence that genuinely states a relationship must get through."""

    async def test_a_well_evidenced_relationship_is_proposed(self) -> None:
        quote = "Micron Technology will supply HBM3E to NVIDIA for its data centre accelerators."
        proposals, report = await _run(
            [
                {
                    "source": "Micron Technology",
                    "target": "NVIDIA",
                    "relation": "supplies",
                    "quote": quote,
                }
            ],
            _document("Micron wins NVIDIA HBM order", quote),
        )

        assert report.proposed == 1
        assert proposals[0].source_slug == "micron"
        assert proposals[0].target_slug == "nvidia"
        assert proposals[0].kind is RelationKind.SUPPLIES
        assert proposals[0].quote == quote

    async def test_an_inferred_proposal_never_outranks_a_curated_edge(self) -> None:
        """An inferred edge must never outrank a disclosed one.

        Curated edges sit near 0.95. This must stay far below, and below the
        0.5 floor impact propagation applies by default.
        """
        quote = "Micron Technology will supply HBM3E to NVIDIA for its data centre accelerators."
        proposals, _ = await _run(
            [
                {
                    "source": "Micron Technology",
                    "target": "NVIDIA",
                    "relation": "supplies",
                    "quote": quote,
                }
            ],
            _document("Micron wins NVIDIA HBM order", quote),
        )

        assert proposals[0].confidence < 0.7


class TestRejection:
    """Every case here is a real proposal this pipeline produced."""

    async def test_a_fabricated_quote_is_rejected(self) -> None:
        """The model's most common failure, and the cheapest to catch."""
        _, report = await _run(
            [
                {
                    "source": "NVIDIA",
                    "target": "TSMC",
                    "relation": "customer_of",
                    "quote": "NVIDIA confirmed it buys every wafer from TSMC exclusively.",
                }
            ],
            _document("TSMC lifts outlook", "TSMC raised guidance. NVIDIA shares rose."),
        )

        assert report.rejected["quote_not_in_source"] == 1

    async def test_a_quote_naming_only_one_party_is_rejected(self) -> None:
        """A quote naming one party cannot relate two.

        Real case: "First-half revenue passed 100 trillion won for the first
        time" was offered as evidence that NVIDIA is a customer of SK Hynix.
        The sentence is genuinely in the article and establishes nothing.
        """
        quote = "First-half revenue passed 100 trillion won for the first time"
        _, report = await _run(
            [{"source": "NVIDIA", "target": "SK Hynix", "relation": "customer_of", "quote": quote}],
            _document("SK Hynix results", f"{quote}. NVIDIA demand stayed strong."),
        )

        assert report.rejected["quote_missing_an_entity"] == 1

    async def test_a_quote_that_merely_names_both_is_rejected(self) -> None:
        """Naming both parties is not asserting a relationship.

        Real case: "AI Boom Spurs Insider Selling by Nvidia, CoreWeave
        Billionaires" was offered as evidence the two compete. Both are named;
        nothing relational is asserted.
        """
        quote = "AI Boom Spurs Insider Selling by Nvidia, CoreWeave Billionaires"
        _, report = await _run(
            [
                {
                    "source": "NVIDIA",
                    "target": "CoreWeave",
                    "relation": "competes_with",
                    "quote": quote,
                }
            ],
            _document(quote),
        )

        assert report.rejected["quote_lacks_relational_cue"] == 1

    async def test_contradictory_directions_are_both_rejected(self) -> None:
        """Contradictory proposals are both rejected.

        Real case: the model proposed Samsung manufactures for SK Hynix *and*
        the exact reverse, from one sentence supporting neither.
        """
        quote = "AI chip demand ties Samsung and SK Hynix to Wall Street."
        _, report = await _run(
            [
                {
                    "source": "Samsung Electronics",
                    "target": "SK Hynix",
                    "relation": "manufactures",
                    "quote": quote,
                },
                {
                    "source": "SK Hynix",
                    "target": "Samsung Electronics",
                    "relation": "manufactures",
                    "quote": quote,
                },
            ],
            _document(quote),
        )

        assert report.proposed == 0
        assert report.rejected["quote_lacks_relational_cue"] == 2

    async def test_an_invented_company_is_rejected(self) -> None:
        """The model may not introduce a company.

        It is handed a numbered list and is
        not permitted to introduce anything else.
        """
        _, report = await _run(
            [
                {
                    "source": "NVIDIA",
                    "target": "Acme Semiconductors",
                    "relation": "supplies",
                    "quote": "NVIDIA supplies Acme Semiconductors with accelerators.",
                }
            ],
            _document("NVIDIA and TSMC", "NVIDIA and TSMC both rose."),
        )

        assert report.rejected["unknown_entity"] == 1

    async def test_a_relation_outside_the_vocabulary_is_rejected(self) -> None:
        _, report = await _run(
            [{"source": "NVIDIA", "target": "TSMC", "relation": "hates", "quote": "x" * 40}],
            _document("NVIDIA and TSMC", "NVIDIA and TSMC both rose."),
        )

        assert report.rejected["unknown_relation"] == 1

    async def test_an_analytical_relation_is_not_extractable(self) -> None:
        """Analytical relations are outside the extractable vocabulary.

        "Depends on" is a judgement rather than something an article asserts.
        Allowing it produced editorialising dressed as extraction.
        """
        quote = "NVIDIA depends on TSMC for every leading-edge wafer it ships."
        _, report = await _run(
            [{"source": "NVIDIA", "target": "TSMC", "relation": "depends_on", "quote": quote}],
            _document("NVIDIA and TSMC", quote),
        )

        assert report.rejected["relation_not_extractable"] == 1

    async def test_a_self_relationship_is_rejected(self) -> None:
        quote = "NVIDIA supplies NVIDIA with accelerators for its own use."
        _, report = await _run(
            [{"source": "NVIDIA", "target": "Nvidia", "relation": "supplies", "quote": quote}],
            _document("NVIDIA", f"{quote} TSMC also rose."),
        )

        assert report.rejected["self_relationship"] == 1

    @pytest.mark.parametrize(
        "reply",
        ["not json at all", "```json\nnope\n```", "", "{}"],
        ids=["prose", "bad-fence", "empty", "object-not-array"],
    )
    async def test_unparseable_output_yields_nothing(self, reply: str) -> None:
        """A reply that is not JSON contains no verified relationships."""
        proposals, report = await _run(reply, _document("NVIDIA and TSMC", "Both rose today."))

        assert proposals == []
        assert report.proposed == 0


class TestNoModel:
    """Behaviour when no generative model is reachable."""

    async def test_extraction_refuses_rather_than_silently_yielding_nothing(self) -> None:
        """A run with no model refuses rather than yielding a misleading zero.

        The extractive fallback selects sentences, never JSON. Every
        document would be read and yield nothing, and the run would look like
        "this corpus has no relationships" when it means "no model was here".
        """
        llm = ScriptedLlm([], generative=False)
        extractor = RelationshipExtractor(llm=llm, entities=ENTITIES)  # type: ignore[arg-type]

        _, report = await extractor.extract([_document("NVIDIA and TSMC", "Both rose.")])

        assert llm.calls == 0
        assert report.rejected["no_generative_model"] == 1
