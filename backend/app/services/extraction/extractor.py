"""Proposing graph edges from news text.

The design assumption is that **the model will get things wrong**, and that this
is fine provided nothing it says is believed without checking. A 3B model
running locally is a competent reader of a single sentence and a poor reasoner
about an industry; asked to do only the former, and checked mechanically on the
latter, it is useful.

Four defences, in order of how much they catch:

1. **A closed vocabulary.** The model is not asked "what relationships are in
   this article". It is handed the entities a deterministic detector already
   found in the text and asked which of *those* are related. It has nowhere to
   put an invented company.

2. **A required verbatim quote.** Every proposal must include the span of the
   article that supports it, and that span is checked against the source with a
   substring match. A model that fabricates a relationship almost always
   fabricates the quote too, and the check catches it without anyone reading a
   word. This is the single most effective filter here.

3. **A closed relation vocabulary**, validated against the enum.

4. **Nothing merges.** Proposals land in their own table with
   ``EvidenceSource.INFERRED`` and low confidence, and become graph edges only
   when a person accepts them. The curated backbone stays clean, which is what
   makes it useful as the thing inferences are checked against.

What this deliberately does *not* do is let the model assign confidence. Asked
how sure it is, a small model answers 0.9 to everything. Confidence here is
computed from properties that can be measured: whether the quote verified,
whether both entities were detected rather than merely named, and how directly
the relation vocabulary matched.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.models.enums import RelationKind
from app.models.graph import Entity
from app.services.extraction.detector import MentionDetector
from app.services.rag.llm import LlmClient

logger = get_logger(__name__)

#: Relations the extractor may propose. Narrower than the full enum on purpose:
#: these are the ones a news sentence actually states. "Depends on" is an
#: analytical judgement rather than something an article asserts, and letting
#: the model propose it produced editorialising dressed as extraction.
EXTRACTABLE: frozenset[RelationKind] = frozenset(
    {
        RelationKind.SUPPLIES,
        RelationKind.MANUFACTURES,
        RelationKind.CUSTOMER_OF,
        RelationKind.PARTNERS_WITH,
        RelationKind.COMPETES_WITH,
        RelationKind.INVESTS_IN,
        RelationKind.ACQUIRED,
    }
)

#: An article naming two entities does not relate them. Two is the minimum for a
#: relationship to be possible at all, and below it no model call is made.
_MIN_ENTITIES = 2

#: Ceiling on how many proposals one short article may produce. A model given a
#: 200-character summary that emits nine relationships is looping, not reading.
_MAX_PER_DOCUMENT = 4

_INSTRUCTION = """\
You extract business relationships from financial news. You are given a text \
and a numbered list of companies that appear in it.

Return ONLY a JSON array. Each element must be an object with exactly these \
keys:
  "source": the company name, copied exactly from the list
  "target": the company name, copied exactly from the list
  "relation": one of {relations}
  "quote": the exact sentence from the text that states this relationship

Rules:
1. Use ONLY companies from the numbered list. Never introduce another name.
2. The "quote" must be copied character-for-character from the text. Do not \
paraphrase, do not summarise, do not fix punctuation.
3. Only extract a relationship the text actually states. Two companies \
appearing in the same article are not related by that fact.
4. Direction matters: "source supplies target" means source is the supplier.
5. If the text states no relationship between the listed companies, return [].

Return the JSON array and nothing else."""


@dataclass(frozen=True, slots=True)
class ProposedRelation:
    """One candidate edge, after validation."""

    source_slug: str
    target_slug: str
    kind: RelationKind
    quote: str
    confidence: float
    document_id: str
    document_title: str
    document_url: str | None = None


@dataclass
class ExtractionReport:
    """What one extraction run did, and why it rejected what it rejected."""

    documents_seen: int = 0
    documents_read: int = 0
    proposed: int = 0
    #: Counted by cause. The distribution is the signal that tells you whether
    #: the prompt is working or the model is drifting, and it is the first
    #: thing to look at when yield drops.
    rejected: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        """Record one rejection."""
        self.rejected[reason] = self.rejected.get(reason, 0) + 1


class RelationshipExtractor:
    """Reads documents and proposes graph edges, verifying every claim."""

    def __init__(self, *, llm: LlmClient, entities: Sequence[Entity]) -> None:
        """Bind the extractor to a model and the known entity vocabulary."""
        self._llm = llm
        self._entities = entities
        self._detector = MentionDetector(entities)
        self._by_name: dict[str, Entity] = {}
        for entity in entities:
            for surface in (entity.name, entity.slug, *entity.aliases):
                if surface:
                    self._by_name[surface.lower()] = entity

    async def extract(
        self,
        documents: Sequence[dict[str, object]],
        report: ExtractionReport | None = None,
    ) -> tuple[list[ProposedRelation], ExtractionReport]:
        """Propose edges from a batch of documents.

        Args:
            documents: Each needs ``_id``, ``title`` and ``summary``.
            report: Accumulates across batches when supplied.

        Returns:
            Validated proposals and the run's statistics.
        """
        run = report or ExtractionReport()
        proposals: list[ProposedRelation] = []

        # Extraction needs generation. The extractive fallback answers by
        # selecting sentences, which is never JSON, so every document would be
        # read and silently yield nothing -- a run that looks like "no
        # relationships in this corpus" when it means "no model was available".
        if not self._llm.is_generative:
            run.reject("no_generative_model")
            logger.warning(
                "extraction_skipped",
                reason="no generative model available",
                model=self._llm.model_name,
            )
            return proposals, run

        for document in documents:
            run.documents_seen += 1
            text = _text_of(document)
            mentions = self._detector.detect(text)

            if len(mentions) < _MIN_ENTITIES:
                run.reject("fewer_than_two_entities")
                continue

            run.documents_read += 1
            try:
                raw = await self._ask(text, [mention.entity for mention in mentions])
            except Exception as exc:  # noqa: BLE001 - one bad document is not a failed run
                logger.warning(
                    "extraction_failed", document=str(document.get("_id")), error=str(exc)
                )
                run.reject("model_error")
                continue

            for candidate in raw[:_MAX_PER_DOCUMENT]:
                proposal = self._validate(candidate, text, document, run)
                if proposal is not None:
                    proposals.append(proposal)
                    run.proposed += 1

        logger.info(
            "extraction_complete",
            seen=run.documents_seen,
            read=run.documents_read,
            proposed=run.proposed,
            rejected=run.rejected,
        )
        return proposals, run

    async def _ask(self, text: str, entities: Sequence[Entity]) -> list[dict[str, str]]:
        """Ask the model for relationships and parse the reply."""
        listing = "\n".join(f"{index}. {e.name}" for index, e in enumerate(entities, start=1))
        instruction = _INSTRUCTION.format(
            relations=", ".join(sorted(kind.value for kind in EXTRACTABLE))
        )
        response = await self._llm.complete(
            question=instruction,
            context=f"Companies present:\n{listing}\n\nText:\n{text}",
        )
        return _parse_json_array(response.text)

    def _validate(  # noqa: PLR0911 - one early return per rejection cause is
        # clearer than a nested chain; each is a distinct, named failure.
        self,
        candidate: dict[str, str],
        text: str,
        document: dict[str, object],
        run: ExtractionReport,
    ) -> ProposedRelation | None:
        """Check one candidate against the source, rejecting anything unproven."""
        source = self._by_name.get(str(candidate.get("source", "")).strip().lower())
        target = self._by_name.get(str(candidate.get("target", "")).strip().lower())
        if source is None or target is None:
            # The model named something outside the vocabulary it was given.
            run.reject("unknown_entity")
            return None
        if source.slug == target.slug:
            run.reject("self_relationship")
            return None

        try:
            kind = RelationKind(str(candidate.get("relation", "")).strip().lower())
        except ValueError:
            run.reject("unknown_relation")
            return None
        if kind not in EXTRACTABLE:
            run.reject("relation_not_extractable")
            return None

        quote = str(candidate.get("quote", "")).strip()
        if not _quote_appears_in(quote, text):
            # A fabricated relationship nearly always comes with a fabricated
            # quote, and this catches it without a human reading anything.
            run.reject("quote_not_in_source")
            return None

        # Existing in the source is not the same as supporting the claim. The
        # model reliably picks a real sentence that merely *names* both
        # companies: "AI Boom Spurs Insider Selling by Nvidia, CoreWeave
        # Billionaires" was offered as evidence that they compete. Both checks
        # below are what make the quote evidence rather than decoration.
        if not _quote_names_both(quote, source, target):
            run.reject("quote_missing_an_entity")
            return None
        if not _quote_supports(quote, kind):
            run.reject("quote_lacks_relational_cue")
            return None

        return ProposedRelation(
            source_slug=source.slug,
            target_slug=target.slug,
            kind=kind,
            quote=quote,
            confidence=_confidence_for(kind),
            document_id=str(document.get("_id", "")),
            document_title=str(document.get("title", "")),
            document_url=str(document["url"]) if document.get("url") else None,
        )


def _text_of(document: dict[str, object]) -> str:
    """Join a document's readable fields."""
    title = str(document.get("title") or "")
    summary = str(document.get("summary") or "")
    content = str(document.get("content") or "")
    return "\n".join(part for part in (title, summary, content) if part).strip()


#: Confidence by how directly a news sentence can establish the relation.
#:
#: Not asked of the model: a small model answers 0.9 to every "how sure are
#: you". These are properties of the *claim type*. An announced partnership or
#: acquisition is a stated fact a headline reports accurately; "competes with"
#: is frequently the journalist's framing rather than either company's claim.
_KIND_CONFIDENCE: dict[RelationKind, float] = {
    RelationKind.ACQUIRED: 0.7,
    RelationKind.INVESTS_IN: 0.65,
    RelationKind.PARTNERS_WITH: 0.6,
    RelationKind.SUPPLIES: 0.55,
    RelationKind.MANUFACTURES: 0.55,
    RelationKind.CUSTOMER_OF: 0.5,
    RelationKind.COMPETES_WITH: 0.4,
}


def _confidence_for(kind: RelationKind) -> float:
    """Return the confidence an inferred edge of this kind starts with.

    Every value sits well below the curated floor of ~0.95, so an inferred edge
    can never outrank a disclosed one and is filtered out by default from impact
    propagation, which requires 0.5.
    """
    return _KIND_CONFIDENCE.get(kind, 0.35)


def _normalise(value: str) -> str:
    """Collapse whitespace and smart punctuation for comparison.

    Models reliably reproduce a quote's words while normalising its typography
    -- a curly apostrophe becomes straight, a non-breaking space becomes a
    space. Comparing raw would reject correct quotes on punctuation alone.
    """
    replaced = (
        value.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u00a0", " ")
    )
    return re.sub(r"\s+", " ", replaced).strip().lower()


#: Shortest quote worth trusting. A three-word fragment appears in almost any
#: text by chance, so a short "quote" verifies while proving nothing.
_MIN_QUOTE_CHARS = 25


def _quote_appears_in(quote: str, text: str) -> bool:
    """Whether the quoted span is really present in the source."""
    if len(quote) < _MIN_QUOTE_CHARS:
        return False
    return _normalise(quote) in _normalise(text)


#: Words that must appear in a quote for it to establish each relation.
#:
#: Crude, and deliberately so. The alternative is asking a second model whether
#: the first model's evidence supports its claim, which is a weaker check that
#: costs another ten seconds. A sentence asserting that one company supplies
#: another essentially always contains one of these stems; one that merely
#: mentions both companies essentially never does.
_RELATION_CUES: dict[RelationKind, tuple[str, ...]] = {
    RelationKind.SUPPLIES: ("suppl", "provide", "ship", "deliver", "sell"),
    RelationKind.MANUFACTURES: ("manufactur", "fabricat", "produce", "make", "foundry", "build"),
    RelationKind.CUSTOMER_OF: ("customer", "buy", "purchas", "order", "client"),
    RelationKind.PARTNERS_WITH: ("partner", "collaborat", "joint", "alliance", "team up", "deal"),
    RelationKind.COMPETES_WITH: ("compet", "rival", "versus", " vs ", "challeng", "take on"),
    RelationKind.INVESTS_IN: ("invest", "stake", "funding", "back", "round"),
    RelationKind.ACQUIRED: ("acqui", "buyout", "takeover", "purchase of", "merg"),
}


def _quote_supports(quote: str, kind: RelationKind) -> bool:
    """Whether the quote contains language consistent with the relation."""
    lowered = _normalise(quote)
    return any(cue in lowered for cue in _RELATION_CUES.get(kind, ()))


def _surfaces_of(entity: Entity) -> list[str]:
    """Every name an entity might appear under, for quote checking."""
    names = [entity.name, *entity.aliases]
    if entity.symbol and "." not in entity.symbol:
        names.append(entity.symbol)
    return [_normalise(name) for name in names if name]


def _quote_names_both(quote: str, source: Entity, target: Entity) -> bool:
    """Whether both endpoints are actually named inside the quoted span.

    A relationship stated in a sentence names both parties in that sentence.
    Without this, the model cites a line about one company as evidence for a
    claim about two -- which is how "First-half revenue passed 100 trillion won"
    came to be offered as proof that NVIDIA is a customer of SK Hynix.
    """
    lowered = _normalise(quote)
    return any(name in lowered for name in _surfaces_of(source)) and any(
        name in lowered for name in _surfaces_of(target)
    )


def _parse_json_array(reply: str) -> list[dict[str, str]]:
    """Extract the JSON array from a model reply.

    Small models wrap JSON in prose and fences however they please, so the array
    is located by its brackets rather than by trusting the reply to be clean.
    Returning an empty list on unparseable output is correct: a reply that is
    not JSON contains no verified relationships.
    """
    start = reply.find("[")
    end = reply.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        parsed = json.loads(reply[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []
