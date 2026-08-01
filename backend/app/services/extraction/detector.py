"""Finding known entities in free text.

Runs before any model call and decides whether one happens at all. Two reasons
that ordering matters.

**Cost.** Local inference is roughly ten seconds an article. Over 642 articles
that is nearly two hours, and most of them mention no two graph entities and
therefore cannot possibly yield an edge. Detection is a millisecond and filters
the corpus to the fraction worth reading.

**Grounding.** The detected entities become the *closed vocabulary* the model is
allowed to relate. A model told "here is an article, extract relationships"
invents companies; a model told "these four entities appear in this text, which
of them are related and how" has nowhere to put an invention -- and anything it
does invent fails validation mechanically rather than on review.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.models.graph import Entity

#: Shortest surface form worth matching. Two-character tickers appear inside
#: ordinary prose and inside other tickers, and no stop list makes them safe.
_MIN_SURFACE_LENGTH = 3


@dataclass(frozen=True, slots=True)
class EntityMention:
    """One entity found in a text, and the surface form that matched."""

    entity: Entity
    matched: str
    position: int


class MentionDetector:
    """Matches entity names, aliases and tickers against text."""

    def __init__(self, entities: Sequence[Entity]) -> None:
        """Compile one pattern per entity from its names and aliases."""
        self._patterns: list[tuple[Entity, re.Pattern[str]]] = []
        for entity in entities:
            surfaces = self._surfaces(entity)
            if not surfaces:
                continue
            # Longest first so "Taiwan Semiconductor Manufacturing" wins over a
            # bare "Taiwan" if both were ever listed.
            alternation = "|".join(
                re.escape(surface) for surface in sorted(surfaces, key=len, reverse=True)
            )
            # Word boundaries stop "AMD" matching inside "AMDAHL" and, more
            # importantly, stop the two-letter tickers from matching everywhere.
            self._patterns.append((entity, re.compile(rf"\b({alternation})\b", re.IGNORECASE)))

    @staticmethod
    def _surfaces(entity: Entity) -> set[str]:
        """Return every string that should count as naming this entity.

        The ticker is included only when it is distinctive. Symbols like "MU"
        and "TSM" appear inside ordinary prose and inside other tickers; a
        three-character minimum keeps the obvious false positives out without
        needing a stop list.
        """
        surfaces = {entity.name, *entity.aliases}
        if entity.symbol and "." not in entity.symbol and len(entity.symbol) >= _MIN_SURFACE_LENGTH:
            surfaces.add(entity.symbol)
        return {surface for surface in surfaces if surface and len(surface) >= _MIN_SURFACE_LENGTH}

    def detect(self, text: str) -> list[EntityMention]:
        """Return every distinct entity mentioned, in order of first appearance.

        One mention per entity: an article naming NVIDIA six times is still one
        participant, and returning six would let a duplicate dominate the
        vocabulary handed to the model.
        """
        found: dict[str, EntityMention] = {}
        for entity, pattern in self._patterns:
            match = pattern.search(text)
            if match is not None and entity.slug not in found:
                found[entity.slug] = EntityMention(
                    entity=entity, matched=match.group(1), position=match.start()
                )
        return sorted(found.values(), key=lambda mention: mention.position)
