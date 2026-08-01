"""Graph traversal and impact propagation.

The propagation is the part that answers "who benefits". It is deliberately a
transparent, inspectable rule rather than a model: every score comes with the
path that produced it, so a reader can see *why* Vertiv appears on a list about
Microsoft's capital spending and disagree with the reasoning rather than with an
oracle.

A language model is the wrong tool for this specific job. Asked "who benefits if
Microsoft doubles AI spending", a model produces a plausible list from memory --
including companies that are not in this graph and relationships that do not
exist. The graph produces a list that is *derived*, with each step attributable
to a disclosure. The model's job comes afterwards: explaining the paths in
prose, which is what it is good at.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from app.core.logging import get_logger
from app.models.enums import SYMMETRIC_RELATIONS, EntityKind, RelationKind
from app.models.graph import Entity
from app.repositories.graph import Edge, EntityRepository, RelationshipRepository

logger = get_logger(__name__)

#: How a shock travels along each relation type, as a signed multiplier.
#:
#: Positive means the effect keeps its sign: if demand for NVIDIA rises, demand
#: for its suppliers rises. Negative means the sign flips: a competitor gaining
#: is, all else equal, a loss. The magnitudes say how much of the effect
#: survives the hop, and they are judgements -- stated here so they can be
#: argued with rather than buried in a scoring function.
_TRANSMISSION: dict[RelationKind, float] = {
    # A supplier's fortunes track its customer's volume closely. This is the
    # strongest link in the chain and the reason the graph exists.
    RelationKind.SUPPLIES: 0.85,
    RelationKind.MANUFACTURES: 0.85,
    RelationKind.CUSTOMER_OF: 0.7,
    RelationKind.DEPENDS_ON: 0.8,
    RelationKind.PRODUCES: 0.7,
    RelationKind.USES: 0.5,
    RelationKind.DEPLOYS: 0.6,
    # A partner shares in the upside but runs its own business.
    RelationKind.PARTNERS_WITH: 0.45,
    RelationKind.INVESTS_IN: 0.5,
    # Competition inverts. Not fully -- a rising tide in AI capex lifts rivals
    # too, so this is a partial offset rather than a mirror image.
    RelationKind.COMPETES_WITH: -0.35,
    RelationKind.ACQUIRED: 0.6,
    RelationKind.OPERATES: 0.4,
    RelationKind.LOCATED_IN: 0.1,
}


class Flow(StrEnum):
    """Which way along the supply chain a step travels.

    The graph's arrows point in whatever direction the relation was named --
    ``SUPPLIES`` points supplier→customer while ``CUSTOMER_OF`` points
    customer→supplier -- so raw arrow direction says nothing about economic
    direction. This normalises it: UPSTREAM always means "toward suppliers".
    """

    UPSTREAM = "upstream"
    DOWNSTREAM = "downstream"
    #: Competition, partnership and investment sit across the chain rather than
    #: along it. They neither establish nor violate a path's direction.
    LATERAL = "lateral"
    #: The origin, before any step has fixed a direction.
    UNSET = "unset"


#: Which way each relation points when followed *along* its arrow.
_FORWARD_FLOW: dict[RelationKind, Flow] = {
    # Arrow points supplier -> customer, i.e. down the chain toward demand.
    RelationKind.SUPPLIES: Flow.DOWNSTREAM,
    RelationKind.MANUFACTURES: Flow.DOWNSTREAM,
    RelationKind.PRODUCES: Flow.DOWNSTREAM,
    # Arrow points dependent -> dependency, i.e. up the chain toward supply.
    RelationKind.CUSTOMER_OF: Flow.UPSTREAM,
    RelationKind.DEPENDS_ON: Flow.UPSTREAM,
    RelationKind.USES: Flow.UPSTREAM,
    RelationKind.DEPLOYS: Flow.UPSTREAM,
    RelationKind.OPERATES: Flow.DOWNSTREAM,
    RelationKind.LOCATED_IN: Flow.LATERAL,
    RelationKind.COMPETES_WITH: Flow.LATERAL,
    RelationKind.PARTNERS_WITH: Flow.LATERAL,
    RelationKind.INVESTS_IN: Flow.LATERAL,
    RelationKind.ACQUIRED: Flow.LATERAL,
}


def _flow_of(kind: RelationKind, *, forwards: bool) -> Flow:
    """Return the economic direction of one traversal step."""
    flow = _FORWARD_FLOW.get(kind, Flow.LATERAL)
    if forwards or flow is Flow.LATERAL:
        return flow
    return Flow.UPSTREAM if flow is Flow.DOWNSTREAM else Flow.DOWNSTREAM


#: Below this, a path is noise. Without a floor a five-hop chain multiplies out
#: to a score indistinguishable from zero and clutters the answer with
#: companies whose connection is real but immaterial.
_SIGNIFICANCE_FLOOR = 0.05


@dataclass(frozen=True, slots=True)
class ImpactPath:
    """One route by which a shock reaches an entity."""

    #: Human-readable chain: "Microsoft → buys from → NVIDIA → supplied by → Micron".
    steps: tuple[str, ...]
    score: float
    #: Lowest confidence along the path. A chain is only as trustworthy as its
    #: weakest link, so this is a minimum rather than a product -- multiplying
    #: confidences would make every long path look untrustworthy even when each
    #: step is individually well-sourced.
    confidence: float


@dataclass(frozen=True, slots=True)
class ImpactResult:
    """An entity the shock reaches, and how."""

    entity: Entity
    score: float
    confidence: float
    #: Ordered best-first. Several routes to the same company is itself a
    #: finding: it means the exposure is structural rather than incidental.
    paths: tuple[ImpactPath, ...]

    @property
    def direction(self) -> str:
        """Whether this entity benefits or suffers."""
        if self.score > 0:
            return "benefits"
        return "at_risk" if self.score < 0 else "neutral"


class GraphService:
    """Traversal, neighbourhood assembly and impact propagation."""

    def __init__(
        self,
        *,
        entities: EntityRepository,
        relationships: RelationshipRepository,
    ) -> None:
        """Wire the service to its repositories."""
        self._entities = entities
        self._relationships = relationships

    async def get_entity(self, slug_or_symbol: str) -> Entity | None:
        """Resolve an entity by slug, falling back to ticker.

        Both, because the ecosystem map is navigated by slug while the rest of
        the platform is navigated by symbol, and a reader arriving from a
        company page has only the latter.
        """
        entity = await self._entities.get_by_slug(slug_or_symbol.lower())
        if entity is not None:
            return entity
        return await self._entities.get_by_symbol(slug_or_symbol)

    async def list_entities(
        self,
        *,
        kinds: Sequence[EntityKind] | None = None,
        tags: Sequence[str] | None = None,
    ) -> Sequence[Entity]:
        """Return entities, optionally narrowed by kind or stack layer."""
        return await self._entities.list_all(kinds=kinds, tags=tags)

    async def search(self, query: str, *, limit: int = 20) -> Sequence[Entity]:
        """Return entities matching a name, ticker or layer fragment."""
        return await self._entities.search(query, limit=limit)

    async def neighbourhood(
        self,
        entity: Entity,
        *,
        depth: int = 1,
        min_confidence: float = 0.0,
    ) -> list[Edge]:
        """Return the edges around an entity, out to ``depth`` hops."""
        if depth <= 1:
            return await self._relationships.neighbours(entity.id, min_confidence=min_confidence)
        return await self._relationships.subgraph(
            entity.id, depth=depth, min_confidence=min_confidence
        )

    async def supply_chain(self, entity: Entity, *, as_of: date | None = None) -> list[Edge]:
        """Return only the edges that form a supply chain.

        Competition and investment are real relationships and belong on the
        ecosystem map, but a supply chain diagram containing "competes with"
        arrows is not a supply chain -- it is a network diagram wearing one's
        name.
        """
        return await self._relationships.neighbours(
            entity.id,
            kinds=[
                RelationKind.SUPPLIES,
                RelationKind.MANUFACTURES,
                RelationKind.CUSTOMER_OF,
                RelationKind.DEPENDS_ON,
                RelationKind.PRODUCES,
            ],
            as_of=as_of,
        )

    async def propagate(
        self,
        origin: Entity,
        *,
        magnitude: float = 1.0,
        max_depth: int = 3,
        min_confidence: float = 0.5,
    ) -> list[ImpactResult]:
        """Trace a shock outward from one entity and score who it reaches.

        Breadth-first so the shortest, strongest route to each company is found
        first. A company reachable by several routes accumulates their scores,
        because two independent exposures really are more exposure than one.

        Args:
            origin: Where the shock starts.
            magnitude: Signed size. Negative models bad news -- a production
                cut rather than an expansion -- and every sign downstream flips
                with it.
            max_depth: Hops to follow. Three reaches from a hyperscaler through
                the accelerator designer and the foundry to the equipment
                makers, which is about where materiality runs out.
            min_confidence: Edges below this are not traversed at all, so a
                speculative relationship cannot introduce a whole branch.

        Returns:
            Affected entities ordered by absolute score, strongest first.
        """
        edges = await self._relationships.subgraph(
            origin.id, depth=max_depth, min_confidence=min_confidence
        )
        adjacency: dict[int, list[Edge]] = {}
        for edge in edges:
            adjacency.setdefault(edge.relationship.source_id, []).append(edge)
            adjacency.setdefault(edge.relationship.target_id, []).append(edge)

        # (entity id, score, weakest confidence, path text, flow established so far)
        queue: deque[tuple[int, float, float, tuple[str, ...], Flow]] = deque(
            [(origin.id, magnitude, 1.0, (origin.name,), Flow.UNSET)]
        )
        seen_depth: dict[int, int] = {origin.id: 0}
        reached: dict[int, list[ImpactPath]] = {}
        nodes: dict[int, Entity] = {origin.id: origin}

        while queue:
            current_id, score, confidence, path, flow = queue.popleft()
            depth = len(path) - 1
            if depth >= max_depth:
                continue

            for edge in adjacency.get(current_id, []):
                relation = edge.relationship
                transmission = _TRANSMISSION.get(relation.kind, 0.0)
                if transmission == 0.0:
                    continue

                forwards = relation.source_id == current_id
                neighbour = edge.target if forwards else edge.source
                if not forwards and not (
                    relation.kind in SYMMETRIC_RELATIONS or _reverse_is_meaningful(relation.kind)
                ):
                    continue

                # A shock travels one way along the chain. Once a path has gone
                # upstream (a customer's demand reaching its suppliers) it may
                # not turn around and come back down through that supplier's
                # *other* customers -- that is a different mechanism entirely
                # (capacity reallocation), and usually the opposite sign.
                #
                # Without this rule, "TSMC cuts production" reached Samsung as a
                # loss via TSMC → ASML → Samsung, reasoning that ASML is hurt so
                # its customers must be too. Samsung is TSMC's competitor: a TSMC
                # cut is, if anything, good for it.
                step_flow = _flow_of(relation.kind, forwards=forwards)
                if (
                    step_flow is not Flow.LATERAL
                    and flow is not Flow.UNSET
                    and step_flow is not flow
                ):
                    continue

                neighbour_id = neighbour.id
                if neighbour_id == origin.id:
                    continue

                next_score = score * transmission * relation.weight
                next_confidence = min(confidence, relation.confidence)
                if abs(next_score) < _SIGNIFICANCE_FLOOR:
                    continue

                next_path = (*path, f"{_verb(relation.kind, forwards=forwards)} {neighbour.name}")
                reached.setdefault(neighbour_id, []).append(
                    ImpactPath(steps=next_path, score=next_score, confidence=next_confidence)
                )
                nodes[neighbour_id] = neighbour

                # Revisit a node only if this route arrives no deeper than the
                # last. Without the guard a dense graph re-expands the same
                # nodes exponentially; with a strict `not in seen` the second,
                # possibly stronger route to a company would never be recorded.
                # A hop that flips the sign ends the path. Competition says what
                # happens to the *competitor*; continuing past it into that
                # competitor's own suppliers is a weaker, different effect that
                # overlaps the first-order one and nets to nonsense.
                #
                # Concretely: "Microsoft → competes with → Amazon → buys from →
                # NVIDIA" scored NVIDIA negatively on news that Microsoft was
                # spending *more*, reasoning that Amazon loses so Amazon's
                # suppliers lose. NVIDIA sells to both. It knocked the single
                # most obvious beneficiary off the list.
                if transmission < 0:
                    continue

                previous = seen_depth.get(neighbour_id)
                if previous is None or depth + 1 <= previous:
                    seen_depth[neighbour_id] = depth + 1
                    next_flow = flow if step_flow is Flow.LATERAL else step_flow
                    queue.append((neighbour_id, next_score, next_confidence, next_path, next_flow))

        results = [
            ImpactResult(
                entity=nodes[entity_id],
                score=sum(path.score for path in paths),
                confidence=max(path.confidence for path in paths),
                paths=tuple(sorted(paths, key=lambda p: abs(p.score), reverse=True)[:3]),
            )
            for entity_id, paths in reached.items()
        ]
        results.sort(key=lambda result: abs(result.score), reverse=True)

        logger.info(
            "impact_propagated",
            origin=origin.slug,
            magnitude=magnitude,
            reached=len(results),
            edges=len(edges),
        )
        return results

    async def stats(self) -> dict[str, object]:
        """Return the graph's size and composition."""
        return {
            "entities": await self._entities.count(),
            "relationships": await self._relationships.count(),
            "by_kind": await self._entities.count_by_kind(),
        }


def _reverse_is_meaningful(kind: RelationKind) -> bool:
    """Whether a shock travels backwards along this relation.

    Supply relationships conduct in both directions, and that is not a
    simplification. If NVIDIA's volumes rise its suppliers benefit; if a
    supplier cannot deliver, NVIDIA suffers. An arrow that only conducted
    forwards would miss every supply-disruption question -- which is half of
    what anyone wants to ask a supply chain.
    """
    return kind in {
        RelationKind.SUPPLIES,
        RelationKind.MANUFACTURES,
        RelationKind.CUSTOMER_OF,
        RelationKind.DEPENDS_ON,
        RelationKind.PRODUCES,
    }


#: Readable connective for each relation, in both reading directions.
#:
#: The reverse form is not decoration. Traversing "TSMC manufactures for NVIDIA"
#: backwards and still printing "manufactures for" produced the line
#: "NVIDIA → manufactures for → TSMC", which states the relationship the wrong
#: way round in the one place the platform is meant to be explaining itself.
_VERBS: dict[RelationKind, tuple[str, str]] = {
    RelationKind.SUPPLIES: ("→ supplies →", "→ is supplied by →"),
    RelationKind.MANUFACTURES: ("→ manufactures for →", "→ is manufactured by →"),
    RelationKind.CUSTOMER_OF: ("→ buys from →", "→ sells to →"),
    RelationKind.DEPENDS_ON: ("→ depends on →", "→ is depended on by →"),
    RelationKind.PRODUCES: ("→ produces →", "→ is produced by →"),
    RelationKind.USES: ("→ uses →", "→ is used by →"),
    RelationKind.DEPLOYS: ("→ deploys →", "→ is deployed by →"),
    RelationKind.PARTNERS_WITH: ("→ partners with →", "→ partners with →"),
    RelationKind.INVESTS_IN: ("→ invests in →", "→ is backed by →"),
    RelationKind.COMPETES_WITH: ("→ competes with →", "→ competes with →"),
    RelationKind.ACQUIRED: ("→ acquired →", "→ was acquired by →"),
    RelationKind.OPERATES: ("→ operates →", "→ is operated by →"),
    RelationKind.LOCATED_IN: ("→ located in →", "→ hosts →"),
}


def _verb(kind: RelationKind, *, forwards: bool) -> str:
    """Return the readable connective, phrased for the direction travelled."""
    pair = _VERBS.get(kind)
    if pair is None:  # pragma: no cover - the mapping is exhaustive
        return f"→ {kind.value} →"
    return pair[0] if forwards else pair[1]
