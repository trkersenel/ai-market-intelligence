"""Schemas for the AI ecosystem knowledge graph."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from app.models.graph import Entity
from app.repositories.graph import Edge
from app.services.graph import ImpactResult


class EntityNode(BaseModel):
    """One node, as the graph view renders it."""

    model_config = ConfigDict(from_attributes=True)

    slug: str
    name: str
    kind: str
    symbol: str | None = None
    country: str | None = None
    tags: list[str] = Field(default_factory=list)
    summary: str | None = None

    @classmethod
    def from_model(cls, entity: Entity) -> EntityNode:
        """Build from the ORM node."""
        return cls(
            slug=entity.slug,
            name=entity.name,
            kind=entity.kind.value,
            symbol=entity.symbol,
            country=entity.country,
            tags=list(entity.tags),
            summary=entity.summary,
        )


class RelationshipEdge(BaseModel):
    """One edge, with the provenance that makes it checkable."""

    source: str
    target: str
    kind: str
    description: str | None = None
    weight: float
    confidence: float
    #: How the platform knows. Rendered, not hidden: a curated edge from a 10-K
    #: and one a model proposed from a headline must never look alike.
    evidence: str
    citation: str | None = None
    valid_from: date | None = None
    valid_to: date | None = None

    @classmethod
    def from_edge(cls, edge: Edge) -> RelationshipEdge:
        """Build from the repository's hydrated edge."""
        relation = edge.relationship
        return cls(
            source=edge.source.slug,
            target=edge.target.slug,
            kind=relation.kind.value,
            description=relation.description,
            weight=relation.weight,
            confidence=relation.confidence,
            evidence=relation.source_kind.value,
            citation=relation.citation,
            valid_from=relation.valid_from,
            valid_to=relation.valid_to,
        )


class EcosystemGraph(BaseModel):
    """A neighbourhood: the nodes and edges a view needs to draw itself."""

    root: str
    nodes: list[EntityNode]
    edges: list[RelationshipEdge]

    @classmethod
    def from_edges(cls, root: Entity, edges: list[Edge]) -> EcosystemGraph:
        """Assemble from a traversal, deduplicating the nodes it touched.

        The root is included even when it has no edges, so a node the graph
        knows about but has not yet connected renders as an isolated point
        rather than an empty canvas.
        """
        nodes: dict[str, Entity] = {root.slug: root}
        for edge in edges:
            nodes[edge.source.slug] = edge.source
            nodes[edge.target.slug] = edge.target
        return cls(
            root=root.slug,
            nodes=[EntityNode.from_model(entity) for entity in nodes.values()],
            edges=[RelationshipEdge.from_edge(edge) for edge in edges],
        )


class ImpactPathResponse(BaseModel):
    """One route by which a shock reached an entity."""

    steps: list[str]
    score: float
    confidence: float


class ImpactedEntity(BaseModel):
    """An affected company, with the reasoning that reached it."""

    entity: EntityNode
    score: float
    confidence: float
    direction: str = Field(description="benefits, at_risk or neutral.")
    paths: list[ImpactPathResponse]

    @classmethod
    def from_result(cls, result: ImpactResult) -> ImpactedEntity:
        """Build from the propagation result."""
        return cls(
            entity=EntityNode.from_model(result.entity),
            score=round(result.score, 4),
            confidence=round(result.confidence, 3),
            direction=result.direction,
            paths=[
                ImpactPathResponse(
                    steps=list(path.steps),
                    score=round(path.score, 4),
                    confidence=round(path.confidence, 3),
                )
                for path in result.paths
            ],
        )


class ImpactAnalysis(BaseModel):
    """Who a shock reaches, split by direction."""

    origin: str
    magnitude: float
    winners: list[ImpactedEntity]
    losers: list[ImpactedEntity]

    #: Stated rather than implied. Every score here is derived from curated
    #: edges by an inspectable rule, not predicted -- and a company absent from
    #: the graph is absent from the answer, which is a coverage limit rather
    #: than a judgement that it is unaffected.
    method: str = "graph_propagation"


class GraphStats(BaseModel):
    """Size and composition of the graph."""

    entities: int
    relationships: int
    by_kind: dict[str, int]
