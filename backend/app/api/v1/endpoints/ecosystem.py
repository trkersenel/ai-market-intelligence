"""AI ecosystem graph endpoints.

The surface behind the ecosystem map, the supply chain view and impact
analysis. Every response carries provenance and confidence alongside the shape,
because a relationship diagram that cannot be checked is decoration.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query, status

from app.api.deps import GraphServiceDep
from app.core.exceptions import NotFoundError
from app.models.enums import EntityKind
from app.schemas.graph import (
    EcosystemGraph,
    EntityNode,
    GraphStats,
    ImpactAnalysis,
    ImpactedEntity,
    RelationshipEdge,
)

router = APIRouter(tags=["ecosystem"])

#: Expected for most of the 5,664 browsable listings: the graph covers the AI
#: ecosystem, not the exchange, and a symbol with no node is a coverage limit
#: rather than a fault.
_NOT_IN_GRAPH: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"description": "The graph has no node for this identifier."}
}


@router.get("/stats", response_model=GraphStats, summary="Size of the knowledge graph")
async def get_stats(graph: GraphServiceDep) -> GraphStats:
    """Return how many entities and relationships the graph holds."""
    return GraphStats.model_validate(await graph.stats())


@router.get(
    "/entities",
    response_model=list[EntityNode],
    summary="Browse the ecosystem by layer",
    description=(
        "Filter by node kind or by stack layer -- 'hbm', 'foundry', 'cooling', "
        "'semicap', 'foundation-model'. This is what turns 'data centre cooling "
        "companies' into a query rather than a search."
    ),
)
async def list_entities(
    graph: GraphServiceDep,
    kind: Annotated[list[EntityKind] | None, Query(description="Node kinds.")] = None,
    tag: Annotated[list[str] | None, Query(description="Stack layers, matched as any-of.")] = None,
) -> list[EntityNode]:
    """Return entities, optionally narrowed."""
    entities = await graph.list_entities(kinds=kind, tags=tag)
    return [EntityNode.from_model(entity) for entity in entities]


@router.get(
    "/search",
    response_model=list[EntityNode],
    summary="Search the ecosystem by name, ticker or layer",
)
async def search_entities(
    graph: GraphServiceDep,
    q: Annotated[str, Query(min_length=1, max_length=60)],
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> list[EntityNode]:
    """Return matching entities."""
    return [EntityNode.from_model(entity) for entity in await graph.search(q, limit=limit)]


@router.get(
    "/{identifier}",
    response_model=EcosystemGraph,
    summary="The ecosystem around one company",
    responses=_NOT_IN_GRAPH,
)
async def get_ecosystem(
    identifier: str,
    graph: GraphServiceDep,
    depth: Annotated[int, Query(ge=1, le=3, description="Hops to expand.")] = 1,
    min_confidence: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
) -> EcosystemGraph:
    """Return the neighbourhood around an entity.

    Accepts either a slug or a ticker, so a reader arriving from a company page
    needs no translation step.

    Raises:
        NotFoundError: If the graph has no node for this identifier. Expected
            for most of the 5,664 browsable listings -- the graph covers the AI
            ecosystem, not the exchange.
    """
    entity = await graph.get_entity(identifier)
    if entity is None:
        msg = f"{identifier} is not in the ecosystem graph."
        raise NotFoundError(msg, details={"identifier": identifier})

    edges = await graph.neighbourhood(entity, depth=depth, min_confidence=min_confidence)
    return EcosystemGraph.from_edges(entity, edges)


@router.get(
    "/{identifier}/supply-chain",
    response_model=list[RelationshipEdge],
    summary="Supply relationships only",
    responses=_NOT_IN_GRAPH,
)
async def get_supply_chain(identifier: str, graph: GraphServiceDep) -> list[RelationshipEdge]:
    """Return only the edges that form a supply chain.

    Competition and investment belong on the ecosystem map but not here: a
    supply chain diagram containing "competes with" arrows is a network diagram
    wearing the name.

    Raises:
        NotFoundError: If the graph has no node for this identifier.
    """
    entity = await graph.get_entity(identifier)
    if entity is None:
        msg = f"{identifier} is not in the ecosystem graph."
        raise NotFoundError(msg, details={"identifier": identifier})

    return [RelationshipEdge.from_edge(edge) for edge in await graph.supply_chain(entity)]


@router.get(
    "/{identifier}/impact",
    response_model=ImpactAnalysis,
    summary="Who benefits and who is at risk",
    description=(
        "Traces a shock outward from one company and scores everything it "
        "reaches, returning the path that produced each score. Scores are "
        "derived from curated relationships by an inspectable rule -- not "
        "predicted -- so a reader can disagree with the reasoning rather than "
        "with an oracle."
    ),
    responses=_NOT_IN_GRAPH,
)
async def get_impact(
    identifier: str,
    graph: GraphServiceDep,
    magnitude: Annotated[
        float,
        Query(ge=-1.0, le=1.0, description="Signed size. Negative models bad news."),
    ] = 1.0,
    depth: Annotated[int, Query(ge=1, le=4)] = 3,
    min_confidence: Annotated[float, Query(ge=0.0, le=1.0)] = 0.5,
    limit: Annotated[int, Query(ge=1, le=50)] = 12,
) -> ImpactAnalysis:
    """Return affected entities, split into winners and losers.

    Raises:
        NotFoundError: If the graph has no node for this identifier.
    """
    entity = await graph.get_entity(identifier)
    if entity is None:
        msg = f"{identifier} is not in the ecosystem graph."
        raise NotFoundError(msg, details={"identifier": identifier})

    results = await graph.propagate(
        entity, magnitude=magnitude, max_depth=depth, min_confidence=min_confidence
    )
    return ImpactAnalysis(
        origin=entity.slug,
        magnitude=magnitude,
        winners=[ImpactedEntity.from_result(r) for r in results if r.score > 0][:limit],
        losers=[ImpactedEntity.from_result(r) for r in results if r.score < 0][:limit],
    )
