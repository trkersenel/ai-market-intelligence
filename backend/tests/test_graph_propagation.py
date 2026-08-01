"""Tests for impact propagation.

The rules under test are the ones that separate a defensible causal chain from a
plausible-looking one. Each was written after the propagation produced an answer
that was obviously wrong to anyone who knows the industry -- which is the only
reliable way to find a bug in a scoring function, since every version of it
returns a confident ranked list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.models.enums import RelationKind
from app.services.graph.service import Flow, GraphService, _flow_of, _verb


@dataclass
class FakeEntity:
    """Stands in for the ORM node."""

    id: int
    slug: str
    name: str
    symbol: str | None = None


@dataclass
class FakeRelation:
    """Stands in for the ORM edge."""

    source_id: int
    target_id: int
    kind: RelationKind
    weight: float = 1.0
    confidence: float = 1.0


@dataclass
class FakeEdge:
    """Stands in for the repository's hydrated edge."""

    relationship: FakeRelation
    source: FakeEntity
    target: FakeEntity


class FakeRelationshipRepository:
    """Serves a fixed edge list regardless of the traversal requested."""

    def __init__(self, edges: list[FakeEdge]) -> None:
        self._edges = edges

    async def subgraph(self, entity_id: int, **_: Any) -> list[FakeEdge]:
        return self._edges

    async def neighbours(self, entity_id: int, **_: Any) -> list[FakeEdge]:
        return [
            edge
            for edge in self._edges
            if entity_id in (edge.relationship.source_id, edge.relationship.target_id)
        ]


def _service(edges: list[FakeEdge]) -> GraphService:
    return GraphService(entities=None, relationships=FakeRelationshipRepository(edges))  # type: ignore[arg-type]


def _scores(results: list[Any]) -> dict[str, float]:
    return {result.entity.slug: result.score for result in results}


# A minimal slice of the real ecosystem, enough to exercise every rule.
MSFT = FakeEntity(1, "microsoft", "Microsoft", "MSFT")
NVDA = FakeEntity(2, "nvidia", "NVIDIA", "NVDA")
TSMC = FakeEntity(3, "tsmc", "TSMC", "TSM")
AMZN = FakeEntity(4, "amazon", "Amazon", "AMZN")
AMD = FakeEntity(5, "amd", "AMD", "AMD")
ASML = FakeEntity(6, "asml", "ASML", "ASML")
SAMSUNG = FakeEntity(7, "samsung", "Samsung", None)


class TestDirectionOfTravel:
    """A shock travels one way along the chain."""

    async def test_demand_reaches_suppliers_upstream(self) -> None:
        """The core case: a customer spending more lifts its supply chain."""
        edges = [
            FakeEdge(FakeRelation(MSFT.id, NVDA.id, RelationKind.CUSTOMER_OF, 0.9), MSFT, NVDA),
            FakeEdge(FakeRelation(TSMC.id, NVDA.id, RelationKind.MANUFACTURES, 0.95), TSMC, NVDA),
        ]

        results = await _service(edges).propagate(MSFT, magnitude=1.0)  # type: ignore[arg-type]
        scores = _scores(results)

        assert scores["nvidia"] > 0
        assert scores["tsmc"] > 0, "the shock must reach the foundry two hops up"

    async def test_a_path_may_not_reverse_direction(self) -> None:
        """The bug this rule exists for.

        A TSMC production cut reached Samsung as a *loss* via
        TSMC → ASML → Samsung: ASML sells less to TSMC, so ASML is hurt, so
        ASML's other customers must be hurt too. Samsung is TSMC's competitor;
        a TSMC cut is if anything good for it. Going upstream and then back
        down is a different mechanism -- capacity reallocation -- not the same
        shock continuing.
        """
        edges = [
            FakeEdge(FakeRelation(ASML.id, TSMC.id, RelationKind.SUPPLIES, 0.95), ASML, TSMC),
            FakeEdge(FakeRelation(ASML.id, SAMSUNG.id, RelationKind.SUPPLIES, 0.85), ASML, SAMSUNG),
        ]

        results = await _service(edges).propagate(TSMC, magnitude=-1.0)  # type: ignore[arg-type]
        scores = _scores(results)

        assert scores["asml"] < 0, "ASML loses a customer's volume"
        assert "samsung" not in scores, "must not continue back down into ASML's other customers"


class TestCompetition:
    """Sign-flipping hops."""

    async def test_a_competitor_is_scored_in_the_opposite_direction(self) -> None:
        edges = [
            FakeEdge(FakeRelation(NVDA.id, AMD.id, RelationKind.COMPETES_WITH, 0.9), NVDA, AMD),
        ]

        results = await _service(edges).propagate(NVDA, magnitude=1.0)  # type: ignore[arg-type]

        assert _scores(results)["amd"] < 0

    async def test_a_competitive_hop_does_not_continue(self) -> None:
        """The bug that knocked the obvious beneficiary off the list.

        "Microsoft → competes with → Amazon → buys from → NVIDIA" scored NVIDIA
        *negatively* on news that Microsoft was spending more -- reasoning that
        Amazon loses, so Amazon's suppliers lose. NVIDIA sells to both. The net
        pushed the single most obvious beneficiary out of the top results
        entirely.
        """
        edges = [
            FakeEdge(FakeRelation(MSFT.id, NVDA.id, RelationKind.CUSTOMER_OF, 0.9), MSFT, NVDA),
            FakeEdge(FakeRelation(MSFT.id, AMZN.id, RelationKind.COMPETES_WITH, 0.85), MSFT, AMZN),
            FakeEdge(FakeRelation(AMZN.id, NVDA.id, RelationKind.CUSTOMER_OF, 0.8), AMZN, NVDA),
        ]

        results = await _service(edges).propagate(MSFT, magnitude=1.0)  # type: ignore[arg-type]
        scores = _scores(results)

        assert scores["amazon"] < 0, "the competitor itself is still scored"
        assert scores["nvidia"] > 0, "a supplier to both must not be penalised"
        assert scores["nvidia"] == pytest.approx(0.63), "and must keep the undiluted direct score"


class TestScoring:
    """How magnitude, weight and confidence combine."""

    async def test_a_negative_shock_inverts_every_downstream_sign(self) -> None:
        edges = [
            FakeEdge(FakeRelation(TSMC.id, NVDA.id, RelationKind.MANUFACTURES, 0.95), TSMC, NVDA),
        ]

        good = await _service(edges).propagate(TSMC, magnitude=1.0)  # type: ignore[arg-type]
        bad = await _service(edges).propagate(TSMC, magnitude=-1.0)  # type: ignore[arg-type]

        assert _scores(good)["nvidia"] == pytest.approx(-_scores(bad)["nvidia"])

    async def test_weight_scales_the_effect(self) -> None:
        """Existence and materiality are different questions.

        TSMC supplies both NVIDIA and a hundred small fabless firms with equal
        certainty and nothing like equal materiality.
        """
        strong = [FakeEdge(FakeRelation(1, 2, RelationKind.SUPPLIES, 0.9), MSFT, NVDA)]
        weak = [FakeEdge(FakeRelation(1, 2, RelationKind.SUPPLIES, 0.2), MSFT, NVDA)]

        heavy = _scores(await _service(strong).propagate(MSFT, magnitude=1.0))  # type: ignore[arg-type]
        light = _scores(await _service(weak).propagate(MSFT, magnitude=1.0))  # type: ignore[arg-type]

        assert heavy["nvidia"] > light["nvidia"]

    async def test_confidence_is_the_weakest_link(self) -> None:
        """A chain is only as trustworthy as its least certain step.

        A minimum rather than a product: multiplying would make a five-step
        chain of well-sourced edges look less certain than a single shaky one.
        """
        edges = [
            FakeEdge(
                FakeRelation(MSFT.id, NVDA.id, RelationKind.CUSTOMER_OF, 0.9, confidence=0.9),
                MSFT,
                NVDA,
            ),
            FakeEdge(
                FakeRelation(TSMC.id, NVDA.id, RelationKind.MANUFACTURES, 0.95, confidence=0.6),
                TSMC,
                NVDA,
            ),
        ]

        results = await _service(edges).propagate(MSFT, magnitude=1.0, min_confidence=0.0)  # type: ignore[arg-type]
        by_slug = {result.entity.slug: result for result in results}

        assert by_slug["nvidia"].confidence == pytest.approx(0.9)
        assert by_slug["tsmc"].confidence == pytest.approx(0.6)

    async def test_an_immaterial_path_is_dropped(self) -> None:
        """Without a floor, long chains produce a tail of true but useless rows."""
        edges = [FakeEdge(FakeRelation(1, 2, RelationKind.SUPPLIES, 0.01), MSFT, NVDA)]

        results = await _service(edges).propagate(MSFT, magnitude=1.0)  # type: ignore[arg-type]

        assert results == []

    async def test_the_origin_never_scores_itself(self) -> None:
        edges = [
            FakeEdge(FakeRelation(MSFT.id, NVDA.id, RelationKind.CUSTOMER_OF, 0.9), MSFT, NVDA),
            FakeEdge(FakeRelation(NVDA.id, MSFT.id, RelationKind.SUPPLIES, 0.9), NVDA, MSFT),
        ]

        results = await _service(edges).propagate(MSFT, magnitude=1.0)  # type: ignore[arg-type]

        assert "microsoft" not in _scores(results)


class TestReadableDirection:
    """The prose the platform shows for each step."""

    @pytest.mark.parametrize(
        ("kind", "forwards", "expected"),
        [
            (RelationKind.MANUFACTURES, True, "→ manufactures for →"),
            (RelationKind.MANUFACTURES, False, "→ is manufactured by →"),
            (RelationKind.SUPPLIES, True, "→ supplies →"),
            (RelationKind.SUPPLIES, False, "→ is supplied by →"),
            (RelationKind.CUSTOMER_OF, True, "→ buys from →"),
            (RelationKind.CUSTOMER_OF, False, "→ sells to →"),
        ],
    )
    def test_a_reversed_step_is_phrased_in_reverse(
        self, kind: RelationKind, forwards: bool, expected: str
    ) -> None:
        """A step travelled backwards is phrased backwards.

        Keeping the forward verb printed "NVIDIA → manufactures for → TSMC",
        stating the relationship the wrong way round in the one place the
        platform is meant to be explaining itself.
        """
        assert _verb(kind, forwards=forwards) == expected

    @pytest.mark.parametrize(
        ("kind", "forwards", "expected"),
        [
            # The arrow points supplier -> customer, so following it is downstream.
            (RelationKind.SUPPLIES, True, Flow.DOWNSTREAM),
            (RelationKind.SUPPLIES, False, Flow.UPSTREAM),
            # This one points customer -> supplier, so following it is upstream.
            (RelationKind.CUSTOMER_OF, True, Flow.UPSTREAM),
            (RelationKind.CUSTOMER_OF, False, Flow.DOWNSTREAM),
            (RelationKind.COMPETES_WITH, True, Flow.LATERAL),
            (RelationKind.COMPETES_WITH, False, Flow.LATERAL),
        ],
    )
    def test_flow_is_normalised_across_relation_kinds(
        self, kind: RelationKind, forwards: bool, expected: Flow
    ) -> None:
        """Economic direction is normalised across relation kinds.

        Raw arrow direction says nothing about it: SUPPLIES and CUSTOMER_OF
        point opposite ways along the same chain.
        """
        assert _flow_of(kind, forwards=forwards) is expected
