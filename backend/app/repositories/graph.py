"""Queries over the knowledge graph.

The traversal is a recursive CTE rather than repeated round trips. At this
graph's size either would be fast, but N round trips means N times the latency
and a neighbourhood assembled in Python from partial results -- and the cycle
handling would have to be written by hand, which is exactly where a graph
traversal goes wrong.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from sqlalchemy import Select, func, or_, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import aliased

from app.models.enums import SYMMETRIC_RELATIONS, EntityKind, ProposalStatus, RelationKind
from app.models.graph import Entity, Relationship, RelationshipProposal
from app.repositories.base import BaseRepository


@dataclass(frozen=True, slots=True)
class Edge:
    """One relationship with both endpoints resolved.

    Returned instead of the ORM object because callers always want the two
    entities as well, and ``lazy="raise_on_sql"`` on the model means reaching
    for them lazily is a deliberate error rather than a silent N+1.
    """

    relationship: Relationship
    source: Entity
    target: Entity

    def other_than(self, entity_id: int) -> Entity:
        """Return whichever endpoint is not ``entity_id``."""
        return self.target if self.source.id == entity_id else self.source

    @property
    def is_symmetric(self) -> bool:
        """Whether direction carries meaning for this relation kind."""
        return self.relationship.kind in SYMMETRIC_RELATIONS


@dataclass(frozen=True, slots=True)
class ProposalRow:
    """One proposal with both endpoints resolved.

    Its own type rather than reusing :class:`Edge`: a proposal has no
    provenance columns or validity window -- it has a quote and a status
    instead -- and pretending otherwise pushes a union into every caller.
    """

    proposal: RelationshipProposal
    source: Entity
    target: Entity


class EntityRepository(BaseRepository[Entity, int]):
    """Reads and writes over graph nodes."""

    model = Entity

    async def get_by_slug(self, slug: str) -> Entity | None:
        """Return one entity by its stable identifier."""
        result = await self._session.execute(select(Entity).where(Entity.slug == slug))
        return result.scalar_one_or_none()

    async def get_by_symbol(self, symbol: str) -> Entity | None:
        """Return the entity for a listed symbol, if the graph knows one.

        Most of the 5,664 browsable listings have no node: the graph covers the
        AI infrastructure ecosystem, not the exchange. Returning ``None`` is the
        normal answer for the rest, not an error.
        """
        result = await self._session.execute(
            select(Entity).where(func.upper(Entity.symbol) == symbol.upper())
        )
        return result.scalar_one_or_none()

    async def list_all(
        self,
        *,
        kinds: Sequence[EntityKind] | None = None,
        tags: Sequence[str] | None = None,
    ) -> Sequence[Entity]:
        """Return entities, optionally narrowed by kind or by stack layer."""
        statement: Select[tuple[Entity]] = select(Entity).order_by(Entity.name)
        if kinds:
            statement = statement.where(Entity.kind.in_(kinds))
        if tags:
            # Overlap rather than containment: "cooling or power" is the useful
            # question, and requiring every tag would return almost nothing.
            statement = statement.where(Entity.tags.overlap(list(tags)))
        result = await self._session.execute(statement)
        return result.scalars().all()

    async def search(self, query: str, *, limit: int = 20) -> Sequence[Entity]:
        """Return entities whose name, slug, symbol or tags match a fragment."""
        term = query.strip().lower()
        if not term:
            return []
        pattern = f"%{term}%"
        result = await self._session.execute(
            select(Entity)
            .where(
                or_(
                    func.lower(Entity.name).like(pattern),
                    func.lower(Entity.slug).like(pattern),
                    func.lower(func.coalesce(Entity.symbol, "")).like(pattern),
                    Entity.tags.overlap([term]),
                )
            )
            .order_by(func.length(Entity.name), Entity.name)
            .limit(limit)
        )
        return result.scalars().all()

    async def count_by_kind(self) -> dict[str, int]:
        """Return how many entities exist of each kind."""
        result = await self._session.execute(
            select(Entity.kind, func.count()).group_by(Entity.kind)
        )
        return {kind.value: count for kind, count in result.all()}


class RelationshipRepository(BaseRepository[Relationship, int]):
    """Reads and writes over graph edges."""

    model = Relationship

    def _hydrated(self) -> Select[tuple[Relationship, Entity, Entity]]:
        """Select edges with both endpoints joined."""
        source = aliased(Entity, name="src")
        target = aliased(Entity, name="tgt")
        return (
            select(Relationship, source, target)
            .join(source, Relationship.source_id == source.id)
            .join(target, Relationship.target_id == target.id)
        )

    @staticmethod
    def _as_of(
        statement: Select[tuple[Relationship, Entity, Entity]], moment: date | None
    ) -> Select[tuple[Relationship, Entity, Entity]]:
        """Restrict to edges valid at a point in time.

        The whole reason ``valid_from``/``valid_to`` exist. Without this the
        graph would present Microsoft's 2019 OpenAI investment and a supply
        relationship that ended in 2020 as equally current facts.
        """
        if moment is None:
            # "Current" means not yet ended. An edge with no start date is
            # treated as always-having-been-true, which is the right reading of
            # "TSMC manufactures for NVIDIA" -- nobody knows the first day.
            return statement.where(
                or_(Relationship.valid_to.is_(None), Relationship.valid_to >= func.current_date())
            )
        return statement.where(
            or_(Relationship.valid_from.is_(None), Relationship.valid_from <= moment),
            or_(Relationship.valid_to.is_(None), Relationship.valid_to >= moment),
        )

    async def neighbours(
        self,
        entity_id: int,
        *,
        kinds: Sequence[RelationKind] | None = None,
        as_of: date | None = None,
        min_confidence: float = 0.0,
    ) -> list[Edge]:
        """Return every edge touching one entity, in either direction.

        Both directions, because "who supplies NVIDIA" and "who does NVIDIA
        supply" are both things a reader clicking on NVIDIA wants, and making
        the caller issue two queries to assemble one neighbourhood would push
        graph mechanics into the service layer.
        """
        statement = self._hydrated().where(
            or_(Relationship.source_id == entity_id, Relationship.target_id == entity_id),
            Relationship.confidence >= min_confidence,
        )
        if kinds:
            statement = statement.where(Relationship.kind.in_(kinds))
        statement = self._as_of(statement, as_of)

        result = await self._session.execute(statement)
        return [Edge(rel, src, tgt) for rel, src, tgt in result.all()]

    async def subgraph(
        self,
        entity_id: int,
        *,
        depth: int = 2,
        min_confidence: float = 0.0,
    ) -> list[Edge]:
        """Return every edge within ``depth`` hops of an entity.

        A recursive CTE walks outward in both directions, tracking the visited
        set to terminate on cycles -- and this graph is full of them: NVIDIA
        depends on TSMC, TSMC's equipment comes from ASML, and ASML's own
        customers include the firms NVIDIA competes with. Without cycle
        handling the traversal does not merely repeat work, it does not stop.

        Args:
            entity_id: Where to start.
            depth: Hops to expand. Two is the useful default -- one shows direct
                partners, two shows the second-order exposure that is the point
                of drawing a supply chain at all.
            min_confidence: Drop edges the platform is not sure enough about.

        Returns:
            Deduplicated edges, each with both endpoints resolved.
        """
        # Written as text because SQLAlchemy's recursive CTE construction for a
        # bidirectional walk is markedly less readable than the SQL, and this
        # query is the one place where the graph's shape is actually expressed.
        walk = text("""
            WITH RECURSIVE reachable(id, depth) AS (
                -- Cast explicitly. An untyped bind in a recursive CTE's
                -- anchor row leaves PostgreSQL to infer the column type, and it
                -- infers `text` -- so the recursive term then compares
                -- `bigint = text` and the whole query fails to plan.
                SELECT CAST(:root AS BIGINT), 0
                UNION
                SELECT next_id, r.depth + 1
                FROM reachable r
                JOIN LATERAL (
                    SELECT CASE WHEN rel.source_id = r.id THEN rel.target_id
                                ELSE rel.source_id END AS next_id
                    FROM relationship rel
                    WHERE (rel.source_id = r.id OR rel.target_id = r.id)
                      AND rel.confidence >= CAST(:min_confidence AS DOUBLE PRECISION)
                      AND (rel.valid_to IS NULL OR rel.valid_to >= CURRENT_DATE)
                ) AS step ON TRUE
                WHERE r.depth < CAST(:depth AS INTEGER)
            )
            SELECT DISTINCT rel.id
            FROM relationship rel
            WHERE rel.source_id IN (SELECT id FROM reachable)
              AND rel.target_id IN (SELECT id FROM reachable)
              AND rel.confidence >= CAST(:min_confidence AS DOUBLE PRECISION)
              AND (rel.valid_to IS NULL OR rel.valid_to >= CURRENT_DATE)
        """)
        found = await self._session.execute(
            walk, {"root": entity_id, "depth": depth, "min_confidence": min_confidence}
        )
        edge_ids = [row[0] for row in found.all()]
        if not edge_ids:
            return []

        result = await self._session.execute(self._hydrated().where(Relationship.id.in_(edge_ids)))
        return [Edge(rel, src, tgt) for rel, src, tgt in result.all()]

    async def count(self) -> int:
        """Return how many edges the graph holds."""
        result = await self._session.execute(select(func.count()).select_from(Relationship))
        return int(result.scalar_one())


class ProposalRepository(BaseRepository[RelationshipProposal, int]):
    """Reads and writes over model-proposed edges awaiting review."""

    model = RelationshipProposal

    async def upsert_many(self, rows: Sequence[dict[str, object]]) -> int:
        """Store proposals, ignoring ones already recorded.

        ``DO NOTHING`` rather than ``DO UPDATE``: the natural key includes the
        source document, so a collision means this exact claim from this exact
        article was already proposed. Overwriting would silently reset a
        decision a reviewer has already made.
        """
        if not rows:
            return 0
        statement = pg_insert(RelationshipProposal).values(list(rows))
        return await self._execute_dml(
            statement.on_conflict_do_nothing(constraint="uq_proposal_natural")
        )

    async def list_pending(self, *, limit: int = 50) -> list[ProposalRow]:
        """Return proposals awaiting review, most confident first."""
        source = aliased(Entity, name="p_src")
        target = aliased(Entity, name="p_tgt")
        result = await self._session.execute(
            select(RelationshipProposal, source, target)
            .join(source, RelationshipProposal.source_id == source.id)
            .join(target, RelationshipProposal.target_id == target.id)
            .where(RelationshipProposal.status == ProposalStatus.PENDING)
            .order_by(RelationshipProposal.confidence.desc(), RelationshipProposal.created_at)
            .limit(limit)
        )
        return [ProposalRow(proposal, src, tgt) for proposal, src, tgt in result.all()]

    async def counts_by_status(self) -> dict[str, int]:
        """Return how many proposals sit in each state."""
        result = await self._session.execute(
            select(RelationshipProposal.status, func.count()).group_by(RelationshipProposal.status)
        )
        return {status.value: count for status, count in result.all()}
