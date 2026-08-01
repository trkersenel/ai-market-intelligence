"""Loading the curated graph into PostgreSQL."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.logging import get_logger
from app.models.graph import Entity, Relationship
from app.repositories.graph import EntityRepository
from app.services.graph.seed import ENTITIES, RELATIONS

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SeedReport:
    """What one seeding run wrote."""

    entities: int
    relations: int
    skipped: tuple[str, ...] = ()


class GraphSeeder:
    """Loads the curated ecosystem, idempotently."""

    def __init__(self, entities: EntityRepository) -> None:
        """Bind the seeder to a unit of work."""
        self._entities = entities
        self._session = entities.session

    async def seed(self) -> SeedReport:
        """Insert or refresh every curated entity and relationship.

        Idempotent: running twice leaves the same graph. Curated data is the
        backbone every inferred edge is checked against, so re-running after
        editing the seed table must converge rather than accumulate duplicates.

        Returns:
            Counts, plus any relation whose endpoints could not be resolved --
            surfaced rather than silently dropped, because a typo in a slug
            would otherwise remove an edge from the graph invisibly.
        """
        for entity in ENTITIES:
            statement = pg_insert(Entity).values(
                slug=entity.slug,
                name=entity.name,
                kind=entity.kind,
                symbol=entity.symbol,
                country=entity.country,
                tags=list(entity.tags),
                summary=entity.summary,
            )
            await self._session.execute(
                statement.on_conflict_do_update(
                    index_elements=[Entity.slug],
                    set_={
                        column: statement.excluded[column]
                        for column in ("name", "kind", "symbol", "country", "tags", "summary")
                    },
                )
            )
        await self._session.flush()

        ids = {
            slug: entity_id
            for entity_id, slug in (
                await self._session.execute(select(Entity.id, Entity.slug))
            ).all()
        }

        written = 0
        skipped: list[str] = []
        for relation in RELATIONS:
            source_id = ids.get(relation.source)
            target_id = ids.get(relation.target)
            if source_id is None or target_id is None:
                skipped.append(f"{relation.source} -> {relation.target}")
                continue

            statement = pg_insert(Relationship).values(
                source_id=source_id,
                target_id=target_id,
                kind=relation.kind,
                confidence=relation.confidence,
                weight=relation.weight,
                source_kind=relation.source_kind,
                citation=relation.citation,
                valid_from=relation.valid_from,
                valid_to=relation.valid_to,
                description=relation.description,
            )
            await self._session.execute(
                statement.on_conflict_do_update(
                    # Matches the natural key, which includes valid_from so a
                    # lapsed-and-resumed relationship stays two rows.
                    index_elements=[
                        Relationship.source_id,
                        Relationship.target_id,
                        Relationship.kind,
                        Relationship.valid_from,
                    ],
                    set_={
                        column: statement.excluded[column]
                        for column in (
                            "confidence",
                            "weight",
                            "source_kind",
                            "citation",
                            "valid_to",
                            "description",
                        )
                    },
                )
            )
            written += 1

        if skipped:
            logger.warning("graph_seed_unresolved", pairs=skipped)
        logger.info("graph_seeded", entities=len(ENTITIES), relations=written, skipped=len(skipped))
        return SeedReport(entities=len(ENTITIES), relations=written, skipped=tuple(skipped))
