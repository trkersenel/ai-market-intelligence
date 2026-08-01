"""The knowledge graph: entities and the typed relationships between them.

This is the table that makes the platform something other than a stock screener.
A screener can tell you Micron is up 18%; only a graph can tell you that Micron
supplies HBM to NVIDIA, that NVIDIA's parts are fabricated by TSMC, and that
TSMC's lithography comes from a single supplier in the Netherlands -- which is
what turns "up 18%" into an explanation.

**Why PostgreSQL rather than a graph database.** The obvious answer is Neo4j,
and for a graph of hundreds of millions of edges it would be the right one.
This graph is a few hundred entities and a few thousand edges: the entire thing
fits in a page of memory, and a recursive CTE traverses it in under a
millisecond. Adding a second stateful service to the deployment would cost more
in operations than it could possibly return in query time. The traversal is
already isolated behind a repository, so the day the graph outgrows this the
change is one class.

**Every edge carries its provenance and its confidence.** That is not
bookkeeping -- it is the product requirement. A platform that says "NVIDIA
depends on TSMC" without saying how it knows is asking to be believed; one that
answers "curated from TSMC's published customer disclosures, confidence 0.95"
can be checked. Edges proposed by a model are stored with their source and a
lower confidence, and are visibly distinguishable from curated ones.

**Edges are temporal.** ``valid_from`` and ``valid_to`` mean the graph records
that Microsoft's OpenAI investment began in 2019 and that Intel stopped
supplying Apple in 2020, rather than presenting a snapshot as though it were
timeless. Asking what the ecosystem looked like a year ago is then a WHERE
clause instead of a rebuild.
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import (
    CheckConstraint,
    Date,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, IntIdMixin, TimestampMixin
from app.models.enums import EntityKind, EvidenceSource, RelationKind, pg_enum


class Entity(IntIdMixin, TimestampMixin, Base):
    """A node: a company, technology, product, facility, country or person.

    Deliberately wider than "company". The AI ecosystem's most consequential
    actors include one that has never been listed (OpenAI), one that is a
    manufacturing process rather than an organisation (EUV lithography), and one
    that is a building (TSMC's Arizona fab). A graph restricted to tickers could
    not express why any of the three matters.
    """

    __table_args__ = (
        Index("ix_entity_kind", "kind"),
        Index("ix_entity_tags_gin", "tags", postgresql_using="gin"),
        {"comment": "Nodes of the AI ecosystem knowledge graph."},
    )

    #: Stable, human-readable identifier: "nvidia", "tsmc", "euv-lithography".
    #: Used in URLs and in seed data, so it must not change once published.
    slug: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    kind: Mapped[EntityKind] = mapped_column(pg_enum(EntityKind, "entity_kind"))

    #: Exchange ticker when the entity is publicly listed. Nullable by design:
    #: OpenAI, Anthropic and TSMC's Arizona fab are all first-class nodes with
    #: no symbol, and forcing one would either exclude them or invent it.
    symbol: Mapped[str | None] = mapped_column(String(20), index=True)

    #: ISO 3166-1 alpha-2. The supply chain is a geopolitical object as much as
    #: an industrial one -- "which of these sit in Taiwan" is a real question.
    country: Mapped[str | None] = mapped_column(String(2), index=True)

    #: Layer of the stack: "foundry", "hbm", "gpu", "cloud", "power", "cooling",
    #: "networking", "eda", "foundation-model". A GIN-indexed array so
    #: "every cooling company" is one query with no join table.
    tags: Mapped[list[str]] = mapped_column(ARRAY(String(40)), default=list, server_default="{}")

    summary: Mapped[str | None] = mapped_column(Text)

    outgoing: Mapped[list[Relationship]] = relationship(
        back_populates="source",
        foreign_keys="Relationship.source_id",
        cascade="all, delete-orphan",
        lazy="raise_on_sql",
    )
    incoming: Mapped[list[Relationship]] = relationship(
        back_populates="target",
        foreign_keys="Relationship.target_id",
        cascade="all, delete-orphan",
        lazy="raise_on_sql",
    )

    def __repr__(self) -> str:
        """Return a debugging representation."""
        return f"<Entity {self.slug!r} kind={self.kind.value}>"


class Relationship(IntIdMixin, TimestampMixin, Base):
    """A directed, typed, sourced, time-bounded edge between two entities."""

    __table_args__ = (
        # One edge per (source, target, kind, start date). The start date is in
        # the key so a relationship that lapses and later resumes is two rows
        # rather than one row that silently overwrites its own history.
        UniqueConstraint(
            "source_id", "target_id", "kind", "valid_from", name="uq_relationship_natural"
        ),
        CheckConstraint("source_id <> target_id", name="no_self_relationship"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_is_a_probability"),
        CheckConstraint("weight >= 0 AND weight <= 1", name="weight_is_normalised"),
        Index("ix_relationship_source_kind", "source_id", "kind"),
        Index("ix_relationship_target_kind", "target_id", "kind"),
        {"comment": "Typed, sourced, temporal edges of the knowledge graph."},
    )

    source_id: Mapped[int] = mapped_column(ForeignKey("entity.id", ondelete="CASCADE"), index=True)
    target_id: Mapped[int] = mapped_column(ForeignKey("entity.id", ondelete="CASCADE"), index=True)
    kind: Mapped[RelationKind] = mapped_column(pg_enum(RelationKind, "relation_kind"))

    #: How sure the platform is that this relationship exists at all. A curated
    #: edge from a company's own 10-K sits near 1.0; one a model proposed from a
    #: news article sits far lower, and the UI renders the difference.
    confidence: Mapped[float] = mapped_column(Float, default=1.0, server_default="1.0")

    #: How much this relationship *matters* to the source, which is a different
    #: question from whether it exists. TSMC supplies both NVIDIA and a hundred
    #: small fabless firms; the edges are equally certain and nothing like
    #: equally material. Impact propagation multiplies by this, not confidence.
    weight: Mapped[float] = mapped_column(Float, default=0.5, server_default="0.5")

    #: Where this claim came from, and how to check it.
    source_kind: Mapped[EvidenceSource] = mapped_column(
        pg_enum(EvidenceSource, "evidence_source"),
        default=EvidenceSource.CURATED,
    )
    citation: Mapped[str | None] = mapped_column(Text)

    #: Temporal validity. ``valid_to`` NULL means "still true as far as the
    #: platform knows" -- distinct from "true forever", which nothing is.
    valid_from: Mapped[date | None] = mapped_column(Date)
    valid_to: Mapped[date | None] = mapped_column(Date)

    #: One line a person can read, used verbatim in the UI and as evidence in
    #: generated analysis. Written at curation time rather than composed from
    #: the enum, because "fabricates NVIDIA's GPUs on N4 and N3" carries the
    #: information that "MANUFACTURES" does not.
    description: Mapped[str | None] = mapped_column(Text)

    source: Mapped[Entity] = relationship(
        back_populates="outgoing", foreign_keys=[source_id], lazy="raise_on_sql"
    )
    target: Mapped[Entity] = relationship(
        back_populates="incoming", foreign_keys=[target_id], lazy="raise_on_sql"
    )

    def __repr__(self) -> str:
        """Return a debugging representation."""
        return f"<Relationship {self.source_id}-[{self.kind.value}]->{self.target_id}>"
