"""add ai ecosystem knowledge graph

The tables that make this a research platform rather than a screener. A screener
reports that Micron is up 18%; only a graph can say that Micron supplies HBM to
NVIDIA, that NVIDIA's silicon is fabricated by TSMC, and that TSMC's lithography
has exactly one supplier -- which is what turns a number into an explanation.

Two columns carry most of the design. `confidence` records how sure the platform
is that an edge exists, and `source_kind` records where the claim came from, so a
curated edge from a 10-K and one a model proposed from a headline never look
alike to a reader. `valid_from`/`valid_to` make the graph temporal: Microsoft's
OpenAI investment began in 2019, and asking what the ecosystem looked like a year
ago becomes a WHERE clause instead of a rebuild.

Both tables are seeded from `app/services/graph/seed.py` and are safe to drop:
re-running the seeder restores them exactly.

Revision ID: e287c1cfd9f1
Revises: 51bb8426cf10
Created: 2026-08-01 16:35:19.679840
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "e287c1cfd9f1"
down_revision: str | None = "51bb8426cf10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the schema change."""
    op.create_table(
        "entity",
        sa.Column("slug", sa.String(length=120), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column(
            "kind",
            sa.Enum(
                "company",
                "organisation",
                "technology",
                "product",
                "ai_model",
                "facility",
                "country",
                "person",
                name="entity_kind",
            ),
            nullable=False,
        ),
        sa.Column("symbol", sa.String(length=20), nullable=True),
        sa.Column("country", sa.String(length=2), nullable=True),
        sa.Column(
            "tags", postgresql.ARRAY(sa.String(length=40)), server_default="{}", nullable=False
        ),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_entity")),
        comment="Nodes of the AI ecosystem knowledge graph.",
    )
    op.create_index(op.f("ix_entity_country"), "entity", ["country"], unique=False)
    op.create_index("ix_entity_kind", "entity", ["kind"], unique=False)
    op.create_index(op.f("ix_entity_slug"), "entity", ["slug"], unique=True)
    op.create_index(op.f("ix_entity_symbol"), "entity", ["symbol"], unique=False)
    op.create_index("ix_entity_tags_gin", "entity", ["tags"], unique=False, postgresql_using="gin")
    op.create_table(
        "relationship",
        sa.Column("source_id", sa.BigInteger(), nullable=False),
        sa.Column("target_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "kind",
            sa.Enum(
                "supplies",
                "manufactures",
                "customer_of",
                "competes_with",
                "partners_with",
                "depends_on",
                "uses",
                "produces",
                "invests_in",
                "acquired",
                "deploys",
                "operates",
                "located_in",
                name="relation_kind",
            ),
            nullable=False,
        ),
        sa.Column("confidence", sa.Float(), server_default="1.0", nullable=False),
        sa.Column("weight", sa.Float(), server_default="0.5", nullable=False),
        sa.Column(
            "source_kind",
            sa.Enum(
                "curated", "filing", "press_release", "news", "inferred", name="evidence_source"
            ),
            nullable=False,
        ),
        sa.Column("citation", sa.Text(), nullable=True),
        sa.Column("valid_from", sa.Date(), nullable=True),
        sa.Column("valid_to", sa.Date(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name=op.f("ck_relationship_confidence_is_a_probability"),
        ),
        sa.CheckConstraint(
            "source_id <> target_id", name=op.f("ck_relationship_no_self_relationship")
        ),
        sa.CheckConstraint(
            "weight >= 0 AND weight <= 1", name=op.f("ck_relationship_weight_is_normalised")
        ),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["entity.id"],
            name=op.f("fk_relationship_source_id_entity"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["target_id"],
            ["entity.id"],
            name=op.f("fk_relationship_target_id_entity"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_relationship")),
        sa.UniqueConstraint(
            "source_id", "target_id", "kind", "valid_from", name="uq_relationship_natural"
        ),
        comment="Typed, sourced, temporal edges of the knowledge graph.",
    )
    op.create_index(op.f("ix_relationship_source_id"), "relationship", ["source_id"], unique=False)
    op.create_index(
        "ix_relationship_source_kind", "relationship", ["source_id", "kind"], unique=False
    )
    op.create_index(op.f("ix_relationship_target_id"), "relationship", ["target_id"], unique=False)
    op.create_index(
        "ix_relationship_target_kind", "relationship", ["target_id", "kind"], unique=False
    )


def downgrade() -> None:
    """Revert the schema change."""
    op.drop_index("ix_relationship_target_kind", table_name="relationship")
    op.drop_index(op.f("ix_relationship_target_id"), table_name="relationship")
    op.drop_index("ix_relationship_source_kind", table_name="relationship")
    op.drop_index(op.f("ix_relationship_source_id"), table_name="relationship")
    op.drop_table("relationship")
    op.drop_index("ix_entity_tags_gin", table_name="entity", postgresql_using="gin")
    op.drop_index(op.f("ix_entity_symbol"), table_name="entity")
    op.drop_index(op.f("ix_entity_slug"), table_name="entity")
    op.drop_index("ix_entity_kind", table_name="entity")
    op.drop_index(op.f("ix_entity_country"), table_name="entity")
    op.drop_table("entity")
