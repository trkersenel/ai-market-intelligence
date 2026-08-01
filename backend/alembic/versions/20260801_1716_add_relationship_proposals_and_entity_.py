"""add relationship proposals and entity aliases

`relationship_proposal` is deliberately not `relationship`. The curated graph is
the thing model inferences get checked against, and it stops being that the
moment unreviewed inferences are written into it. Accepting a proposal copies it
across as an INFERRED edge; until then it is visible and queryable but does not
participate in traversal or impact propagation.

Each row stores the verbatim span of the source article that supports the claim.
That span was checked to actually appear in the document before the row was
written, so a reviewer is judging whether the sentence establishes the
relationship -- not whether the model invented the sentence.

`entity.aliases` feeds the mention detector. An article writing "Taiwan
Semiconductor Manufacturing" names TSMC, and a detector holding only the
canonical name drops the article from extraction silently.

Revision ID: 3d44693b732f
Revises: e287c1cfd9f1
Created: 2026-08-01 17:16:47.570702
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "3d44693b732f"
down_revision: str | None = "e287c1cfd9f1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the schema change."""
    op.create_table(
        "relationship_proposal",
        sa.Column("source_id", sa.BigInteger(), nullable=False),
        sa.Column("target_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "kind",
            # The dialect type with create_type=False, not sa.Enum: the
            # relation_kind type already exists from the knowledge-graph
            # migration, and autogenerate cannot tell a reused enum from a new
            # one. Re-declaring it fails with "type already exists".
            postgresql.ENUM(name="relation_kind", create_type=False),
            nullable=False,
        ),
        sa.Column("confidence", sa.Float(), server_default="0.5", nullable=False),
        sa.Column(
            "status",
            sa.Enum("pending", "accepted", "rejected", name="proposal_status"),
            nullable=False,
        ),
        sa.Column("quote", sa.Text(), nullable=False),
        sa.Column("document_id", sa.String(length=64), nullable=False),
        sa.Column("document_title", sa.Text(), nullable=True),
        sa.Column("document_url", sa.Text(), nullable=True),
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
            "source_id <> target_id", name=op.f("ck_relationship_proposal_no_self_proposal")
        ),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["entity.id"],
            name=op.f("fk_relationship_proposal_source_id_entity"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["target_id"],
            ["entity.id"],
            name=op.f("fk_relationship_proposal_target_id_entity"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_relationship_proposal")),
        sa.UniqueConstraint(
            "source_id", "target_id", "kind", "document_id", name="uq_proposal_natural"
        ),
        comment="Model-proposed edges awaiting human review.",
    )
    op.create_index(
        "ix_proposal_status_created",
        "relationship_proposal",
        ["status", "created_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_relationship_proposal_document_id"),
        "relationship_proposal",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_relationship_proposal_source_id"),
        "relationship_proposal",
        ["source_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_relationship_proposal_target_id"),
        "relationship_proposal",
        ["target_id"],
        unique=False,
    )
    op.add_column(
        "entity",
        sa.Column(
            "aliases", postgresql.ARRAY(sa.String(length=120)), server_default="{}", nullable=False
        ),
    )


def downgrade() -> None:
    """Revert the schema change."""
    op.drop_column("entity", "aliases")
    op.drop_index(op.f("ix_relationship_proposal_target_id"), table_name="relationship_proposal")
    op.drop_index(op.f("ix_relationship_proposal_source_id"), table_name="relationship_proposal")
    op.drop_index(op.f("ix_relationship_proposal_document_id"), table_name="relationship_proposal")
    op.drop_index("ix_proposal_status_created", table_name="relationship_proposal")
    op.drop_table("relationship_proposal")
