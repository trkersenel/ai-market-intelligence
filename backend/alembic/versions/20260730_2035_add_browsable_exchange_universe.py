"""add browsable exchange universe

Adds the table backing symbol search across every listing on the exchange --
roughly 5,700 rows for NASDAQ. Kept separate from ``ticker``, which holds the
few dozen symbols the platform actually spends API quota analysing: the
scheduler iterates tickers on every run, and ``ticker`` requires a parent
company for equities, so merging the two would mean both an all-day ingestion
run and thousands of empty ``company`` rows.

The table is disposable. It is rebuilt from the provider's symbol file by the
daily sync, so a downgrade loses nothing that one run does not restore.

Revision ID: 51bb8426cf10
Revises: af2648749c96
Created: 2026-07-30 20:35:59.313344
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "51bb8426cf10"
down_revision: str | None = "af2648749c96"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the schema change."""
    op.create_table(
        "listing",
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("exchange", sa.String(length=20), server_default="", nullable=False),
        sa.Column("currency", sa.String(length=3), server_default="USD", nullable=False),
        sa.Column("security_type", sa.String(length=40), server_default="common", nullable=False),
        sa.Column("figi", sa.String(length=20), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("source", sa.String(length=30), server_default="", nullable=False),
        sa.Column("synced_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.PrimaryKeyConstraint("id", name=op.f("pk_listing")),
        comment="Browsable exchange universe; tracked symbols also exist in ticker.",
    )
    op.create_index("ix_listing_exchange", "listing", ["exchange"], unique=False)
    op.create_index(op.f("ix_listing_figi"), "listing", ["figi"], unique=False)
    op.create_index(op.f("ix_listing_symbol"), "listing", ["symbol"], unique=True)
    # ### end Alembic commands ###


def downgrade() -> None:
    """Revert the schema change."""
    op.drop_index(op.f("ix_listing_symbol"), table_name="listing")
    op.drop_index(op.f("ix_listing_figi"), table_name="listing")
    op.drop_index("ix_listing_exchange", table_name="listing")
    op.drop_table("listing")
    # ### end Alembic commands ###
