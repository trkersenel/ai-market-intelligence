"""mark provisional price bars

Adds a flag distinguishing a completed trading session from one still in
progress. Vendors return the current day's bar mid-session with partial volume
and a close that is not yet the close, and every statistic derived from such a
bar is wrong in a way that looks like a signal -- a volume ratio of 0.13 reads
as a collapse in participation rather than as lunchtime.

Backfill: existing rows default to ``false``. That is correct for all but
possibly the most recent session per ticker, which the next ingestion run
re-fetches and rewrites through the usual overlapping window.

Revision ID: af2648749c96
Revises: efe849285eef
Created: 2026-07-29 19:12:26.500487
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "af2648749c96"
down_revision: str | None = "efe849285eef"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the provisional flag to daily_price."""
    op.add_column(
        "daily_price",
        sa.Column(
            "is_provisional",
            sa.Boolean(),
            server_default="false",
            nullable=False,
            comment="True while the session is still trading; its OHLCV is partial.",
        ),
    )


def downgrade() -> None:
    """Drop the provisional flag."""
    op.drop_column("daily_price", "is_provisional")
