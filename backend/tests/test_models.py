"""Schema-level invariants, asserted against ``Base.metadata``.

These need no database. They catch the class of mistake that otherwise only
surfaces when a migration is applied to a real PostgreSQL instance -- by which
point the migration is already written, reviewed and possibly merged.
"""

from __future__ import annotations

from collections import Counter

import pytest
from sqlalchemy import Enum as SqlEnum
from sqlalchemy import Float, Numeric

from app.db.seed_data import COMPANIES
from app.models import Base
from app.models.enums import AssetType, DetectionMethod, EcosystemTag, Severity
from app.repositories.anomaly import AnomalyRepository

#: Tables whose rows are written by ingestion and must be re-runnable.
IDEMPOTENT_TABLES = {
    "daily_price",
    "technical_indicator",
    "anomaly",
    "market_calendar",
    "daily_market_summary",
}


def test_constraint_and_index_names_are_unique_across_the_schema() -> None:
    """PostgreSQL puts constraints and indexes in one per-schema namespace.

    Two tables declaring a constraint with the same literal name is accepted by
    SQLAlchemy and only fails when the second CREATE TABLE runs. The naming
    convention prevents it by prefixing the table -- this asserts nobody has
    bypassed it with an explicit ``name=``.
    """
    names: list[str] = []
    for table in Base.metadata.sorted_tables:
        names.extend(
            constraint.name for constraint in table.constraints if constraint.name is not None
        )
        names.extend(index.name for index in table.indexes if index.name is not None)

    duplicates = [name for name, count in Counter(names).items() if count > 1]
    assert not duplicates, f"names collide across tables: {duplicates}"


@pytest.mark.parametrize("table_name", sorted(IDEMPOTENT_TABLES))
def test_ingested_tables_have_a_natural_unique_key(table_name: str) -> None:
    """Every re-runnable write target needs a key for ON CONFLICT to target.

    Without one, a retried ingestion job silently duplicates rows instead of
    updating them, and every downstream aggregate doubles.
    """
    table = Base.metadata.tables[table_name]
    unique_constraints = [
        constraint
        for constraint in table.constraints
        if constraint.__class__.__name__ == "UniqueConstraint"
    ]
    assert unique_constraints, f"{table_name} has no unique constraint"


def test_monetary_columns_are_numeric_not_float() -> None:
    """Prices and quantities must not round-trip through binary floating point.

    ``Float`` is permitted only for detector scores, which are statistical
    outputs where the last bit of precision is meaningless.
    """
    allowed_float_columns = {"anomaly.score", "anomaly.confidence"}
    offenders = [
        f"{table.name}.{column.name}"
        for table in Base.metadata.sorted_tables
        for column in table.columns
        if isinstance(column.type, Float)
        and not isinstance(column.type, Numeric)
        and f"{table.name}.{column.name}" not in allowed_float_columns
    ]
    assert not offenders, f"floating-point columns holding exact values: {offenders}"


def test_enum_columns_persist_values_not_member_names() -> None:
    """Guards the ``values_callable`` in :func:`app.models.enums.pg_enum`.

    Without it SQLAlchemy stores ``'Z_SCORE'`` while every comparison in the
    codebase uses ``'z_score'``, and CHECK constraints written against the
    values silently never match.
    """
    enum_columns = [
        column
        for table in Base.metadata.sorted_tables
        for column in table.columns
        if isinstance(column.type, SqlEnum)
    ]
    assert enum_columns, "expected enum-typed columns in the schema"

    for column in enum_columns:
        stored = set(column.type.enums)
        assert all(value.islower() or "_" in value for value in stored), (
            f"{column.table.name}.{column.name} stores member names: {sorted(stored)}"
        )

    asset_type = next(column for column in enum_columns if column.type.name == "asset_type")
    assert set(asset_type.type.enums) == {member.value for member in AssetType}


def test_foreign_keys_to_owned_data_cascade() -> None:
    """A deleted parent must not strand its children.

    Enforced in the database rather than only in the ORM, so it holds for
    deletes issued by migrations and by psql too.
    """
    non_cascading = [
        f"{table.name}.{fk.parent.name} -> {fk.column.table.name}"
        for table in Base.metadata.sorted_tables
        for fk in table.foreign_keys
        if fk.ondelete != "CASCADE"
    ]
    assert not non_cascading, f"foreign keys without ON DELETE CASCADE: {non_cascading}"


def test_every_table_carries_timestamps() -> None:
    """Rows must record when they were written; ingestion debugging depends on it."""
    missing = [
        table.name
        for table in Base.metadata.sorted_tables
        if not {"created_at", "updated_at"} <= set(table.columns.keys())
    ]
    assert not missing, f"tables without audit timestamps: {missing}"


def test_ecosystem_tags_cover_the_seeded_universe() -> None:
    """Seed data may only use tags the enum declares."""
    declared = {tag.value for tag in EcosystemTag}
    used = {tag.value for company in COMPANIES for tag in company.tags}
    assert used <= declared


def test_severity_ladder_is_ordered() -> None:
    """The anomaly filter's severity ladder must stay in ascending order."""
    assert AnomalyRepository._at_least(Severity.HIGH) == (Severity.HIGH, Severity.EXTREME)
    assert len(AnomalyRepository._at_least(Severity.LOW)) == len(list(Severity))
    assert AnomalyRepository._at_least(Severity.EXTREME) == (Severity.EXTREME,)


def test_detection_methods_are_distinguishable() -> None:
    """Scores are not comparable across detectors, so method is part of the key."""
    anomaly = Base.metadata.tables["anomaly"]
    unique = next(
        constraint
        for constraint in anomaly.constraints
        if constraint.__class__.__name__ == "UniqueConstraint"
    )
    columns = {column.name for column in unique.columns}
    assert "method" in columns
    assert {member.value for member in DetectionMethod} == {"z_score", "isolation_forest"}
