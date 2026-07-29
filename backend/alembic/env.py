"""Alembic migration environment.

Two decisions worth calling out:

1. The database URL comes from :func:`app.core.config.get_settings`, never from
   ``alembic.ini``. There is exactly one source of truth for connection details.
2. Migrations run against the *synchronous* psycopg driver. Alembic's autogenerate
   inspection is inherently blocking, and a sync connection avoids an event-loop
   dance for no benefit -- the async driver is an application-runtime concern.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.core.config import get_settings

# Importing the package -- not just ``Base`` -- is what registers every ORM model
# on ``Base.metadata`` and makes autogenerate see the full schema.
from app.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.postgres.sync_dsn)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting, for review or manual application."""
    context.configure(
        url=settings.postgres.sync_dsn,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live connection inside one transaction.

    ``engine.begin()`` rather than ``engine.connect()``: the transaction is
    owned here and committed on a clean exit, rolled back on any exception. The
    whole migration is therefore atomic -- PostgreSQL supports transactional
    DDL, so a failure half way through leaves no partially-created schema.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.begin() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            include_schemas=False,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
