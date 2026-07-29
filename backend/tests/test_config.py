"""Tests for configuration parsing, validation and secret handling."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import (
    Environment,
    MongoSettings,
    PostgresSettings,
    SecuritySettings,
    Settings,
)


def test_postgres_dsn_uses_the_async_driver() -> None:
    dsn = PostgresSettings(host="db", port=5433, user="u", db="market").async_dsn

    assert dsn.startswith("postgresql+asyncpg://")
    assert "db:5433" in dsn
    assert dsn.endswith("/market")


def test_alembic_dsn_uses_the_sync_driver() -> None:
    dsn = PostgresSettings().sync_dsn

    assert dsn.startswith("postgresql+psycopg://")


def test_password_is_not_leaked_by_repr() -> None:
    settings = PostgresSettings(password="hunter2")  # type: ignore[arg-type]

    assert "hunter2" not in repr(settings)
    assert settings.password.get_secret_value() == "hunter2"
    # ...but the DSN, which is consumed by the driver, still carries it.
    assert "hunter2" in settings.async_dsn


def test_cors_origins_accept_a_comma_separated_string() -> None:
    settings = Settings(cors_origins="http://a.test, http://b.test")  # type: ignore[arg-type]

    assert settings.cors_origins == ["http://a.test", "http://b.test"]


def test_empty_secret_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SecuritySettings(secret_key="   ")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        (Environment.LOCAL, True),
        (Environment.TEST, True),
        (Environment.STAGING, False),
        (Environment.PRODUCTION, False),
    ],
)
def test_docs_are_hidden_in_production_like_environments(
    environment: Environment, expected: bool
) -> None:
    settings = Settings(environment=environment, docs_enabled=True)

    assert settings.expose_docs is expected


def test_docs_can_be_disabled_explicitly() -> None:
    assert Settings(environment=Environment.LOCAL, docs_enabled=False).expose_docs is False


def test_pool_size_is_bounded() -> None:
    with pytest.raises(ValidationError):
        PostgresSettings(pool_size=0)


def test_mongo_defaults_name_the_vector_index() -> None:
    assert MongoSettings().vector_index_name == "rag_documents_vector_index"
