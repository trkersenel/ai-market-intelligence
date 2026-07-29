"""Tests that settings parse correctly from real environment variables.

These guard the seam that unit-constructing ``Settings(...)`` does not exercise:
pydantic-settings decodes complex types from the environment before validators
run, which is exactly where a container-only startup failure hides.
"""

from __future__ import annotations

import pytest

from app.core.config import Environment, Settings


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove any ambient configuration so each case starts from defaults."""
    for key in ("CORS_ORIGINS", "ENVIRONMENT", "ENV", "APP_ENV", "LOG_LEVEL", "POSTGRES_HOST"):
        monkeypatch.delenv(key, raising=False)


def _settings() -> Settings:
    """Build settings from the environment only, ignoring any local .env file."""
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_comma_separated_cors_origins_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test,http://b.test")

    assert _settings().cors_origins == ["http://a.test", "http://b.test"]


def test_json_array_cors_origins_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", '["http://a.test", "http://b.test"]')

    assert _settings().cors_origins == ["http://a.test", "http://b.test"]


def test_environment_alias_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")

    settings = _settings()

    assert settings.environment is Environment.PRODUCTION
    assert settings.expose_docs is False


def test_nested_settings_read_their_prefixed_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = _settings()

    assert settings.postgres.host == "db.internal"
    assert settings.observability.level == "DEBUG"
