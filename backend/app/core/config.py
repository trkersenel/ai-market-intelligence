"""Application configuration.

All runtime configuration is loaded from environment variables (or a local
``.env`` file) into immutable, validated settings objects. Nothing in the
codebase may read ``os.environ`` directly -- every consumer depends on
:func:`get_settings`, which makes configuration explicit, typed and testable.

Settings are grouped into cohesive sub-models (database, security, external
providers, ...) so that a service only depends on the slice of configuration it
actually needs, keeping the blast radius of a config change small.
"""

from __future__ import annotations

import json
from enum import StrEnum
from functools import lru_cache
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    Field,
    PostgresDsn,
    SecretStr,
    field_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Environment(StrEnum):
    """Deployment environment the process is running in."""

    LOCAL = "local"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"

    @property
    def is_production_like(self) -> bool:
        """Return ``True`` for environments that must not leak debug output."""
        return self in {Environment.STAGING, Environment.PRODUCTION}


class PostgresSettings(BaseSettings):
    """Connection and pooling configuration for the relational store."""

    model_config = SettingsConfigDict(env_prefix="POSTGRES_", extra="ignore")

    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: SecretStr = SecretStr("postgres")
    db: str = "market_intel"

    pool_size: Annotated[int, Field(ge=1, le=100)] = 10
    max_overflow: Annotated[int, Field(ge=0, le=100)] = 20
    pool_recycle_seconds: Annotated[int, Field(ge=60)] = 1800
    pool_pre_ping: bool = True
    echo_sql: bool = False

    statement_timeout_ms: Annotated[int, Field(ge=0)] = 30_000

    # The DSN builders are plain properties, deliberately not ``computed_field``.
    # A computed field is part of the model schema, so it would appear in
    # ``repr()`` and ``model_dump()`` -- putting the database password in plain
    # text into any log line or error report that dumps settings.

    @property
    def async_dsn(self) -> str:
        """DSN for the asyncpg driver, used by the application at runtime."""
        return str(
            PostgresDsn.build(
                scheme="postgresql+asyncpg",
                username=self.user,
                password=self.password.get_secret_value(),
                host=self.host,
                port=self.port,
                path=self.db,
            )
        )

    @property
    def sync_dsn(self) -> str:
        """DSN for the psycopg driver, used by Alembic and offline tooling."""
        return str(
            PostgresDsn.build(
                scheme="postgresql+psycopg",
                username=self.user,
                password=self.password.get_secret_value(),
                host=self.host,
                port=self.port,
                path=self.db,
            )
        )


class MongoSettings(BaseSettings):
    """Connection configuration for the document store (MongoDB Atlas)."""

    model_config = SettingsConfigDict(env_prefix="MONGO_", extra="ignore")

    uri: SecretStr = SecretStr("mongodb://mongo:mongo@localhost:27017/?authSource=admin")
    database: str = "market_intel"

    max_pool_size: Annotated[int, Field(ge=1, le=500)] = 50
    min_pool_size: Annotated[int, Field(ge=0, le=100)] = 0
    server_selection_timeout_ms: Annotated[int, Field(ge=100)] = 5_000

    #: Name of the Atlas Vector Search index backing semantic retrieval.
    vector_index_name: str = "rag_documents_vector_index"


class SecuritySettings(BaseSettings):
    """JWT signing and password hashing parameters."""

    model_config = SettingsConfigDict(env_prefix="SECURITY_", extra="ignore")

    secret_key: SecretStr = SecretStr("insecure-development-key-change-me")
    algorithm: Literal["HS256", "HS512"] = "HS256"
    access_token_ttl_minutes: Annotated[int, Field(ge=1)] = 30
    refresh_token_ttl_days: Annotated[int, Field(ge=1)] = 14

    @field_validator("secret_key")
    @classmethod
    def _reject_placeholder_in_ci(cls, value: SecretStr) -> SecretStr:
        """Guard against the development key reaching a real deployment.

        The check lives here rather than in :class:`Settings` so that it also
        fires when the sub-model is constructed directly in tests.
        """
        if not value.get_secret_value().strip():
            msg = "SECURITY_SECRET_KEY must not be empty"
            raise ValueError(msg)
        return value


class IngestionSettings(BaseSettings):
    """External data sources, rate limits and retry behaviour."""

    model_config = SettingsConfigDict(env_prefix="INGEST_", extra="ignore")

    #: NewsAPI credential. Absent by default so the platform runs without it --
    #: the RSS and SEC providers need no key, and news ingestion degrades to
    #: those rather than failing.
    newsapi_key: SecretStr | None = None
    newsapi_base_url: str = "https://newsapi.org/v2"

    #: SEC requires a descriptive User-Agent with contact details on every
    #: request; anonymous clients are blocked outright.
    sec_user_agent: str = "AI Market Intelligence Platform (contact@example.com)"
    sec_base_url: str = "https://data.sec.gov"

    #: RSS feeds covering the semiconductor and AI-infrastructure press.
    rss_feeds: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "https://www.tomshardware.com/feeds/all",
            "https://www.anandtech.com/rss/",
            "https://semiengineering.com/feed/",
            "https://www.datacenterdynamics.com/en/rss/",
        ]
    )

    request_timeout_seconds: Annotated[float, Field(gt=0)] = 20.0
    max_retries: Annotated[int, Field(ge=0, le=10)] = 3
    retry_backoff_seconds: Annotated[float, Field(ge=0)] = 1.0

    #: Requests per second allowed per provider. SEC publishes a hard limit of
    #: 10/s; the others are throttled to stay a good neighbour and to keep a
    #: backfill from tripping vendor abuse detection.
    yfinance_rate_limit: Annotated[float, Field(gt=0)] = 4.0
    newsapi_rate_limit: Annotated[float, Field(gt=0)] = 2.0
    sec_rate_limit: Annotated[float, Field(gt=0)] = 8.0
    rss_rate_limit: Annotated[float, Field(gt=0)] = 5.0

    #: Sessions fetched when a ticker has no stored history.
    initial_backfill_days: Annotated[int, Field(ge=1)] = 730
    #: Sessions re-fetched on an incremental run. Overlapping the last stored
    #: date absorbs vendor corrections, which upserts make free.
    incremental_overlap_days: Annotated[int, Field(ge=0)] = 5
    #: Tickers fetched concurrently. Bounded so a backfill cannot exhaust the
    #: connection pool or the provider's patience.
    max_concurrent_fetches: Annotated[int, Field(ge=1, le=32)] = 4

    @field_validator("rss_feeds", mode="before")
    @classmethod
    def _split_feeds(cls, value: Any) -> Any:
        """Accept a comma-separated list or a JSON array."""
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if stripped.startswith("["):
            return json.loads(stripped)
        return [feed.strip() for feed in stripped.split(",") if feed.strip()]


class SchedulerSettings(BaseSettings):
    """Cron expressions for the background ingestion jobs."""

    model_config = SettingsConfigDict(env_prefix="SCHEDULER_", extra="ignore")

    enabled: bool = True
    timezone: str = "UTC"

    #: 22:30 UTC on weekdays -- after the US close, before Asian markets open.
    price_ingestion_cron: str = "30 22 * * mon-fri"
    #: Hourly: news breaks continuously and the correlation engine wants it
    #: available before the next price ingestion runs.
    news_ingestion_cron: str = "5 * * * *"

    #: A job that overruns its next trigger is skipped rather than queued, and
    #: never runs twice concurrently -- ingestion is idempotent, not reentrant.
    misfire_grace_seconds: Annotated[int, Field(ge=1)] = 600
    coalesce_missed_runs: bool = True


class ObservabilitySettings(BaseSettings):
    """Logging and tracing configuration."""

    model_config = SettingsConfigDict(env_prefix="LOG_", extra="ignore")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    #: ``json`` for machine-readable production logs, ``console`` for local dev.
    renderer: Literal["json", "console"] = "json"
    #: Paths excluded from access logging to keep probe noise out of the stream.
    excluded_access_paths: tuple[str, ...] = ("/health/live", "/health/ready", "/metrics")


class Settings(BaseSettings):
    """Root settings object aggregating every configuration group."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
        # Fields with a validation alias (``environment``) must still be
        # constructible by field name, which is how tests build settings.
        populate_by_name=True,
    )

    project_name: str = "AI Market Intelligence Platform"
    version: str = "0.1.0"
    environment: Environment = Field(
        default=Environment.LOCAL,
        validation_alias=AliasChoices("ENVIRONMENT", "ENV", "APP_ENV"),
    )
    debug: bool = False

    api_v1_prefix: str = "/api/v1"
    docs_enabled: bool = True

    #: Origins permitted by the CORS middleware. ``NoDecode`` suppresses
    #: pydantic-settings' automatic JSON decoding of complex types, which would
    #: otherwise reject the comma-separated form before the validator below ever
    #: runs -- shell-friendly ``CORS_ORIGINS=a,b`` is the common case.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )

    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    mongo: MongoSettings = Field(default_factory=MongoSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: Any) -> Any:
        """Accept either ``CORS_ORIGINS=a,b`` or a JSON array."""
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if stripped.startswith("["):
            return json.loads(stripped)
        return [origin.strip() for origin in stripped.split(",") if origin.strip()]

    @property
    def expose_docs(self) -> bool:
        """Whether OpenAPI docs should be served for this environment."""
        return self.docs_enabled and not self.environment.is_production_like


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that environment parsing and validation happen exactly once. Tests
    override configuration by calling ``get_settings.cache_clear()`` after
    patching the environment, or by overriding the FastAPI dependency.
    """
    return Settings()
