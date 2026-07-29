"""Pydantic schemas for the health and readiness endpoints."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class DependencyStatus(StrEnum):
    """Health verdict for a single downstream dependency."""

    UP = "up"
    DOWN = "down"


class DependencyHealth(BaseModel):
    """Result of probing one dependency."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Dependency identifier, e.g. 'postgres'.")
    status: DependencyStatus = Field(description="Whether the probe succeeded.")
    latency_ms: float = Field(ge=0, description="Round-trip time of the probe.")
    error: str | None = Field(default=None, description="Failure reason when status is 'down'.")


class LivenessResponse(BaseModel):
    """Response of the liveness probe: the process is running."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(default="ok")
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    """Response of the readiness probe: dependencies are reachable."""

    model_config = ConfigDict(frozen=True)

    status: DependencyStatus = Field(description="'up' only when every dependency probe succeeded.")
    checked_at: datetime
    dependencies: list[DependencyHealth]
