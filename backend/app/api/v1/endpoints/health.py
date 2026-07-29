"""Liveness and readiness endpoints.

Two distinct probes, because orchestrators treat them differently: a failing
liveness probe restarts the container, while a failing readiness probe only
removes it from the load balancer. Conflating them turns a transient database
blip into a restart loop.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.api.deps import HealthServiceDep, SettingsDep
from app.schemas.health import DependencyStatus, LivenessResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get(
    "/live",
    response_model=LivenessResponse,
    summary="Liveness probe",
    description="Confirms the process is running. Never touches a dependency.",
)
async def liveness(settings: SettingsDep) -> LivenessResponse:
    """Return static process metadata."""
    return LivenessResponse(
        version=settings.version,
        environment=settings.environment.value,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Probes PostgreSQL and MongoDB; 503 when any dependency is down.",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "At least one dependency is unreachable.",
            "model": ReadinessResponse,
        }
    },
)
async def readiness(
    health_service: HealthServiceDep,
    response: Response,
) -> ReadinessResponse:
    """Probe every backing store and reflect the verdict in the status code."""
    report = await health_service.check_readiness()
    if report.status is DependencyStatus.DOWN:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report
