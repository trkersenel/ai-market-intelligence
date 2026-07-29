"""Tests for the health endpoints and the request-context middleware."""

from __future__ import annotations

import uuid

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_health_service
from app.core.middleware import REQUEST_ID_HEADER, RESPONSE_TIME_HEADER
from tests.conftest import StubHealthService


async def test_liveness_reports_version_and_environment(client: AsyncClient) -> None:
    response = await client.get("/health/live")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["environment"] == "test"
    assert body["version"]


async def test_readiness_returns_200_when_all_dependencies_are_up(
    client: AsyncClient,
) -> None:
    response = await client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "up"
    assert {dep["name"] for dep in body["dependencies"]} == {"postgres", "mongodb"}


async def test_readiness_returns_503_when_a_dependency_is_down(app: FastAPI) -> None:
    app.dependency_overrides[get_health_service] = lambda: StubHealthService(healthy=False)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "down"


async def test_middleware_echoes_supplied_request_id(client: AsyncClient) -> None:
    request_id = str(uuid.uuid4())

    response = await client.get("/health/live", headers={REQUEST_ID_HEADER: request_id})

    assert response.headers[REQUEST_ID_HEADER] == request_id
    assert float(response.headers[RESPONSE_TIME_HEADER]) >= 0


async def test_middleware_generates_request_id_when_absent(client: AsyncClient) -> None:
    response = await client.get("/health/live")

    generated = response.headers[REQUEST_ID_HEADER]
    assert uuid.UUID(generated)  # raises if it is not a well-formed UUID


@pytest.mark.parametrize("path", ["/api/v1/does-not-exist", "/nope"])
async def test_unknown_routes_use_the_error_envelope(client: AsyncClient, path: str) -> None:
    response = await client.get(path)

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "http_error"
    assert error["request_id"]
