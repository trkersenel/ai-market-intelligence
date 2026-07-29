"""Tests for the shared HTTP client's retry and error-translation policy.

Uses ``httpx.MockTransport`` -- a real ``AsyncClient`` with a scripted transport
-- so the code under test is exercised end to end, including header handling and
response parsing, without a socket.
"""

from __future__ import annotations

import httpx
import pytest

from app.clients.http import HttpClient
from app.core.config import IngestionSettings
from app.core.exceptions import ExternalServiceError, RateLimitError


def _settings(**overrides: object) -> IngestionSettings:
    """Build ingestion settings with retries fast enough for a test suite."""
    return IngestionSettings(
        max_retries=2,
        retry_backoff_seconds=0.001,
        request_timeout_seconds=1.0,
        **overrides,  # type: ignore[arg-type]
    )


def _client(handler: httpx.MockTransport, **kwargs: object) -> HttpClient:
    """Build an HttpClient over a scripted transport."""
    return HttpClient(
        settings=_settings(),
        rate_limit=1000,
        provider="test",
        client=httpx.AsyncClient(transport=handler, base_url="https://api.test"),
        **kwargs,  # type: ignore[arg-type]
    )


async def test_successful_json_response_is_parsed() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"ok": True}))

    async with _client(transport) as client:
        assert await client.get_json("/thing") == {"ok": True}


async def test_transient_server_error_is_retried_then_succeeds() -> None:
    """A 503 followed by a 200 must return the 200, not raise."""
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"attempt": calls})

    async with _client(httpx.MockTransport(handler)) as client:
        assert await client.get_json("/flaky") == {"attempt": 2}

    assert calls == 2


async def test_permanent_client_error_is_not_retried() -> None:
    """Retrying a 404 only delays the failure; it must fail on the first call."""
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404)

    async with _client(httpx.MockTransport(handler)) as client:
        with pytest.raises(ExternalServiceError) as error:
            await client.get_json("/missing")

    assert calls == 1
    assert error.value.details["status_code"] == 404


async def test_exhausted_retries_raise_external_service_error() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(502)

    async with _client(httpx.MockTransport(handler)) as client:
        with pytest.raises(ExternalServiceError, match="unavailable"):
            await client.get_json("/down")

    assert calls == 3  # the initial attempt plus max_retries


async def test_persistent_throttling_raises_rate_limit_error() -> None:
    """429 is distinguished from a generic outage so callers can back off."""
    transport = httpx.MockTransport(lambda _: httpx.Response(429))

    async with _client(transport) as client:
        with pytest.raises(RateLimitError):
            await client.get_json("/throttled")


async def test_network_failure_is_retried_and_translated() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        msg = "connection reset"
        raise httpx.ConnectError(msg, request=request)

    async with _client(httpx.MockTransport(handler)) as client:
        with pytest.raises(ExternalServiceError):
            await client.get_json("/unreachable")

    assert calls == 3


async def test_non_json_body_is_reported_clearly() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, text="<html>nope</html>"))

    async with _client(transport) as client:
        with pytest.raises(ExternalServiceError, match="non-JSON"):
            await client.get_json("/html")


async def test_text_responses_are_returned_verbatim() -> None:
    feed = "<rss><channel><title>Feed</title></channel></rss>"
    transport = httpx.MockTransport(lambda _: httpx.Response(200, text=feed))

    async with _client(transport) as client:
        assert await client.get_text("/feed.xml") == feed
