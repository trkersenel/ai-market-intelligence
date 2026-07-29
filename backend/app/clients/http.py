"""Shared HTTP client with rate limiting, retries and error translation.

Every outbound call in the platform goes through this class, so retry policy,
throttling and failure semantics are decided once rather than re-implemented
per provider.

Retry policy: only *transient* failures are retried -- timeouts, connection
errors, 429 and 5xx. A 400 or a 404 is a bug or a genuinely missing resource,
and retrying it three times just delays the error by six seconds.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any, Self

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.clients.rate_limiter import RateLimiter
from app.core.config import IngestionSettings
from app.core.exceptions import ExternalServiceError, RateLimitError
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Status codes worth trying again. 429 is included because providers use it for
#: short-lived throttling; the limiter should prevent it, but a shared IP or a
#: vendor-side change can still produce one.
RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


class TransientHttpError(Exception):
    """Internal marker for a failure worth retrying."""


class HttpClient:
    """An ``httpx.AsyncClient`` wrapped in the platform's resilience policy."""

    def __init__(
        self,
        *,
        settings: IngestionSettings,
        base_url: str = "",
        rate_limit: float,
        headers: dict[str, str] | None = None,
        provider: str = "http",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Build a client for one provider.

        Args:
            settings: Timeout and retry configuration.
            base_url: Prefix applied to relative request paths.
            rate_limit: Sustained requests per second for this provider.
            headers: Default headers, e.g. an API key or a required User-Agent.
            provider: Name used in logs.
            client: Pre-built transport, injected by tests to serve recorded
                responses without a network.
        """
        self._settings = settings
        self._provider = provider
        self._limiter = RateLimiter(rate_per_second=rate_limit)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(settings.request_timeout_seconds),
            follow_redirects=True,
        )
        # Applied after construction, not only in the constructor above, so an
        # injected transport carries the same headers as a real one. Without
        # this, credentials and the SEC-mandated User-Agent would be present in
        # production and absent under test -- the one difference guaranteed to
        # make a passing suite meaningless.
        if headers:
            self._client.headers.update(headers)

    async def get_json(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        """Issue a GET and parse the response as JSON.

        Raises:
            ExternalServiceError: On a non-retryable status, an unparseable
                body, or exhausted retries.
            RateLimitError: When the provider reports throttling after retries.
        """
        response = await self._request("GET", url, params=params)
        try:
            return response.json()
        except ValueError as exc:
            msg = f"{self._provider} returned a non-JSON response"
            raise ExternalServiceError(msg, details={"url": url}) from exc

    async def post_json(self, url: str, *, json: dict[str, Any]) -> Any:
        """POST a JSON body and parse the JSON response.

        Raises:
            ExternalServiceError: On a non-retryable status or unparseable body.
        """
        response = await self._request("POST", url, json=json)
        try:
            return response.json()
        except ValueError as exc:
            msg = f"{self._provider} returned a non-JSON response"
            raise ExternalServiceError(msg, details={"url": url}) from exc

    async def get_text(self, url: str, *, params: dict[str, Any] | None = None) -> str:
        """Issue a GET and return the body as text (for RSS and XML feeds)."""
        response = await self._request("GET", url, params=params)
        return response.text

    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Perform a rate-limited request, retrying transient failures."""
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._settings.max_retries + 1),
                wait=wait_exponential_jitter(initial=self._settings.retry_backoff_seconds),
                retry=retry_if_exception_type((TransientHttpError, httpx.TransportError)),
                reraise=False,
            ):
                with attempt:
                    return await self._attempt(method, url, params=params, json=json)
        except RetryError as exc:
            last = exc.last_attempt.exception()
            if isinstance(last, TransientHttpError) and "429" in str(last):
                msg = f"{self._provider} is rate limiting requests"
                raise RateLimitError(msg, details={"url": url}) from exc
            msg = f"{self._provider} is unavailable after {self._settings.max_retries} retries"
            raise ExternalServiceError(msg, details={"url": url}) from exc

        # Unreachable: AsyncRetrying either returns a response or raises.
        raise AssertionError  # pragma: no cover

    async def _attempt(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Issue one rate-limited request and classify the outcome."""
        waited = await self._limiter.acquire()
        if waited > 0:
            logger.debug("rate_limited", provider=self._provider, waited_seconds=waited)

        response = await self._client.request(method, url, params=params, json=json)

        if response.status_code in RETRYABLE_STATUS_CODES:
            msg = f"{self._provider} returned {response.status_code}"
            raise TransientHttpError(msg)

        if response.is_error:
            # Permanent: retrying a 400 or 404 only delays the failure.
            msg = f"{self._provider} returned {response.status_code}"
            raise ExternalServiceError(
                msg,
                details={"url": url, "status_code": response.status_code},
            )
        return response

    async def aclose(self) -> None:
        """Close the transport, unless it was supplied by the caller."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        """Enter the client's lifetime."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the transport on exit."""
        await self.aclose()
