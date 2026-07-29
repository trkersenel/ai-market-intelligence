"""Async token-bucket rate limiter.

Every external provider gets its own limiter. Without one, a backfill across
fourteen tickers issues requests as fast as the event loop can dispatch them --
which for SEC means an outright block, and for the others means tripping abuse
detection at the worst possible moment.

A token bucket rather than a fixed delay: it permits a short burst up to the
bucket's capacity and only then throttles to the sustained rate, which is the
right shape for ingestion that runs hard for a minute and then goes quiet.
"""

from __future__ import annotations

import asyncio
import time
from types import TracebackType

from app.core.logging import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Limits acquisitions to a sustained rate, allowing a bounded burst."""

    def __init__(self, *, rate_per_second: float, burst: int | None = None) -> None:
        """Configure the bucket.

        Args:
            rate_per_second: Sustained refill rate, in tokens per second.
            burst: Bucket capacity. Defaults to one second's worth of tokens
                (minimum 1), which allows a small burst without letting an idle
                period accumulate an unbounded allowance.

        Raises:
            ValueError: If ``rate_per_second`` is not positive.
        """
        if rate_per_second <= 0:
            msg = "rate_per_second must be positive"
            raise ValueError(msg)

        self._rate = rate_per_second
        self._capacity = float(burst if burst is not None else max(1, int(rate_per_second)))
        self._tokens = self._capacity
        self._updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def rate_per_second(self) -> float:
        """The sustained rate this limiter enforces."""
        return self._rate

    async def acquire(self) -> float:
        """Wait until a token is available, then consume it.

        Returns:
            Seconds spent waiting -- zero when a token was immediately free.
            Returned rather than logged so callers can surface throttling in
            their own ingestion metrics.

        Notes:
            The sleep happens while the lock is held. That serialises waiters,
            which is what makes the limiter fair: requests are admitted in
            arrival order rather than in whatever order the loop happens to
            wake them.
        """
        async with self._lock:
            waited = 0.0
            while True:
                self._refill()
                if self._tokens >= 1:
                    self._tokens -= 1
                    return waited

                deficit = 1 - self._tokens
                delay = deficit / self._rate
                waited += delay
                await asyncio.sleep(delay)

    def _refill(self) -> None:
        """Add the tokens accrued since the last check, capped at capacity."""
        now = time.monotonic()
        elapsed = now - self._updated_at
        self._updated_at = now
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)

    async def __aenter__(self) -> RateLimiter:
        """Acquire a token on entry."""
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release nothing -- tokens refill on a timer, not on exit."""
        return None
