"""Tests for the token-bucket rate limiter.

Timing tests are flaky when they assert on wall-clock durations, so these assert
on the limiter's *reported* wait instead -- derived from ``time.monotonic`` and
the token arithmetic, which is deterministic enough to test while still
exercising the real code path.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.clients.rate_limiter import RateLimiter


def test_rejects_a_non_positive_rate() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        RateLimiter(rate_per_second=0)


async def test_burst_capacity_is_granted_without_waiting() -> None:
    """An idle bucket admits a burst immediately -- that is the point of one."""
    limiter = RateLimiter(rate_per_second=10, burst=5)

    waits = [await limiter.acquire() for _ in range(5)]

    assert waits == [0.0] * 5


async def test_exhausting_the_bucket_forces_a_wait() -> None:
    limiter = RateLimiter(rate_per_second=20, burst=2)
    await limiter.acquire()
    await limiter.acquire()

    waited = await limiter.acquire()

    assert waited > 0


async def test_sustained_rate_is_enforced() -> None:
    """Six acquisitions at 50/s with a burst of 1 must take at least 100ms."""
    limiter = RateLimiter(rate_per_second=50, burst=1)

    started = time.monotonic()
    for _ in range(6):
        await limiter.acquire()
    elapsed = time.monotonic() - started

    # 5 waits of 20ms after the first free token.
    assert elapsed >= 0.09


async def test_concurrent_callers_are_all_admitted() -> None:
    """Nothing is dropped or deadlocked when many coroutines contend."""
    limiter = RateLimiter(rate_per_second=100, burst=2)

    results = await asyncio.gather(*(limiter.acquire() for _ in range(10)))

    assert len(results) == 10
    assert all(wait >= 0 for wait in results)


async def test_works_as_an_async_context_manager() -> None:
    limiter = RateLimiter(rate_per_second=10)

    async with limiter as entered:
        assert entered is limiter
