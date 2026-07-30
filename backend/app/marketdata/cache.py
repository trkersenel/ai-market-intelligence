"""Response cache with request coalescing.

Two mechanisms, and the second matters more than the first.

**TTL caching** is obvious: a quote is stale in seconds, a company's IPO date
never changes, and one lifetime for both would either hammer the provider or
serve yesterday's price. Each data kind carries its own.

**Single-flight coalescing** is what actually makes a 60-call-per-minute budget
survive contact with users. Fifty people opening AAPL in the same second is
fifty cache misses, and a plain TTL cache issues fifty upstream calls -- blowing
the minute's entire quota on one symbol. Coalescing lets the first caller make
the request and the other forty-nine await *that same* in-flight future. The
upstream sees one call.

This is a per-process cache, so each replica keeps its own. That is a deliberate
trade rather than an oversight: a shared Redis would add an operational
dependency and a network hop to every read, and at this scale the duplication
costs a handful of extra upstream calls after a deploy. The interface is narrow
enough that swapping in a distributed backend later touches this file only.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class _Entry:
    """A cached value and the moment it stops being usable."""

    value: Any
    expires_at: float

    @property
    def is_fresh(self) -> bool:
        """Whether the entry may still be served."""
        return time.monotonic() < self.expires_at


class ResponseCache:
    """TTL cache that coalesces concurrent misses for the same key."""

    def __init__(self, *, max_entries: int = 10_000) -> None:
        """Create an empty cache.

        Args:
            max_entries: Hard ceiling on retained entries. The NASDAQ universe
                is ~5,600 symbols and a page touches several kinds of data per
                symbol, so an unbounded cache would grow without limit on a
                long-running process.
        """
        self._entries: dict[str, _Entry] = {}
        self._in_flight: dict[str, asyncio.Future[Any]] = {}
        self._max_entries = max_entries
        self._hits = 0
        self._misses = 0
        self._coalesced = 0

    @property
    def stats(self) -> dict[str, int]:
        """Hit, miss and coalesced counts, for the metrics endpoint."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "coalesced": self._coalesced,
            "entries": len(self._entries),
        }

    async def get_or_fetch[T](
        self,
        key: str,
        ttl_seconds: int,
        fetch: Callable[[], Awaitable[T]],
    ) -> T:
        """Return the cached value, or fetch it exactly once.

        Args:
            key: Cache key. Must include everything that varies the result --
                symbol, interval, date window -- or two different requests will
                collide on one entry.
            ttl_seconds: How long the fetched value stays fresh.
            fetch: Coroutine producing the value on a miss.

        Returns:
            The cached or freshly fetched value.

        Notes:
            The three cases, in order:

            1. **Fresh hit** -- returned immediately, no upstream call.
            2. **Miss with a fetch already running** -- awaits the existing
               future rather than starting a second one. This is the case that
               protects the quota, and it is invisible in hit-rate metrics,
               which is why it is counted separately.
            3. **Miss with nothing running** -- becomes the single flight.

            A failed fetch is not cached. Caching an error would turn one
            provider blip into a TTL-long outage for that symbol, and the
            provider is usually recovered well before the entry expires.
        """
        entry = self._entries.get(key)
        if entry is not None and entry.is_fresh:
            self._hits += 1
            return entry.value  # type: ignore[no-any-return]

        running = self._in_flight.get(key)
        if running is not None:
            self._coalesced += 1
            return await running  # type: ignore[no-any-return]

        self._misses += 1
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._in_flight[key] = future

        try:
            value = await fetch()
        except BaseException as exc:
            # Every waiter must learn the outcome, or they hang until timeout.
            # The exception is *not* stored, so the next caller retries.
            if not future.done():
                future.set_exception(exc)
            raise
        else:
            self._store(key, value, ttl_seconds)
            if not future.done():
                future.set_result(value)
            return value
        finally:
            self._in_flight.pop(key, None)
            # Waiters that never observe the result would otherwise log
            # "exception was never retrieved" on garbage collection.
            if future.done() and not future.cancelled():
                future.exception()

    def _store(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Insert an entry, evicting if the cache is full."""
        if len(self._entries) >= self._max_entries:
            self._evict()
        self._entries[key] = _Entry(value=value, expires_at=time.monotonic() + ttl_seconds)

    def _evict(self) -> None:
        """Drop expired entries, then the oldest if that was not enough.

        Expiry-first rather than straight LRU: an expired entry is worthless to
        everyone, while a cold-but-valid one may still save an upstream call.
        Python dicts preserve insertion order, so the oldest is simply the
        first key -- no separate ordering structure needed.
        """
        expired = [key for key, entry in self._entries.items() if not entry.is_fresh]
        for key in expired:
            del self._entries[key]

        overflow = len(self._entries) - self._max_entries + 1
        if overflow > 0:
            for key in list(self._entries)[:overflow]:
                del self._entries[key]
            logger.debug("cache_evicted", expired=len(expired), oldest=overflow)

    def invalidate(self, key: str) -> bool:
        """Drop one entry. Returns whether it was present."""
        return self._entries.pop(key, None) is not None

    def invalidate_prefix(self, prefix: str) -> int:
        """Drop every entry whose key starts with ``prefix``.

        Used when a sync rewrites a symbol's data and the cached derivatives
        are known to be stale before their TTL would say so.
        """
        matching = [key for key in self._entries if key.startswith(prefix)]
        for key in matching:
            del self._entries[key]
        return len(matching)

    def clear(self) -> None:
        """Drop every entry. In-flight fetches are left to complete."""
        self._entries.clear()


def cache_key(*parts: object) -> str:
    """Build a cache key from its parts.

    Every varying input must be a part. A key of just the symbol would serve a
    1-minute chart to a caller asking for daily bars -- the kind of bug that
    looks like a provider fault and is not.
    """
    return ":".join("" if part is None else str(part) for part in parts)
