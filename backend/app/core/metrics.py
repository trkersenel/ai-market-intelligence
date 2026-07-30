"""Prometheus metrics.

Hand-rolled rather than pulling in ``prometheus-client``. The dependency is small
but it brings a global default registry and a multiprocess mode that has to be
configured correctly to be correct at all, and this platform needs four metric
families. Writing the exposition format directly is about ninety lines, has no
global state, and is trivially testable.

What is measured, and why each earns its place:

- **Request rate, status and latency**, labelled by route *template* rather than
  by path. ``/api/v1/prices/{symbol}`` is one series; using the raw path would
  create one per ticker and, on a public API, unbounded cardinality — the classic
  way to take down a Prometheus server with your own instrumentation.
- **Job outcomes and duration**, because the scheduler is where this platform
  actually fails. A silent job that stopped running is invisible in request
  metrics.
- **Data freshness**, which is the metric an operator actually pages on. Requests
  can be healthy while ingestion has been broken for a day, and the difference
  between those two states is the whole point of monitoring this system.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Literal

#: Latency histogram buckets, in seconds. Chosen around what this API actually
#: does: sub-10ms for a cached lookup, ~100ms for a price series, seconds for an
#: ingestion trigger. Default buckets would put half the traffic in one bin.
LATENCY_BUCKETS: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

JobOutcome = Literal["succeeded", "failed"]


@dataclass
class _Histogram:
    """Cumulative bucket counts, a sum and a count."""

    buckets: dict[float, int] = field(default_factory=lambda: dict.fromkeys(LATENCY_BUCKETS, 0))
    total: float = 0.0
    count: int = 0

    def observe(self, value: float) -> None:
        """Record one observation."""
        self.total += value
        self.count += 1
        for bound in LATENCY_BUCKETS:
            if value <= bound:
                self.buckets[bound] += 1


class MetricsRegistry:
    """Collects metrics and renders them in the Prometheus text format.

    An explicit instance held on ``app.state`` rather than a module global, for
    the same reason the database adapters are: two apps in one interpreter (a
    test and the real one) must not share counters.
    """

    def __init__(self) -> None:
        """Create an empty registry."""
        # A lock because Starlette runs sync middleware in a threadpool, so two
        # threads can increment the same counter. The critical sections are a
        # few dict operations, so contention is not a concern.
        self._lock = Lock()
        self._requests: dict[tuple[str, str, int], int] = defaultdict(int)
        self._latency: dict[tuple[str, str], _Histogram] = defaultdict(_Histogram)
        self._jobs: dict[tuple[str, str], int] = defaultdict(int)
        self._job_duration: dict[str, _Histogram] = defaultdict(_Histogram)
        self._freshness: dict[str, float] = {}

    def observe_request(
        self, *, method: str, route: str, status: int, duration_seconds: float
    ) -> None:
        """Record one completed HTTP request."""
        with self._lock:
            self._requests[(method, route, status)] += 1
            self._latency[(method, route)].observe(duration_seconds)

    def observe_job(self, *, job: str, outcome: JobOutcome, duration_seconds: float) -> None:
        """Record one completed scheduled job."""
        with self._lock:
            self._jobs[(job, outcome)] += 1
            self._job_duration[job].observe(duration_seconds)

    def set_freshness(self, *, dataset: str, age_seconds: float) -> None:
        """Record how old the newest row in a dataset is.

        A gauge rather than a counter: the question is "how stale is this now",
        and the answer can go down as well as up.
        """
        with self._lock:
            self._freshness[dataset] = age_seconds

    def render(self) -> str:
        """Return the metrics in Prometheus exposition format."""
        with self._lock:
            lines: list[str] = []

            lines += [
                "# HELP http_requests_total Total HTTP requests by method, route and status.",
                "# TYPE http_requests_total counter",
            ]
            for (method, route, status), count in sorted(self._requests.items()):
                labels = f'method="{method}",route="{_escape(route)}",status="{status}"'
                lines.append(f"http_requests_total{{{labels}}} {count}")

            lines += [
                "# HELP http_request_duration_seconds HTTP request latency.",
                "# TYPE http_request_duration_seconds histogram",
            ]
            for (method, route), histogram in sorted(self._latency.items()):
                base = f'method="{method}",route="{_escape(route)}"'
                cumulative = 0
                for bound in LATENCY_BUCKETS:
                    cumulative = histogram.buckets[bound]
                    lines.append(
                        f'http_request_duration_seconds_bucket{{{base},le="{bound}"}} {cumulative}'
                    )
                # The +Inf bucket must equal the count, or the histogram is
                # malformed and Prometheus will reject the quantile.
                lines.append(
                    f'http_request_duration_seconds_bucket{{{base},le="+Inf"}} {histogram.count}'
                )
                lines.append(f"http_request_duration_seconds_sum{{{base}}} {histogram.total}")
                lines.append(f"http_request_duration_seconds_count{{{base}}} {histogram.count}")

            lines += [
                "# HELP scheduler_jobs_total Scheduled job runs by outcome.",
                "# TYPE scheduler_jobs_total counter",
            ]
            for (job, outcome), count in sorted(self._jobs.items()):
                lines.append(f'scheduler_jobs_total{{job="{job}",outcome="{outcome}"}} {count}')

            lines += [
                "# HELP scheduler_job_duration_seconds Scheduled job duration.",
                "# TYPE scheduler_job_duration_seconds histogram",
            ]
            for job, histogram in sorted(self._job_duration.items()):
                for bound in LATENCY_BUCKETS:
                    lines.append(
                        f'scheduler_job_duration_seconds_bucket{{job="{job}",le="{bound}"}} '
                        f"{histogram.buckets[bound]}"
                    )
                lines.append(
                    f'scheduler_job_duration_seconds_bucket{{job="{job}",le="+Inf"}} '
                    f"{histogram.count}"
                )
                lines.append(f'scheduler_job_duration_seconds_sum{{job="{job}"}} {histogram.total}')
                lines.append(
                    f'scheduler_job_duration_seconds_count{{job="{job}"}} {histogram.count}'
                )

            lines += [
                "# HELP data_age_seconds Age of the newest record in each dataset.",
                "# TYPE data_age_seconds gauge",
            ]
            for dataset, age in sorted(self._freshness.items()):
                lines.append(f'data_age_seconds{{dataset="{dataset}"}} {age:.0f}')

            return "\n".join(lines) + "\n"


def _escape(value: str) -> str:
    """Escape a label value per the exposition format."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class Timer:
    """Measures a block's duration on the monotonic clock."""

    def __init__(self) -> None:
        """Start the timer."""
        self._started = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Seconds since construction."""
        return time.perf_counter() - self._started
