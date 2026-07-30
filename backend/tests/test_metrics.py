"""Tests for the metrics registry and its exposition format.

The format is not free-form: Prometheus rejects a malformed histogram outright,
and a mislabelled series is worse than none because it silently pollutes a
dashboard. So these check the output as a contract, not just that numbers move.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.core.metrics import LATENCY_BUCKETS, MetricsRegistry, Timer


@pytest.fixture
def registry() -> MetricsRegistry:
    return MetricsRegistry()


class TestRequestMetrics:
    """Request counters and the latency histogram."""

    def test_requests_are_counted_by_method_route_and_status(
        self, registry: MetricsRegistry
    ) -> None:
        registry.observe_request(
            method="GET", route="/api/v1/prices/{symbol}", status=200, duration_seconds=0.02
        )
        registry.observe_request(
            method="GET", route="/api/v1/prices/{symbol}", status=404, duration_seconds=0.01
        )

        output = registry.render()

        assert (
            'http_requests_total{method="GET",route="/api/v1/prices/{symbol}",status="200"} 1'
            in output
        )
        assert (
            'http_requests_total{method="GET",route="/api/v1/prices/{symbol}",status="404"} 1'
            in output
        )

    def test_the_infinity_bucket_equals_the_observation_count(
        self, registry: MetricsRegistry
    ) -> None:
        """A histogram whose +Inf bucket disagrees with its count is malformed.

        Prometheus rejects it rather than degrading, so this is a correctness
        check on the exposition, not a nicety.
        """
        for duration in (0.001, 0.03, 0.4, 30.0):
            registry.observe_request(
                method="GET", route="/x", status=200, duration_seconds=duration
            )

        output = registry.render()

        assert 'http_request_duration_seconds_bucket{method="GET",route="/x",le="+Inf"} 4' in output
        assert 'http_request_duration_seconds_count{method="GET",route="/x"} 4' in output

    def test_buckets_are_cumulative(self, registry: MetricsRegistry) -> None:
        """Prometheus histograms are cumulative: each bucket includes the ones below."""
        registry.observe_request(method="GET", route="/x", status=200, duration_seconds=0.001)

        output = registry.render()

        for bound in LATENCY_BUCKETS:
            assert f'le="{bound}"}} 1' in output

    def test_an_observation_beyond_every_bucket_still_counts(
        self, registry: MetricsRegistry
    ) -> None:
        registry.observe_request(method="GET", route="/x", status=200, duration_seconds=999.0)

        output = registry.render()

        assert 'le="10.0"}} 0'.replace("}}", "}") in output or 'le="10.0"} 0' in output
        assert 'le="+Inf"} 1' in output

    def test_label_values_are_escaped(self, registry: MetricsRegistry) -> None:
        """An unescaped quote in a label produces a line Prometheus cannot parse."""
        registry.observe_request(
            method="GET", route='/weird"path', status=200, duration_seconds=0.01
        )

        assert r"/weird\"path" in registry.render()

    def test_the_sum_accumulates_durations(self, registry: MetricsRegistry) -> None:
        registry.observe_request(method="GET", route="/x", status=200, duration_seconds=0.25)
        registry.observe_request(method="GET", route="/x", status=200, duration_seconds=0.75)

        assert 'http_request_duration_seconds_sum{method="GET",route="/x"} 1.0' in registry.render()


class TestJobMetrics:
    """Scheduler job outcomes and durations."""

    def test_outcomes_are_counted_separately(self, registry: MetricsRegistry) -> None:
        registry.observe_job(job="ingest_prices", outcome="succeeded", duration_seconds=3.0)
        registry.observe_job(job="ingest_prices", outcome="failed", duration_seconds=0.5)

        output = registry.render()

        assert 'scheduler_jobs_total{job="ingest_prices",outcome="succeeded"} 1' in output
        assert 'scheduler_jobs_total{job="ingest_prices",outcome="failed"} 1' in output

    def test_job_duration_is_a_well_formed_histogram(self, registry: MetricsRegistry) -> None:
        registry.observe_job(job="detect_anomalies", outcome="succeeded", duration_seconds=4.2)

        output = registry.render()

        assert 'scheduler_job_duration_seconds_bucket{job="detect_anomalies",le="+Inf"} 1' in output
        assert 'scheduler_job_duration_seconds_count{job="detect_anomalies"} 1' in output


class TestFreshness:
    """The gauge an operator actually pages on."""

    def test_freshness_is_a_gauge_that_can_fall(self, registry: MetricsRegistry) -> None:
        """Staleness goes down when ingestion catches up, so it cannot be a counter."""
        registry.set_freshness(dataset="daily_prices", age_seconds=172_800)
        registry.set_freshness(dataset="daily_prices", age_seconds=0)

        assert 'data_age_seconds{dataset="daily_prices"} 0' in registry.render()

    def test_datasets_are_tracked_independently(self, registry: MetricsRegistry) -> None:
        registry.set_freshness(dataset="daily_prices", age_seconds=100)
        registry.set_freshness(dataset="news", age_seconds=200)

        output = registry.render()

        assert 'data_age_seconds{dataset="daily_prices"} 100' in output
        assert 'data_age_seconds{dataset="news"} 200' in output


class TestExpositionFormat:
    """The text format Prometheus parses."""

    def test_every_family_declares_help_and_type(self, registry: MetricsRegistry) -> None:
        registry.observe_request(method="GET", route="/x", status=200, duration_seconds=0.01)

        output = registry.render()

        for family in (
            "http_requests_total",
            "http_request_duration_seconds",
            "scheduler_jobs_total",
            "scheduler_job_duration_seconds",
            "data_age_seconds",
        ):
            assert f"# HELP {family} " in output
            assert f"# TYPE {family} " in output

    def test_output_ends_with_a_newline(self, registry: MetricsRegistry) -> None:
        """The text format requires a trailing newline; some scrapers reject it without."""
        assert registry.render().endswith("\n")

    def test_an_empty_registry_still_renders_valid_output(self, registry: MetricsRegistry) -> None:
        output = registry.render()

        assert "# TYPE http_requests_total counter" in output
        assert output.endswith("\n")


class TestTimer:
    """The monotonic stopwatch jobs are timed with."""

    def test_elapsed_is_monotonic_and_non_negative(self) -> None:
        timer = Timer()

        assert timer.elapsed >= 0
        assert timer.elapsed <= timer.elapsed


class TestMiddlewareIntegration:
    """Label cardinality, end to end through the app."""

    async def test_requests_are_recorded_under_the_route_template(self, app: FastAPI) -> None:
        """The label must be the pattern, not the concrete path.

        One series per ticker would be unbounded cardinality on any endpoint
        taking a free-form identifier -- the standard way an instrumented
        service takes down its own metrics backend.
        """
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            await client.get("/health/live")

        output: str = app.state.metrics.render()

        assert 'route="/health/live"' in output

    async def test_unmatched_paths_collapse_to_one_series(self, app: FastAPI) -> None:
        """Otherwise a 404 scanner can create series at will."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            for path in ("/nope-1", "/nope-2", "/nope-3"):
                await client.get(path)

        output: str = app.state.metrics.render()

        assert 'route="unmatched",status="404"} 3' in output
        assert "nope-1" not in output

    async def test_a_collection_endpoint_declared_with_an_empty_path_is_named(
        self, market_client: AsyncClient, market_app: FastAPI
    ) -> None:
        """`@router.get("")` under a prefix has an empty own-path.

        Testing that path for truthiness rather than for None skipped every
        collection endpoint and filed it under "unmatched" -- with a 200 status,
        which is what gave it away in a live scrape.
        """
        await market_client.get("/api/v1/companies")

        output: str = market_app.state.metrics.render()

        assert 'route="/api/v1/companies",status="200"' in output
        assert 'route="unmatched",status="200"' not in output
