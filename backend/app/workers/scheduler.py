"""APScheduler worker: the platform's background clock.

Runs as its own process, not inside the API. Three reasons, in order of
importance:

1. A scheduler embedded in the API runs once per replica, so scaling the API to
   three pods runs every job three times.
2. A long ingestion job would compete with request handling for the same event
   loop and the same connection pool.
3. The API can be restarted for a deploy without interrupting a running job, and
   the worker can be scaled or paused independently.

APScheduler rather than Celery: the job graph is a handful of cron-like tasks
with no fan-out and no result passing, and Celery's broker would be operational
weight bought for nothing at this scale.
"""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.workers.jobs import (
    JobContext,
    compute_features_job,
    detect_anomalies_job,
    index_documents_job,
    ingest_news_job,
    ingest_prices_job,
    score_sentiment_job,
)

logger = get_logger(__name__)

type Job = Callable[[JobContext], Awaitable[None]]


def build_scheduler(settings: Settings, context: JobContext) -> AsyncIOScheduler:
    """Create the scheduler with every job registered.

    Args:
        settings: Cron expressions and misfire policy.
        context: Shared connection pools passed to each job.

    Returns:
        A configured, not-yet-started scheduler.

    Notes:
        ``max_instances=1`` and ``coalesce`` together give the behaviour
        ingestion needs: a run that overruns its next trigger is skipped rather
        than started concurrently, and a backlog accumulated while the worker
        was down collapses into a single catch-up run. The jobs are idempotent,
        so a skipped tick loses nothing -- the next run's window still covers it.
    """
    scheduler = AsyncIOScheduler(
        timezone=settings.scheduler.timezone,
        job_defaults={
            "coalesce": settings.scheduler.coalesce_missed_runs,
            "max_instances": 1,
            "misfire_grace_time": settings.scheduler.misfire_grace_seconds,
        },
    )

    schedule = (
        JobSpec("ingest_prices", ingest_prices_job, settings.scheduler.price_ingestion_cron),
        JobSpec("ingest_news", ingest_news_job, settings.scheduler.news_ingestion_cron),
        # Deliberately 30 minutes after price ingestion: features are derived
        # from prices, so they must not race the batch that produces them.
        JobSpec(
            "compute_features",
            compute_features_job,
            settings.scheduler.feature_computation_cron,
        ),
        # After features, for the same reason features run after prices: the
        # detectors consume what the previous stage produces.
        JobSpec(
            "detect_anomalies",
            detect_anomalies_job,
            settings.scheduler.anomaly_detection_cron,
        ),
        JobSpec(
            "score_sentiment",
            score_sentiment_job,
            settings.scheduler.sentiment_scoring_cron,
        ),
        JobSpec(
            "index_documents",
            index_documents_job,
            settings.scheduler.embedding_cron,
        ),
    )
    for spec in schedule:
        _register(scheduler, spec, context, timezone=settings.scheduler.timezone)

    _register_heartbeat(scheduler, settings)
    return scheduler


def _register_heartbeat(scheduler: AsyncIOScheduler, settings: Settings) -> None:
    """Register the liveness heartbeat.

    The worker serves no HTTP, so "is it alive?" has to be answered some other
    way. Checking that the *process* exists is not enough: a scheduler whose
    event loop has wedged still has a running process, and would pass such a
    check forever while running nothing.

    Writing a file from inside the scheduler proves the loop is still
    dispatching jobs. The container healthcheck then only has to look at the
    file's age -- no extra packages in the runtime image.
    """
    path = Path(settings.scheduler.heartbeat_path)

    def touch_heartbeat() -> None:
        path.write_text(datetime.now(UTC).isoformat())

    scheduler.add_job(
        touch_heartbeat,
        trigger=IntervalTrigger(seconds=settings.scheduler.heartbeat_interval_seconds),
        id="heartbeat",
        name="heartbeat",
        replace_existing=True,
        next_run_time=datetime.now(UTC),  # write once immediately, not after a full interval
    )
    logger.info("heartbeat_registered", path=str(path))


@dataclass(frozen=True)
class JobSpec:
    """One scheduled job: its id, the coroutine to run, and when to run it."""

    job_id: str
    job: Job
    cron: str


def _register(
    scheduler: AsyncIOScheduler,
    spec: JobSpec,
    context: JobContext,
    *,
    timezone: str,
) -> None:
    """Attach one cron-triggered job to the scheduler."""
    scheduler.add_job(
        spec.job,
        trigger=CronTrigger.from_crontab(spec.cron, timezone=timezone),
        args=[context],
        id=spec.job_id,
        name=spec.job_id,
        replace_existing=True,
    )
    logger.info("job_registered", job=spec.job_id, cron=spec.cron, timezone=timezone)


async def run_worker(settings: Settings | None = None) -> None:
    """Start the scheduler and run until a termination signal arrives.

    Handles SIGTERM and SIGINT so a container stop drains cleanly: the scheduler
    stops accepting new triggers, in-flight jobs finish, and the connection
    pools close. Without this, a deploy would sever a half-written ingestion
    batch -- recoverable, because writes are idempotent, but noisy.
    """
    settings = settings or get_settings()
    configure_logging(settings.observability)

    if not settings.scheduler.enabled:
        logger.warning("scheduler_disabled", reason="SCHEDULER_ENABLED is false")
        return

    context = JobContext.create(settings)
    scheduler = build_scheduler(settings, context)
    shutdown = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown.set)

    scheduler.start()
    logger.info(
        "worker_started",
        jobs=[job.id for job in scheduler.get_jobs()],
        environment=settings.environment.value,
    )

    try:
        await shutdown.wait()
    finally:
        logger.info("worker_stopping")
        scheduler.shutdown(wait=True)
        await context.aclose()
        logger.info("worker_stopped")


def main() -> None:
    """Entrypoint for ``python -m app.workers``."""
    asyncio.run(run_worker())
