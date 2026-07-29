"""Background worker process: the scheduler and its jobs.

Runs separately from the API so that scaling the API does not multiply job
executions, and so a long ingestion run never competes with request handling for
the same event loop.
"""

from app.workers.jobs import JobContext, ingest_news_job, ingest_prices_job
from app.workers.scheduler import build_scheduler, main, run_worker

__all__ = [
    "JobContext",
    "build_scheduler",
    "ingest_news_job",
    "ingest_prices_job",
    "main",
    "run_worker",
]
