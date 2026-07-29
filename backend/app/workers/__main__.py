"""Allow the worker to be started with ``python -m app.workers``."""

from app.workers.scheduler import main

main()
