"""Structured logging setup.

The platform emits machine-parseable JSON logs in deployed environments and
human-friendly coloured output locally. Every log record carries the ambient
request context (request id, method, path) via ``structlog`` context variables,
which is what makes an anomaly or RAG query traceable end to end across the API,
the scheduler and the background workers.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from app.core.config import ObservabilitySettings

#: Third-party loggers that are too chatty at INFO level.
_NOISY_LOGGERS: dict[str, int] = {
    "uvicorn.access": logging.WARNING,
    "asyncio": logging.WARNING,
    "pymongo": logging.WARNING,
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
}


def _drop_color_message_key(_: Any, __: str, event_dict: EventDict) -> EventDict:
    """Remove uvicorn's duplicated ``color_message`` key from the payload."""
    event_dict.pop("color_message", None)
    return event_dict


def configure_logging(settings: ObservabilitySettings) -> None:
    """Configure ``structlog`` and route the stdlib logging tree through it.

    Args:
        settings: Observability configuration controlling level and renderer.

    Notes:
        This is idempotent and safe to call from both the API entrypoint and the
        worker entrypoint.
    """
    level = logging.getLevelNamesMapping()[settings.level]

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.ExtraAdder(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        _drop_color_message_key,
    ]

    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if settings.renderer == "json"
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[
            *shared_processors,
            structlog.processors.format_exc_info,
        ],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    for name, noisy_level in _NOISY_LOGGERS.items():
        logging.getLogger(name).setLevel(max(level, noisy_level))


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structured logger.

    Args:
        name: Logger name, conventionally ``__name__`` of the calling module.

    Returns:
        A ``structlog`` bound logger that inherits the ambient request context.
    """
    return structlog.stdlib.get_logger(name)
