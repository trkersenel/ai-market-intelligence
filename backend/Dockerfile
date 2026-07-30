# syntax=docker/dockerfile:1.7
#
# Multi-stage build for the FastAPI backend.
#
# The builder stage compiles wheels (asyncpg and psycopg need a toolchain); the
# runtime stage installs those prebuilt wheels into a slim image with no
# compilers present. This keeps the shipped image small and removes build tools
# from the production attack surface.

# ---------------------------------------------------------------------------
# Stage 1 - build dependency wheels
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

# Whether to install the optional `ml` extra (torch + transformers, ~2GB).
# Off by default: the API never needs it, and every deploy of the API would
# otherwise pull two gigabytes to run code it does not execute. The worker
# image is built with it on, so FinBERT runs where sentiment scoring runs.
ARG INSTALL_ML=false

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only the dependency manifest first: this layer is cached and rebuilt
# only when dependencies actually change, not on every source edit.
COPY pyproject.toml ./
COPY app/__init__.py ./app/

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel && pip install .

# torch from the CPU-only index, not PyPI. The default Linux wheel bundles
# the CUDA runtime and cuDNN -- roughly 2.5GB of GPU libraries for a
# container that will never see a GPU. The CPU build is an order of
# magnitude smaller and identical for inference here, which is what makes
# shipping FinBERT in the worker image practical at all.
RUN if [ "$INSTALL_ML" = "true" ]; then \
        pip install --index-url https://download.pytorch.org/whl/cpu torch \
        && pip install "transformers>=4.44"; \
    fi

# ---------------------------------------------------------------------------
# Stage 2 - runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# Where transformers caches downloaded weights. Set explicitly and owned by
# the app user: the default is under $HOME, which a read-only or differently
# owned root filesystem may not permit writing to.
ENV HF_HOME=/app/.cache/huggingface

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install --no-install-recommends -y libpq5 curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1001 appuser \
    && useradd --system --uid 1001 --gid appuser --create-home appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
RUN mkdir -p /app/.cache/huggingface && chown -R appuser:appuser /app/.cache
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser alembic ./alembic
COPY --chown=appuser:appuser app ./app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://localhost:8000/health/live || exit 1

# Uvicorn runs single-process here; horizontal scaling is the orchestrator's job,
# which keeps restarts, rolling deploys and metrics per-container meaningful.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
