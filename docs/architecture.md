# Architecture

This document explains *why* the system is shaped the way it is. Implementation
detail lives in docstrings; the reasoning lives here.

---

## 1. Layering

```
        ┌─────────────────────────────────────────────┐
        │  api/          routers, deps, HTTP schemas   │  knows HTTP
        ├─────────────────────────────────────────────┤
        │  services/     business rules, orchestration │  knows the domain
        ├─────────────────────────────────────────────┤
        │  repositories/ queries, persistence          │  knows storage
        ├─────────────────────────────────────────────┤
        │  db/, clients/ engines, drivers, HTTP calls  │  knows the outside
        └─────────────────────────────────────────────┘
```

**Dependencies point downward only.** A service must never `import fastapi`; a
repository must never import a service. This is not ceremony — it is what makes
the same `AnomalyDetectionService` usable from an HTTP handler, from an
APScheduler job and from a unit test with an in-memory fake, without change.

The rule is mechanically enforceable: if `app/services/` ever contains the
string `fastapi`, the boundary has been violated.

### Why the repository pattern

Anomaly detection needs *"the last 90 daily closes for these 11 tickers"*. That
is a domain question. Whether it is answered by one SQL query, a window function
or a cache is a storage decision. Repositories are where that decision lives, so
swapping a query for a materialised view later touches one file rather than
every caller.

---

## 2. Why two databases

This is the single most-questioned design choice, so it is worth being precise.

**PostgreSQL** holds everything with a fixed schema and relational integrity
requirements: `companies`, `tickers`, `daily_prices`, `technical_indicators`,
`anomalies`, `users`, `watchlists`, `portfolios`. Prices are the canonical
example — a `(ticker_id, date)` unique constraint is what makes ingestion
idempotent, and time-window queries over indexed columns are what Postgres is
best at.

**MongoDB** holds everything whose shape varies by source: a NewsAPI article, an
RSS item, an SEC 8-K and an earnings-call transcript share almost no fields. A
relational schema here would be a wide table of nulls or an EAV mess. It also
hosts the Atlas Vector Search index, so retrieval reads embeddings and document
text in a single round trip instead of joining a vector store to a metadata
store.

The two are linked by ticker symbol and company id, deliberately kept as a soft
reference rather than a distributed transaction. Cross-store consistency is
eventual and reconciled by the ingestion jobs.

**When one store would do:** if the news volume stayed small and uniform, a
single Postgres with `JSONB` and `pgvector` would be a defensible simpler
choice. The two-store split is chosen because news schema genuinely varies per
source and because Atlas Vector Search gives hybrid (keyword + vector) retrieval
without a second system to operate.

---

## 3. Configuration

Every setting is a typed, validated field on a Pydantic model. No module reads
`os.environ`; everything depends on `get_settings()`.

- **Grouped sub-models** (`PostgresSettings`, `MongoSettings`, …) mean a service
  depends only on the slice of config it needs.
- **`SecretStr`** for credentials, so a stray log line or exception repr prints
  `**********` rather than a password.
- **Cached singleton** so validation runs once per process, at startup — a
  misconfigured deployment fails immediately instead of on first request.

---

## 4. Application lifecycle

`create_app()` is a factory, not a module-level singleton. Connection pools and
long-lived clients are created in the `lifespan` context manager and attached to
`app.state`, then disposed on shutdown.

This matters for a platform that runs the same code in three ways — API server,
scheduler process, and test suite. A factory lets each build an app with its own
configuration inside one interpreter; module-level globals would not.

### Liveness vs readiness

Two distinct probes, because orchestrators react differently:

- `/health/live` — the process is up. Touches nothing. A failure restarts the
  container.
- `/health/ready` — dependencies answer. A failure only pulls the instance out of
  the load balancer.

Conflating them turns a two-second database blip into a restart loop.

---

## 5. Observability

`structlog` emits JSON in deployed environments and coloured text locally. The
request id is bound to a context variable at the edge by
`RequestContextMiddleware`, so every log line emitted anywhere downstream — a
repository query, an OpenAI call, an anomaly explanation — carries it without
being passed explicitly.

For a platform whose value proposition is *explaining* market moves, being able
to reconstruct exactly which documents a given answer retrieved is a product
requirement, not just an ops nicety.

---

## 6. Error handling

Domain code raises domain exceptions (`NotFoundError`, `ExternalServiceError`,
…) that carry a machine-readable `code`, an HTTP status and optional structured
details. A single set of handlers translates them into one envelope:

```json
{
  "error": {
    "code": "not_found",
    "message": "Ticker 'XYZ' is not tracked.",
    "request_id": "6f8c…",
    "details": {}
  }
}
```

The frontend branches on `code`, never on message text. The `request_id` in the
body is the same one in the response header and the server logs, so a user can
paste an error and the exact request is retrievable.

---

## 7. Async throughout

The workload is I/O-bound almost everywhere: database round trips, news API
calls, OpenAI requests, SEC fetches. Async lets one worker hold hundreds of
in-flight requests, and lets a readiness check probe both databases
concurrently rather than serially.

The exception is CPU-bound work — FinBERT inference, Isolation Forest fitting.
That runs in a thread or process pool (Milestone 5), never inline on the event
loop, where it would stall every other request.

---

## 8. Testing strategy

| Layer | Approach | Needs infra |
| ----- | -------- | ----------- |
| Services | Fakes injected via constructor | No |
| API | `ASGITransport` + `dependency_overrides` | No |
| Repositories | Real PostgreSQL, transaction rolled back per test | Yes |
| Ingestion | Recorded HTTP fixtures | No |

Unit tests must run with no Docker and no network, or they stop being run.
Integration tests are marked `@pytest.mark.integration` and run against the
service containers in CI.

---

## 9. Deployment

The production image is multi-stage: wheels compile in a builder stage, and the
runtime stage carries no compilers and runs as a non-root user.

Uvicorn runs single-process per container. Horizontal scaling is the
orchestrator's job — that keeps restarts, rolling deploys and per-container
metrics meaningful, which a multi-worker process manager blurs.

Managed services (Supabase, MongoDB Atlas) are used in production; compose
mirrors them locally with the same major versions so developers hit the same
drivers, dialects and failure modes.

---

## 10. Deferred decisions

Recorded here so later milestones do not re-litigate them:

- **Vector store**: Atlas Vector Search, not pgvector or Pinecone — retrieval
  reads text, metadata and embedding in one query, with hybrid search built in.
- **Scheduling**: APScheduler in a dedicated process, not Celery — the job graph
  is a handful of cron-like tasks, and Celery's broker is unjustified operational
  weight at this scale.
- **LLM provider**: accessed behind a narrow `LLMClient` protocol so OpenAI,
  Azure OpenAI or Anthropic are a config change rather than a rewrite.
- **Caching**: deferred until measurements justify it. Redis appears only if a
  profile shows a hot path that needs it.
