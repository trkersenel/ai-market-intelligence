# Deployment

## Topology

Three services from two images. They are separate because they scale and fail
differently, not for tidiness:

| Service | Image | Shape | Why separate |
| --- | --- | --- | --- |
| API | `backend/Dockerfile` | Stateless, horizontally scalable | Request-bound |
| Worker | same, `INSTALL_ML=true` | **Singleton** | Runs the schedule |
| Frontend | `frontend/Dockerfile` | Static + nginx | No runtime |

**The worker must be a singleton.** Every job is idempotent, so a second replica
would not corrupt data — but it would double every outbound API call, and free
news and price feeds are rate-limited. Scale the API instead.

**The API image deliberately has no torch.** Only the worker scores sentiment, so
the API stays at 971 MB against the worker's 2.14 GB. torch comes from the
CPU-only index: the default Linux wheel bundles the CUDA runtime for a container
that will never see a GPU, which alone would push the image past 4 GB.

## Managed services

| Need | Service | Why not self-hosted |
| --- | --- | --- |
| PostgreSQL | Supabase | Backups and PITR are the whole point |
| Documents + vectors | **MongoDB Atlas** | See below |

Atlas is not a preference. **Local MongoDB has no `$vectorSearch`** — the
platform detects this at startup and falls back to brute-force cosine over the
whole collection, which is correct but linear. That is fine for a few thousand
chunks and wrong for a corpus. Self-hosting the document store silently degrades
retrieval; it does not break it, which is worse.

Create the vector index once, from the definition the code generates:

```bash
python -c "from app.db.mongo_indexes import vector_index_definition; \
import json; print(json.dumps(vector_index_definition(index_name='rag_documents_vector_index'), indent=2))"
```

Dimensions must match `EMBED_DIMENSIONS` exactly. Vectors of different widths
are not comparable, so changing the embedding model means rebuilding the index
**and** re-embedding every document.

## First deploy

```bash
# 1. Schema. Runs from the API image, which ships alembic.
docker compose run --rm api alembic upgrade head

# 2. Reference data: 11 companies, 14 listings. Idempotent.
docker compose run --rm api python -m app.db.seed

# 3. Backfill. Two years of prices, then features, then detection.
curl -X POST "$API/api/v1/ingestion/prices"
curl -X POST "$API/api/v1/indicators/compute?full_history=true"
curl -X POST "$API/api/v1/anomalies/detect?lookback_sessions=600"
```

Order matters and is not arbitrary: features are a pure function of prices, and
anomalies a pure function of features. Running detection first produces nothing
and looks like a failure.

## Migrations

Alembic runs as a deploy step, never on application boot. A boot-time migration
means every replica races to apply the same DDL, and a rollback becomes a
coordination problem. `alembic check` runs in CI, so a model change without a
migration fails the build rather than the deploy.

## Configuration

Everything is environment variables; nothing reads `os.environ` outside
`app.core.config`. Secrets are `SecretStr`, so a settings object in a log line
or a traceback prints `**********`.

Required in production:

| Variable | Consequence if unset |
| --- | --- |
| `SECURITY_SECRET_KEY` | Startup fails on the empty check |
| `POSTGRES_*` | No relational store |
| `MONGO_URI` | No documents, no retrieval |
| `INGEST_SEC_USER_AGENT` | SEC refuses anonymous clients outright |

Optional, and the degradation each causes:

| Variable | Without it |
| --- | --- |
| `LLM_OPENAI_API_KEY` | Answers quote sources instead of synthesising |
| `EMBED_OPENAI_API_KEY` | Retrieval matches wording, not meaning |
| `INGEST_NEWSAPI_KEY` | RSS and SEC only |
| `ANALYSIS_USE_FINBERT` | Lexicon sentiment; no contrastive clauses |

Every one of those degrades a single capability and logs why. None prevents
startup — a platform that refuses to boot without four API keys is a platform
nobody can run.

## Monitoring

`/metrics` serves Prometheus exposition; `deploy/prometheus.yml` and
`deploy/alerts.yml` are ready to use.

Four metric families, and the fourth is the one that matters:

- `http_requests_total`, `http_request_duration_seconds` — labelled by route
  *template*. `/api/v1/prices/{symbol}` is one series; the raw path would be one
  per ticker, which on a public endpoint is unbounded cardinality and the
  standard way to take down a Prometheus server with your own instrumentation.
- `scheduler_jobs_total`, `scheduler_job_duration_seconds` — jobs swallow their
  exceptions so a failure cannot kill the schedule, which means a broken job is
  invisible without this.
- `data_age_seconds` — **the metric to page on.** Every request can be fast and
  successful while ingestion has been dead for two days. Nothing else
  distinguishes those two states.

Computed at scrape time rather than tracked in memory, so a fresh process
reports the true age instead of zero.

## Logs

JSON in deployed environments, with a request id bound at the edge and inherited
by every downstream log line — a repository query, an OpenAI call, an anomaly
explanation. The same id is in the `X-Request-ID` response header and in the
error envelope, so a user pasting an error identifies the exact request.

## Health probes

| Path | Meaning | On failure |
| --- | --- | --- |
| `/health/live` | Process is up. Touches nothing. | Restart |
| `/health/ready` | Dependencies answer. | Remove from load balancer |

Conflating them turns a two-second database blip into a restart loop.

## Rollback

The image is immutable and tagged by commit, so rolling back the service is
redeploying the previous tag. Rolling back a *migration* is
`alembic downgrade -1`, which every migration supports because the constraint
naming convention makes the operations reversible.

Deploy in the order that keeps a rollback possible: migrate, then deploy, and
prefer additive migrations so the previous image still runs against the new
schema. A column drop is a two-deploy operation.

## Cost

At the smallest tier that works: API on a starter web service, worker on a
standard instance (torch needs the memory), Supabase and Atlas free tiers. The
free Atlas tier supports vector search, which is what makes this deployable at
zero marginal cost.

Embeddings cost cents per thousand documents. Chat generation is the only
variable cost worth watching, and the extractive fallback means the platform
still answers when it is switched off.
