# AI Market Intelligence Platform

Real-time intelligence over the AI-infrastructure and semiconductor ecosystem —
HBM, DRAM, GPUs, foundries and hyperscalers. The platform ingests prices and
news, engineers technical features, detects anomalous market behaviour,
correlates those anomalies with news, and answers natural-language questions
through a retrieval-augmented pipeline that always cites its sources.

> **Status:** Milestones 1–2 of 10 complete — backend foundation plus the full
> domain model, migrations and repository layer.

---

## What it answers

- *Why did AMD move today?*
- *What happened in the HBM market this week?*
- *Which companies benefited from AI infrastructure spending?*
- *Show me unusual price movements this month.*
- *Compare NVIDIA and AMD over the last six months.*

Every answer is grounded in retrieved documents — prices, filings, news,
anomalies — and returns its sources and a confidence score. The model is never
allowed to answer from parametric memory alone.

---

## Architecture at a glance

```
┌──────────────┐   REST/WS    ┌──────────────────────────────────────┐
│   Frontend   │ ───────────► │            FastAPI backend           │
│ React + Vite │              │  api → services → repositories → db  │
└──────────────┘              └───────────────┬──────────────────────┘
                                              │
                       ┌──────────────────────┼──────────────────────┐
                       ▼                      ▼                      ▼
              ┌────────────────┐   ┌────────────────────┐   ┌────────────────┐
              │  PostgreSQL    │   │      MongoDB       │   │  OpenAI API    │
              │  prices,       │   │  news, filings,    │   │  embeddings,   │
              │  indicators,   │   │  transcripts,      │   │  synthesis     │
              │  anomalies,    │   │  RAG chunks +      │   └────────────────┘
              │  users         │   │  vector index      │
              └────────────────┘   └────────────────────┘
                       ▲
              ┌────────┴─────────┐
              │  APScheduler     │  price ingest · news ingest · features
              │  background jobs │  sentiment · anomalies · embeddings
              └──────────────────┘
```

Dependencies point inward only: `api → services → repositories → infrastructure`.
A service never imports FastAPI; a repository never imports a service. See
[docs/architecture.md](docs/architecture.md) for the full rationale.

---

## Tech stack

| Layer            | Choice                                                            |
| ---------------- | ----------------------------------------------------------------- |
| API              | Python 3.12, FastAPI, Pydantic v2, Uvicorn                        |
| Relational store | PostgreSQL 16 (Supabase), SQLAlchemy 2 async, Alembic             |
| Document store   | MongoDB 7 (Atlas), Motor, Atlas Vector Search                     |
| AI               | OpenAI (chat + embeddings), FinBERT sentiment, hybrid retrieval   |
| Data             | yfinance, NewsAPI, RSS, SEC EDGAR                                 |
| ML               | Isolation Forest + Z-score anomaly detection, technical indicators |
| Frontend         | React, Vite, TailwindCSS, Recharts / Plotly                       |
| Ops              | Docker, docker compose, GitHub Actions, structlog                 |

---

## Quick start

**Prerequisites:** Docker Desktop, GNU Make. Python 3.12+ only if you want to
run the suite outside a container.

```bash
git clone <your-repo-url> ai-market-intelligence
cd ai-market-intelligence
make env      # creates .env from .env.example
make up       # builds and starts postgres, mongo and the API
make migrate  # applies the schema
make seed     # loads the tracked universe and MongoDB indexes
```

| URL                                | What                     |
| ---------------------------------- | ------------------------ |
| http://localhost:8000/docs         | Swagger UI               |
| http://localhost:8000/redoc        | ReDoc                    |
| http://localhost:8000/health/live  | Liveness probe           |
| http://localhost:8000/health/ready | Dependency readiness     |

Verify the stack is healthy:

```bash
curl -s http://localhost:8000/health/ready | python3 -m json.tool
```

### Running the backend without Docker

```bash
make install
cd backend && .venv/bin/uvicorn app.main:app --reload
```

---

## Development

```bash
make check              # lint + typecheck + tests, exactly what CI runs
make format             # auto-fix and format
make test               # unit tests only, no database needed
make test-integration   # repository tests against live PostgreSQL
make migration m="add news table"
make migrate            # apply migrations
make migration-check    # fail if models have drifted from migrations
make seed               # reload reference data (idempotent)
make logs               # tail API logs
```

Install the git hooks once so failures surface before you push:

```bash
pip install pre-commit && pre-commit install
```

---

## Project layout

```
ai-market-intelligence/
├── backend/
│   ├── app/
│   │   ├── api/            # HTTP layer: routers, dependency wiring
│   │   ├── core/           # config, logging, exceptions, middleware
│   │   ├── db/             # engine/session lifecycle, ORM base
│   │   ├── models/         # SQLAlchemy models
│   │   ├── repositories/   # persistence, the only layer that writes SQL
│   │   ├── schemas/        # Pydantic API contracts
│   │   ├── services/       # business logic and orchestration
│   │   └── main.py         # application factory
│   ├── alembic/            # migrations
│   ├── tests/
│   └── Dockerfile
├── docs/
├── docker-compose.yml
├── Makefile
└── .github/workflows/ci.yml
```

---

## Roadmap

| # | Milestone | Status |
| - | --------- | ------ |
| 1 | Foundation: config, DB layer, observability, Docker, CI | ✅ Done |
| 2 | Domain model, migrations, repository layer, seed data | ✅ Done |
| 3 | Ingestion: yfinance, NewsAPI, RSS, SEC; deduplication | Planned |
| 4 | Feature engineering: returns, RSI, MACD, Bollinger, ATR | Planned |
| 5 | FinBERT sentiment + anomaly detection (Isolation Forest, Z-score) | Planned |
| 6 | Vector store, embeddings, hybrid search | Planned |
| 7 | RAG pipeline, news-correlation engine, chat API | Planned |
| 8 | Auth, watchlists, portfolios, WebSocket price stream | Planned |
| 9 | React frontend: dashboard, heatmap, anomalies, chat | Planned |
| 10 | Deployment, monitoring, documentation polish | Planned |

---

## Licence

MIT
