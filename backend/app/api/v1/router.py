"""Aggregate router for API version 1.

Every feature module contributes one router here. Versioning at the router level
means a breaking change ships as ``/api/v2`` alongside the existing surface
instead of forcing a coordinated client migration.

Health probes are deliberately *not* included: they are an infrastructure
concern mounted at the unversioned root, so orchestrator manifests never have to
change when the API version does.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints import (
    anomalies,
    chat,
    companies,
    indicators,
    ingestion,
    news,
    prices,
    search,
)

api_router = APIRouter()

api_router.include_router(companies.router, prefix="/companies")
api_router.include_router(companies.tickers_router, prefix="/tickers")
api_router.include_router(prices.router, prefix="/prices")
api_router.include_router(indicators.router, prefix="/indicators")
api_router.include_router(anomalies.router, prefix="/anomalies")
api_router.include_router(chat.correlation_router, prefix="/anomalies")
api_router.include_router(chat.router, prefix="/chat")
api_router.include_router(news.router, prefix="/news")
api_router.include_router(search.router, prefix="/search")
api_router.include_router(ingestion.router, prefix="/ingestion")

# Routers added in later milestones:
#   market-summary, chat (RAG), watchlists,
#   portfolios, auth
