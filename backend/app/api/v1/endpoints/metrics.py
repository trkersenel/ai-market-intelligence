"""Prometheus scrape endpoint.

Mounted at the unversioned root alongside the health probes: a scrape config is
infrastructure, and should not have to change when the API version does.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Request, Response
from sqlalchemy import func, select

from app.api.deps import SessionDep
from app.core.metrics import MetricsRegistry
from app.models.price import DailyPrice

router = APIRouter(tags=["monitoring"], include_in_schema=False)

#: Content type Prometheus expects. Serving `text/plain` without the version
#: parameter works with most scrapers but is not what the spec asks for.
EXPOSITION_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


@router.get("/metrics", summary="Prometheus metrics")
async def metrics(request: Request, session: SessionDep) -> Response:
    """Return the current metrics in Prometheus exposition format.

    Data freshness is computed at scrape time rather than tracked incrementally.
    It is one indexed MAX() every fifteen seconds, and computing it here means
    the gauge is correct even after a restart -- a counter maintained in memory
    would read zero on a fresh process while the data behind it was days old,
    which is exactly the situation the metric exists to catch.
    """
    registry: MetricsRegistry = request.app.state.metrics

    newest = await session.execute(
        select(func.max(DailyPrice.trade_date)).where(DailyPrice.is_provisional.is_(False))
    )
    latest_session = newest.scalar_one_or_none()
    if latest_session is not None:
        age = (datetime.now(UTC).date() - latest_session).days * 86_400
        registry.set_freshness(dataset="daily_prices", age_seconds=float(age))

    return Response(content=registry.render(), media_type=EXPOSITION_CONTENT_TYPE)
