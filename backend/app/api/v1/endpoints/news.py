"""News endpoints backed by the document store."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import NewsRepoDep
from app.models.enums import EcosystemTag
from app.schemas.market import NewsArticleResponse

router = APIRouter(tags=["news"])

DEFAULT_LOOKBACK_DAYS = 7
MAX_LOOKBACK_DAYS = 90


@router.get(
    "",
    response_model=list[NewsArticleResponse],
    summary="List recent news",
    description=(
        "Reverse-chronological, filtered by ticker or ecosystem segment. "
        "Article bodies are omitted; use the source URL for the full text."
    ),
)
async def list_news(
    news: NewsRepoDep,
    tickers: Annotated[
        list[str] | None, Query(description="Symbols the article must mention.")
    ] = None,
    tags: Annotated[
        list[EcosystemTag] | None, Query(description="Segments the article must cover.")
    ] = None,
    days: Annotated[
        int, Query(ge=1, le=MAX_LOOKBACK_DAYS, description="Lookback window in days.")
    ] = DEFAULT_LOOKBACK_DAYS,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[NewsArticleResponse]:
    """Return recent articles matching the filters."""
    since = datetime.now(UTC) - timedelta(days=days)
    articles = await news.list_recent(
        since=since,
        tickers=tickers,
        tags=[tag.value for tag in tags] if tags else None,
        limit=limit,
    )
    return [NewsArticleResponse.from_document(article) for article in articles]


@router.get(
    "/search",
    response_model=list[NewsArticleResponse],
    summary="Keyword search over news",
    description=(
        "Full-text search ranked by relevance. This is the keyword half of "
        "hybrid retrieval; semantic search arrives with the vector index."
    ),
)
async def search_news(
    news: NewsRepoDep,
    q: Annotated[str, Query(min_length=2, max_length=200, description="Search terms.")],
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> list[NewsArticleResponse]:
    """Return articles matching ``q``, most relevant first."""
    articles = await news.search_text(q, limit=limit)
    return [NewsArticleResponse.from_document(article) for article in articles]
