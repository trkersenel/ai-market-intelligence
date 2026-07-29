"""API response schemas for reference data, prices and news.

Deliberately separate from the ORM models. A response schema is a *contract*
with the frontend: it changes when the API changes, not when a column is added.
Serialising ORM objects directly would make every internal column a public field
and every schema change a breaking one.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field

from app.models.anomaly import Anomaly
from app.models.company import Company, Ticker
from app.models.enums import AssetType, Sentiment
from app.models.price import DailyPrice
from app.schemas.documents import NewsArticle
from app.services.rag.search_service import SearchResponse


class TickerSummary(BaseModel):
    """A tradable listing, as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    symbol: str
    display_name: str
    exchange: str | None
    currency: str
    asset_type: AssetType
    is_active: bool
    last_price_date: date | None


class CompanySummary(BaseModel):
    """A company without its listings, for list endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    slug: str
    name: str
    sector: str | None
    industry: str | None
    country: str | None
    tags: list[str]
    is_tracked: bool


class CompanyDetail(CompanySummary):
    """A company with its listings and description."""

    website: str | None = None
    description: str | None = None
    tickers: list[TickerSummary] = Field(default_factory=list)

    @classmethod
    def from_model(cls, company: Company) -> Self:
        """Build from an ORM instance whose ``tickers`` were eagerly loaded.

        Explicit rather than relying on ``from_attributes`` recursion: the
        relationship is ``lazy="raise_on_sql"``, so a caller that forgot to load
        it gets a clear error here instead of a lazy-load exception buried in
        serialisation.
        """
        return cls(
            id=company.id,
            slug=company.slug,
            name=company.name,
            sector=company.sector,
            industry=company.industry,
            country=company.country,
            tags=list(company.tags),
            is_tracked=company.is_tracked,
            website=company.website,
            description=company.description,
            tickers=[TickerSummary.model_validate(t) for t in company.tickers],
        )


class PriceBarResponse(BaseModel):
    """One OHLCV bar.

    Decimals serialise as JSON numbers. They stay ``Decimal`` in Python all the
    way to the encoder, so no rounding happens before the wire.
    """

    model_config = ConfigDict(from_attributes=True)

    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    adjusted_close: Decimal
    volume: int


#: A return needs a start and an end; one bar defines no period.
MIN_BARS_FOR_RETURN = 2


class PriceSeries(BaseModel):
    """A ticker's bars over a window, with the derived summary a chart needs."""

    symbol: str
    start: date | None
    end: date | None
    bars: list[PriceBarResponse]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count(self) -> int:
        """Number of sessions returned."""
        return len(self.bars)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def period_return(self) -> Decimal | None:
        """Total return across the window, from adjusted closes.

        Adjusted, not raw: a split inside the window would otherwise show as a
        catastrophic loss. ``None`` when the window is too short to define one.
        """
        if len(self.bars) < MIN_BARS_FOR_RETURN:
            return None
        first = self.bars[0].adjusted_close
        last = self.bars[-1].adjusted_close
        if first == 0:
            return None
        return (last - first) / first


class TickerQuote(BaseModel):
    """The latest bar for a listing, with its one-session change."""

    symbol: str
    display_name: str
    trade_date: date
    close: Decimal
    previous_close: Decimal | None
    volume: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def change_percent(self) -> Decimal | None:
        """Session-over-session percentage change, or ``None`` on the first bar."""
        if self.previous_close is None or self.previous_close == 0:
            return None
        return (self.close - self.previous_close) / self.previous_close * 100


class IndicatorSnapshot(BaseModel):
    """Every computed feature for one listing on one session.

    Fields are optional because an indicator is genuinely undefined during its
    warm-up: a listing with 30 sessions of history has an SMA-20 and no SMA-200.
    Returning ``null`` says so; returning 0 would be a lie the frontend would
    happily chart.
    """

    model_config = ConfigDict(from_attributes=True)

    trade_date: date

    daily_return: Decimal | None = None
    weekly_return: Decimal | None = None
    monthly_return: Decimal | None = None

    sma_20: Decimal | None = None
    sma_50: Decimal | None = None
    sma_200: Decimal | None = None
    ema_12: Decimal | None = None
    ema_26: Decimal | None = None

    rsi_14: Decimal | None = None
    macd: Decimal | None = None
    macd_signal: Decimal | None = None
    macd_histogram: Decimal | None = None

    bollinger_upper: Decimal | None = None
    bollinger_middle: Decimal | None = None
    bollinger_lower: Decimal | None = None
    atr_14: Decimal | None = None
    volatility_20: Decimal | None = None

    volume_sma_20: Decimal | None = None
    volume_ratio: Decimal | None = None
    relative_strength_smh: Decimal | None = None


class IndicatorSeries(BaseModel):
    """A listing's indicators over a window."""

    symbol: str
    start: date | None
    end: date | None
    rows: list[IndicatorSnapshot]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count(self) -> int:
        """Number of sessions returned."""
        return len(self.rows)


class NewsArticleResponse(BaseModel):
    """A news article as returned by the API, without the raw body."""

    id: str | None
    url: str
    title: str
    summary: str | None
    source: str
    source_name: str | None
    published_at: datetime
    tickers: list[str]
    tags: list[str]
    sentiment: Sentiment | None = None
    sentiment_confidence: Annotated[float, Field(ge=0, le=1)] | None = None

    @classmethod
    def from_document(cls, article: NewsArticle) -> Self:
        """Build from a stored document.

        The full ``content`` is omitted: it can be tens of kilobytes, it is only
        needed for embedding and summarisation, and a news feed that ships it
        would be dominated by text nobody renders.
        """
        return cls(
            id=article.id,
            url=article.url,
            title=article.title,
            summary=article.summary,
            source=article.source.value,
            source_name=article.source_name,
            published_at=article.published_at,
            tickers=list(article.tickers),
            tags=list(article.tags),
            sentiment=article.sentiment.label if article.sentiment else None,
            sentiment_confidence=(article.sentiment.confidence if article.sentiment else None),
        )


class AnomalyResponse(BaseModel):
    """A detected anomaly, with the symbol resolved."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    symbol: str
    trade_date: date
    anomaly_type: str
    method: str
    direction: str
    severity: str
    score: float
    confidence: float
    explanation: str | None
    related_document_ids: list[str]

    @classmethod
    def from_row(cls, anomaly: Anomaly, symbol: str) -> Self:
        """Build from the ``(Anomaly, symbol)`` row the repository returns."""
        return cls(
            id=anomaly.id,
            symbol=symbol,
            trade_date=anomaly.trade_date,
            anomaly_type=anomaly.anomaly_type.value,
            method=anomaly.method.value,
            direction=anomaly.direction.value,
            severity=anomaly.severity.value,
            score=anomaly.score,
            confidence=anomaly.confidence,
            explanation=anomaly.explanation,
            related_document_ids=list(anomaly.related_document_ids),
        )


class SearchResultPayload(BaseModel):
    """One fused search result."""

    text: str
    score: float
    source_id: str
    source_url: str | None = None
    title: str | None = None
    tickers: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    published_at: datetime | None = None
    #: Which retrievers surfaced this result, and where each ranked it. Exposed
    #: rather than kept internal because a RAG citation has to be auditable: a
    #: user asking "why was this cited?" deserves an answer that does not
    #: require re-running the query.
    matched_by: list[str] = Field(default_factory=list)
    ranks: dict[str, int] = Field(default_factory=dict)


class SearchResponsePayload(BaseModel):
    """A search response, including how it was served."""

    query: str
    mode: str
    #: 'atlas' or 'brute_force'. Surfaced so a result set is attributable to the
    #: backend that produced it -- approximate and exact search do not always
    #: agree, and silently switching between them would make that undebuggable.
    backend: str
    count: int
    results: list[SearchResultPayload]

    @classmethod
    def from_response(cls, response: SearchResponse) -> Self:
        """Build from the service's response object."""
        return cls(
            query=response.query,
            mode=response.mode.value,
            backend=response.backend,
            count=response.count,
            results=[
                SearchResultPayload(
                    text=result.text,
                    score=result.score,
                    source_id=result.source_id,
                    source_url=result.source_url,
                    title=result.title,
                    tickers=list(result.tickers),
                    tags=list(result.tags),
                    published_at=result.published_at,
                    matched_by=list(result.matched_by),
                    ranks=dict(result.ranks),
                )
                for result in response.results
            ],
        )


class IngestionRunResponse(BaseModel):
    """Outcome of a manually triggered ingestion run."""

    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    items_written: int
    failures: list[str] = Field(default_factory=list)


def quote_from_bars(ticker: Ticker, latest: DailyPrice, previous: DailyPrice | None) -> TickerQuote:
    """Assemble a quote from a listing and its two most recent bars."""
    return TickerQuote(
        symbol=ticker.symbol,
        display_name=ticker.display_name,
        trade_date=latest.trade_date,
        close=latest.close,
        previous_close=previous.close if previous else None,
        volume=latest.volume,
    )
