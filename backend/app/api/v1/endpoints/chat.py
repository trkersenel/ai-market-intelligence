"""Conversational question-answering endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Query, status
from pydantic import BaseModel, Field

from app.api.deps import ChatServiceDep, CorrelationEngineDep
from app.schemas.market import ChatAnswerPayload, ChatMessagePayload, IngestionRunResponse

router = APIRouter(tags=["chat"])

#: Placeholder until authentication lands in Milestone 8. Conversations are
#: already scoped by user id in storage, so adding real identity is a change to
#: this one dependency rather than to the schema.
ANONYMOUS_USER = "anonymous"


class ChatRequest(BaseModel):
    """A question, optionally continuing a conversation."""

    question: Annotated[str, Field(min_length=3, max_length=1000)]
    conversation_id: str | None = Field(
        default=None, description="Continue an existing conversation."
    )
    tickers: list[str] | None = Field(
        default=None, description="Restrict retrieval to these symbols."
    )
    days: int | None = Field(
        default=None, ge=1, le=365, description="Restrict retrieval to recent documents."
    )


@router.post(
    "",
    response_model=ChatAnswerPayload,
    summary="Ask a grounded question",
    description=(
        "Retrieves evidence, then answers from it. Every answer carries its "
        "citations and a confidence derived from the evidence rather than "
        "self-reported by the model. When nothing relevant is retrieved the "
        "platform declines to answer instead of guessing."
    ),
)
async def ask(request: ChatRequest, service: ChatServiceDep) -> ChatAnswerPayload:
    """Answer a question and record the exchange."""
    turn = await service.ask(
        request.question,
        user_id=ANONYMOUS_USER,
        conversation_id=request.conversation_id,
        tickers=request.tickers,
        days=request.days,
    )
    return ChatAnswerPayload.from_turn(turn)


@router.get(
    "/{conversation_id}",
    response_model=list[ChatMessagePayload],
    summary="Replay a conversation",
    responses={status.HTTP_404_NOT_FOUND: {"description": "No such conversation."}},
)
async def get_conversation(
    conversation_id: str,
    service: ChatServiceDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[ChatMessagePayload]:
    """Return a conversation in order, with the evidence each answer used."""
    messages = await service.history(conversation_id, limit=limit)
    return [ChatMessagePayload.from_message(message) for message in messages]


correlation_router = APIRouter(tags=["anomalies"])


@correlation_router.post(
    "/explain",
    response_model=IngestionRunResponse,
    summary="Explain unexplained anomalies",
    description=(
        "Correlates each unexplained anomaly with news published around its "
        "session and writes a ranked, hedged explanation. Results are "
        "correlations in time and topic, not established causes."
    ),
)
async def explain_anomalies(
    engine: CorrelationEngineDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> IngestionRunResponse:
    """Run the correlation engine over pending anomalies."""
    started = datetime.now(UTC)
    results = await engine.explain_pending(limit=limit)
    finished = datetime.now(UTC)

    return IngestionRunResponse(
        started_at=started,
        finished_at=finished,
        duration_seconds=round((finished - started).total_seconds(), 3),
        items_written=len(results),
        failures=[],
    )
