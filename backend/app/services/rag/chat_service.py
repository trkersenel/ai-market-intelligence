"""Conversational wrapper over the RAG pipeline.

Conversations are persisted so an answer can be audited later -- which documents
were retrieved, which model produced it, how confident the platform was. For a
tool whose output informs money decisions, "what did it tell me last Tuesday and
on what basis" has to be answerable.

History is deliberately *not* fed back into retrieval as raw text. Concatenating
previous turns into the query dilutes it: a question about Micron following one
about NVIDIA retrieves a blend of both and answers neither well. Only an explicit
follow-up reference is resolved, and only from the immediately preceding turn.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from app.core.logging import get_logger
from app.repositories.documents import ChatRepository
from app.schemas.documents import ChatMessage
from app.services.rag.rag_service import RagAnswer, RagService

logger = get_logger(__name__)

#: Phrases that make a question depend on the previous turn. Only these trigger
#: history resolution; everything else is treated as a fresh question.
_FOLLOW_UP_MARKERS = re.compile(
    r"\b(it|its|that|this|they|them|those|the company|the stock|same)\b", re.IGNORECASE
)

#: Turns kept for display. The pipeline itself uses at most one.
DEFAULT_HISTORY_TURNS = 20


@dataclass(frozen=True)
class ChatTurn:
    """One question and its answer."""

    conversation_id: str
    answer: RagAnswer
    #: What the user typed. Kept separately because ``answer.question`` holds
    #: the *resolved* text -- the pipeline never sees the original -- so
    #: comparing the two would always report "not resolved".
    original_question: str
    resolved_question: str

    @property
    def was_resolved(self) -> bool:
        """Whether the question was rewritten using the previous turn."""
        return self.resolved_question != self.original_question


class ChatService:
    """Runs the RAG pipeline in the context of a stored conversation."""

    def __init__(self, *, rag: RagService, chat: ChatRepository) -> None:
        """Wire the service to its collaborators."""
        self._rag = rag
        self._chat = chat

    async def ask(
        self,
        question: str,
        *,
        user_id: str,
        conversation_id: str | None = None,
        tickers: list[str] | None = None,
        days: int | None = None,
    ) -> ChatTurn:
        """Answer a question and record both sides of the exchange.

        Args:
            question: The user's question.
            user_id: Owner of the conversation.
            conversation_id: Continues an existing conversation, or starts one.
            tickers: Optional symbol filter.
            days: Optional recency filter.
        """
        conversation = conversation_id or str(uuid.uuid4())
        resolved = await self._resolve(question, conversation)

        started = datetime.now(UTC)
        answer = await self._rag.answer(resolved, tickers=tickers, days=days)
        latency_ms = (datetime.now(UTC) - started).total_seconds() * 1000

        await self._chat.append(
            ChatMessage(
                conversation_id=conversation,
                user_id=user_id,
                role="user",
                content=question,
                created_at=started,
            )
        )
        await self._chat.append(
            ChatMessage(
                conversation_id=conversation,
                user_id=user_id,
                role="assistant",
                content=answer.answer,
                created_at=datetime.now(UTC),
                retrieved_document_ids=[c.source_id for c in answer.citations],
                confidence=answer.confidence,
                model_name=answer.model_name,
                latency_ms=round(latency_ms, 2),
            )
        )

        return ChatTurn(
            conversation_id=conversation,
            answer=answer,
            original_question=question,
            resolved_question=resolved,
        )

    async def history(
        self, conversation_id: str, *, limit: int = DEFAULT_HISTORY_TURNS
    ) -> list[ChatMessage]:
        """Return a conversation in order."""
        return await self._chat.list_conversation(conversation_id, limit=limit)

    async def _resolve(self, question: str, conversation_id: str) -> str:
        """Rewrite a follow-up into a standalone question.

        Only fires when the question contains a referring expression *and* a
        previous user turn exists. Prepending history unconditionally would
        blend two topics into one query and answer neither well; doing it never
        would break the natural "and what about volume?" follow-up.

        The resolution is deliberately crude -- prepending the previous
        question's subject rather than doing coreference properly. A wrong
        rewrite is visible in the response, since the resolved question is
        returned alongside the answer.
        """
        if not _FOLLOW_UP_MARKERS.search(question):
            return question

        previous = await self._chat.last_user_message(conversation_id)
        if previous is None:
            return question

        logger.info("chat_followup_resolved", conversation_id=conversation_id)
        return f"{previous.content} {question}"
