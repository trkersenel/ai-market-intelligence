"""Split documents into passages for embedding.

Chunk size is a retrieval decision, not a formatting one. Embed a whole article
and its vector becomes the average of everything it discusses, so a story that
mentions HBM once in paragraph nine is no longer retrievable by an HBM query.
Chunk too finely and each passage loses the context that made it meaningful.

Boundaries are chosen at natural breaks -- paragraph, then sentence, then word --
rather than at a fixed character offset, because a chunk that starts mid-clause
embeds badly and reads worse when it is later quoted as a citation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Break points in descending order of preference. A paragraph boundary is the
#: best place to split; a mid-word cut is the worst and is only ever a fallback.
_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

#: A trailing fragment shorter than this is folded into the previous chunk
#: instead of being emitted alone. A 30-character chunk carries no retrievable
#: meaning but still costs an embedding, an index entry and a candidate slot.
MIN_CHUNK_CHARS = 120


@dataclass(frozen=True)
class Chunk:
    """One passage, with its position in the parent document."""

    text: str
    index: int

    @property
    def char_count(self) -> int:
        """Length of the passage."""
        return len(self.text)


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[Chunk]:
    """Split ``text`` into overlapping passages.

    Args:
        text: The document body.
        chunk_size: Target characters per chunk.
        overlap: Characters of the previous chunk repeated at the start of the
            next. Prevents a sentence spanning a boundary from being lost --
            without it, the single most relevant passage can be the one split
            down the middle, retrievable from neither side.

    Returns:
        Passages in document order, each carrying its index. An empty or
        whitespace-only input yields no chunks.

    Raises:
        ValueError: If ``overlap`` is not smaller than ``chunk_size``, which
            would make the window advance by zero and loop forever.
    """
    if chunk_size <= 0:
        msg = "chunk_size must be positive"
        raise ValueError(msg)
    if overlap >= chunk_size:
        msg = "overlap must be smaller than chunk_size"
        raise ValueError(msg)

    normalised = _normalise(text)
    if not normalised:
        return []
    if len(normalised) <= chunk_size:
        return [Chunk(text=normalised, index=0)]

    chunks: list[str] = []
    start = 0
    while start < len(normalised):
        end = min(start + chunk_size, len(normalised))
        if end < len(normalised):
            end = _best_boundary(normalised, start=start, end=end)

        piece = normalised[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= len(normalised):
            break
        start = max(end - overlap, start + 1)

    merged = _fold_short_tail(chunks)
    return [Chunk(text=piece, index=index) for index, piece in enumerate(merged)]


def _normalise(text: str) -> str:
    """Collapse runs of whitespace while preserving paragraph breaks."""
    collapsed = re.sub(r"[ \t]+", " ", text.strip())
    return re.sub(r"\n{3,}", "\n\n", collapsed)


def _best_boundary(text: str, *, start: int, end: int) -> int:
    """Find the most natural split point at or before ``end``.

    Searches backwards through the final third of the window only. Looking
    further back would honour the boundary at the cost of a chunk half the
    intended size, which trades one problem for another.
    """
    floor = start + (end - start) * 2 // 3

    paragraph = _last_match(_PARAGRAPH, text, floor, end)
    if paragraph is not None:
        return paragraph

    sentence = _last_match(_SENTENCE_END, text, floor, end)
    if sentence is not None:
        return sentence

    space = text.rfind(" ", floor, end)
    if space > floor:
        return space + 1

    # No natural boundary in range: a hard cut is better than an oversized chunk.
    return end


def _last_match(pattern: re.Pattern[str], text: str, floor: int, end: int) -> int | None:
    """Return the end offset of the last pattern match within a window."""
    matches = list(pattern.finditer(text, floor, end))
    return matches[-1].end() if matches else None


def _fold_short_tail(chunks: list[str]) -> list[str]:
    """Merge a too-short final chunk into its predecessor."""
    if len(chunks) > 1 and len(chunks[-1]) < MIN_CHUNK_CHARS:
        tail = chunks.pop()
        chunks[-1] = f"{chunks[-1]} {tail}".strip()
    return chunks
