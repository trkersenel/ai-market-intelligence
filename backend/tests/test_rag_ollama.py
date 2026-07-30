"""Contract tests for the local Ollama adapters.

Every test serves a recorded response through ``httpx.MockTransport``, so the
suite runs identically on a machine with Ollama installed and one without. That
matters more here than for a remote vendor: Ollama's presence is a property of
the developer's laptop, and a suite whose outcome depends on it is a suite that
proves nothing.
"""

from __future__ import annotations

import re

import httpx
import pytest

from app.core.config import IngestionSettings, OllamaSettings
from app.services.rag.ollama import (
    OllamaChatClient,
    OllamaEmbeddingProvider,
    OllamaUnavailableError,
    _model_names,
)

FAST = IngestionSettings(max_retries=0, retry_backoff_seconds=0.001)


def _client(handler: object) -> httpx.AsyncClient:
    """Build a transport that answers with the given handler."""
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
        base_url="http://ollama.test",
    )


class TestProbe:
    """Availability detection during provider selection."""

    def test_both_spellings_of_a_model_are_reported(self) -> None:
        """A user may configure "llama3.2" for a server holding "llama3.2:3b".

        Returning only the tagged form would make selection fall back to OpenAI
        while the model sits pulled and unused -- a silent downgrade rather than
        an error.
        """
        names = _model_names({"models": [{"name": "llama3.2:3b"}, {"name": "nomic-embed-text"}]})

        assert names == {"llama3.2:3b", "llama3.2", "nomic-embed-text"}

    @pytest.mark.parametrize(
        "payload",
        [None, {}, {"models": "not-a-list"}, {"models": [None, {"name": ""}]}],
        ids=["null", "empty", "wrong-type", "unusable-entries"],
    )
    def test_a_malformed_body_yields_no_models(self, payload: object) -> None:
        """Selection reads this during startup; a surprising body must not raise."""
        assert _model_names(payload) == set()

    async def test_an_unreachable_server_is_an_answer_not_an_error(self) -> None:
        """Selection must never raise: a missing Ollama degrades a feature."""
        names = await OllamaEmbeddingProvider.probe(
            OllamaSettings(base_url="http://ollama.invalid:11434", probe_timeout_seconds=0.05),
            FAST,
        )

        assert names == set()


class TestEmbedding:
    """The vector path."""

    async def test_width_is_discovered_from_the_response(self) -> None:
        """Configured width is a hint; the server's answer is the truth.

        Trusting configuration here would let a mismatch corrupt the collection
        silently, because cosine over differently sized vectors is undefined
        rather than erroneous.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embeddings": [[0.1] * 768]})

        provider = OllamaEmbeddingProvider(
            OllamaSettings(embedding_dimensions=1536), FAST, client=_client(handler)
        )
        vectors = await provider.embed(["one"])

        assert len(vectors[0]) == 768
        assert provider.dimensions == 768
        await provider.aclose()

    async def test_a_short_response_raises_rather_than_misaligning(self) -> None:
        """A short response must raise rather than misalign the corpus.

        Pairing each chunk with its neighbour's embedding is corruption no
        later assertion catches -- every search simply returns the wrong
        article, forever.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})

        provider = OllamaEmbeddingProvider(OllamaSettings(), FAST, client=_client(handler))

        with pytest.raises(OllamaUnavailableError, match="ollama pull"):
            await provider.embed(["one", "two"])
        await provider.aclose()

    async def test_batches_are_requested_in_configured_chunks(self) -> None:
        """Inputs beyond the batch size become several requests, in order."""
        batches: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json  # noqa: PLC0415

            size = len(json.loads(request.content)["input"])
            batches.append(size)
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2]] * size})

        provider = OllamaEmbeddingProvider(
            OllamaSettings(embedding_batch_size=2), FAST, client=_client(handler)
        )
        vectors = await provider.embed(["a", "b", "c", "d", "e"])

        assert batches == [2, 2, 1]
        assert len(vectors) == 5
        await provider.aclose()

    async def test_no_input_makes_no_request(self) -> None:
        """An empty index run must not wake the model."""
        called = False

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return httpx.Response(200, json={"embeddings": []})

        provider = OllamaEmbeddingProvider(OllamaSettings(), FAST, client=_client(handler))

        assert await provider.embed([]) == []
        assert called is False
        await provider.aclose()


class TestChat:
    """The generation path."""

    async def test_the_grounding_prompt_and_question_are_sent(self) -> None:
        """The system prompt is what holds a small model to the passages."""
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json  # noqa: PLC0415

            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "message": {"content": "HBM demand rose [1]."},
                    "prompt_eval_count": 211,
                    "eval_count": 42,
                },
            )

        client = OllamaChatClient(OllamaSettings(), FAST, client=_client(handler))
        answer = await client.complete(question="Why?", context="[1] HBM demand rose.")

        messages: list[dict[str, str]] = captured["messages"]  # type: ignore[assignment]
        assert messages[0]["role"] == "system"
        assert "[1] HBM demand rose." in messages[1]["content"]
        assert captured["stream"] is False
        assert answer.text == "HBM demand rose [1]."
        assert (answer.prompt_tokens, answer.completion_tokens) == (211, 42)
        await client.aclose()

    async def test_an_empty_completion_names_the_missing_model(self) -> None:
        """Ollama answers 200 with an empty body for a model it does not have."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"message": {"content": "  "}})

        client = OllamaChatClient(OllamaSettings(), FAST, client=_client(handler))

        with pytest.raises(OllamaUnavailableError, match=re.escape("ollama pull llama3.2:3b")):
            await client.complete(question="Why?", context="[1] Something.")
        await client.aclose()


class TestFailureDiagnosis:
    """Telling a stopped server apart from a slow one."""

    async def test_a_timeout_does_not_advise_starting_the_server(self) -> None:
        """A timeout must not be reported as a stopped server.

        The original bug: embedding a corpus timed out on a *running* server and
        the message told the user to start it. Advice for a condition already
        satisfied is worse than a bare error -- it sends the reader to verify
        the one thing that is not wrong.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        client = OllamaChatClient(
            OllamaSettings(request_timeout_seconds=120), FAST, client=_client(handler)
        )

        with pytest.raises(OllamaUnavailableError) as caught:
            await client.complete(question="Why?", context="[1] Something.")

        message = str(caught.value)
        assert "did not respond within 120s" in message
        assert "ollama serve" not in message
        await client.aclose()

    async def test_a_refused_connection_does_advise_starting_the_server(self) -> None:
        """The other branch, which the fix must not have broken."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        client = OllamaChatClient(OllamaSettings(), FAST, client=_client(handler))

        with pytest.raises(OllamaUnavailableError, match="ollama serve"):
            await client.complete(question="Why?", context="[1] Something.")
        await client.aclose()
