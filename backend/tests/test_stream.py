"""Tests for the WebSocket price stream.

Uses Starlette's synchronous ``TestClient``, which is the only supported way to
drive a WebSocket route: ``httpx``'s ASGI transport speaks HTTP only.

The protocol tests matter more than they look. A socket is long-lived, so a frame
handler that raises on bad input does not return an error -- it drops a connection
the client then has to notice and rebuild.
"""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.endpoints import stream
from app.core.config import Environment, Settings
from app.core.exceptions import AuthenticationError
from app.core.security import create_token_pair
from app.main import create_app

TODAY = date(2026, 7, 30)


class FakeUser:
    """Stands in for an authenticated account."""

    def __init__(self, *, is_active: bool = True) -> None:
        self.id = uuid.uuid4()
        self.is_active = is_active


class FakePostgres:
    """Placeholder adapter; the quote loader is patched out in these tests."""


@pytest.fixture
def settings() -> Settings:
    """Test settings with a real signing key."""
    return Settings(
        environment=Environment.TEST,
        debug=True,
        security={"secret_key": "stream-tests-signing-key"},  # type: ignore[arg-type]
    )


@pytest.fixture
def app(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """App with a stubbed quote loader and a stubbed token resolver."""
    application = create_app(settings)
    application.state.postgres = FakePostgres()

    async def fake_resolve(token: str | None, app_state: object, postgres: object) -> FakeUser:
        if token != "valid":
            msg = "Could not validate credentials."
            raise AuthenticationError(msg)
        return FakeUser()

    monkeypatch.setattr(stream, "get_current_user_from_token", fake_resolve)
    monkeypatch.setattr(stream, "POLL_INTERVAL_SECONDS", 0.05)
    return application


def _quotes(*symbols: str) -> list[stream.Quote]:
    return [
        stream.Quote(
            symbol=symbol,
            close=Decimal("100.50"),
            trade_date=TODAY,
            is_provisional=False,
            previous_close=Decimal("100.00"),
        )
        for symbol in symbols
    ]


class TestAuthentication:
    """The handshake."""

    def test_a_missing_token_is_refused_with_a_readable_error(
        self, app: FastAPI, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Accepted then closed, rather than rejected at the handshake.

        A rejected handshake surfaces in the browser as an opaque network error,
        so the client cannot tell an expired token from a dead server.
        """
        monkeypatch.setattr(stream, "_load_quotes", _noop_loader)

        with TestClient(app).websocket_connect("/api/v1/stream/prices") as socket:
            frame = socket.receive_json()

        assert frame["type"] == "error"
        assert "credentials" in frame["message"]

    def test_an_invalid_token_is_refused(
        self, app: FastAPI, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(stream, "_load_quotes", _noop_loader)

        with TestClient(app).websocket_connect("/api/v1/stream/prices?token=forged") as socket:
            frame = socket.receive_json()

        assert frame["type"] == "error"

    def test_a_valid_token_connects(self, app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(stream, "_load_quotes", _noop_loader)

        with TestClient(app).websocket_connect("/api/v1/stream/prices?token=valid") as socket:
            socket.send_json({"action": "ping"})
            assert socket.receive_json() == {"type": "pong"}


class TestProtocol:
    """Frame handling."""

    def test_subscribing_is_acknowledged_with_the_current_set(
        self, app: FastAPI, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(stream, "_load_quotes", _noop_loader)

        with TestClient(app).websocket_connect("/api/v1/stream/prices?token=valid") as socket:
            socket.send_json({"action": "subscribe", "symbols": ["nvda", "mu"]})
            frame = socket.receive_json()

        assert frame["type"] == "subscribed"
        assert frame["symbols"] == ["MU", "NVDA"]

    def test_unsubscribing_removes_symbols(
        self, app: FastAPI, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(stream, "_load_quotes", _noop_loader)

        with TestClient(app).websocket_connect("/api/v1/stream/prices?token=valid") as socket:
            socket.send_json({"action": "subscribe", "symbols": ["NVDA", "MU"]})
            socket.receive_json()
            socket.send_json({"action": "unsubscribe", "symbols": ["MU"]})
            frame = socket.receive_json()

        assert frame["symbols"] == ["NVDA"]

    def test_a_malformed_frame_is_reported_not_fatal(
        self, app: FastAPI, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The connection must survive bad input.

        Dropping the socket would make one typo look like a server failure and
        force the client to reconnect.
        """
        monkeypatch.setattr(stream, "_load_quotes", _noop_loader)

        with TestClient(app).websocket_connect("/api/v1/stream/prices?token=valid") as socket:
            socket.send_json({"action": "nonsense"})
            error = socket.receive_json()

            socket.send_json({"action": "ping"})
            pong = socket.receive_json()

        assert error["type"] == "error"
        assert pong == {"type": "pong"}

    def test_a_non_object_frame_is_reported(
        self, app: FastAPI, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(stream, "_load_quotes", _noop_loader)

        with TestClient(app).websocket_connect("/api/v1/stream/prices?token=valid") as socket:
            socket.send_json(["not", "an", "object"])
            frame = socket.receive_json()

        assert frame["type"] == "error"
        assert "object" in frame["message"]

    def test_the_subscription_cap_is_enforced(
        self, app: FastAPI, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each symbol is a query per poll, so the list cannot be unbounded."""
        monkeypatch.setattr(stream, "_load_quotes", _noop_loader)
        too_many = [f"SYM{index}" for index in range(stream.MAX_SUBSCRIPTIONS + 1)]

        with TestClient(app).websocket_connect("/api/v1/stream/prices?token=valid") as socket:
            socket.send_json({"action": "subscribe", "symbols": too_many})
            frame = socket.receive_json()

        assert frame["type"] == "error"
        assert str(stream.MAX_SUBSCRIPTIONS) in frame["message"]


class TestQuoteDelivery:
    """What the server pushes."""

    def test_a_subscribed_symbol_receives_a_quote(
        self, app: FastAPI, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def loader(postgres: object, symbols: list[str]) -> list[stream.Quote]:
            return _quotes(*symbols)

        monkeypatch.setattr(stream, "_load_quotes", loader)

        with TestClient(app).websocket_connect("/api/v1/stream/prices?token=valid") as socket:
            socket.send_json({"action": "subscribe", "symbols": ["NVDA"]})
            socket.receive_json()  # subscription ack
            frame = socket.receive_json()

        assert frame["type"] == "quote"
        assert frame["data"]["symbol"] == "NVDA"
        assert frame["data"]["close"] == "100.50"

    def test_an_unchanged_quote_is_not_resent(
        self, app: FastAPI, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An identical bar must not be pushed twice.

        Otherwise the stream repeats itself every poll and the client has to
        deduplicate what the server already knew was unchanged.
        """

        async def loader(postgres: object, symbols: list[str]) -> list[stream.Quote]:
            return _quotes("NVDA")

        monkeypatch.setattr(stream, "_load_quotes", loader)

        with TestClient(app).websocket_connect("/api/v1/stream/prices?token=valid") as socket:
            socket.send_json({"action": "subscribe", "symbols": ["NVDA"]})
            socket.receive_json()
            socket.receive_json()  # the one quote

            # A ping round-trip proves the socket is live and idle: had the poller
            # been resending, a quote frame would arrive before the pong.
            socket.send_json({"action": "ping"})
            assert socket.receive_json() == {"type": "pong"}


class TestQuoteSerialisation:
    """The wire format."""

    def test_prices_are_strings_not_floats(self) -> None:
        """Decimals cross the wire as strings.

        A price that survived NUMERIC through the whole backend should not lose
        precision in its last two metres to a JSON float.
        """
        message = _quotes("NVDA")[0].to_message()

        assert isinstance(message["close"], str)
        assert message["close"] == "100.50"

    def test_the_provisional_flag_is_always_present(self) -> None:
        """A client must be able to tell a settled close from a mid-session one."""
        message = _quotes("NVDA")[0].to_message()

        assert "is_provisional" in message
        assert message["is_provisional"] is False

    def test_the_session_change_is_computed(self) -> None:
        message = _quotes("NVDA")[0].to_message()

        assert message["change_percent"] == "0.5000"

    def test_a_first_ever_bar_reports_no_change(self) -> None:
        quote = stream.Quote(
            symbol="NEW", close=Decimal("10"), trade_date=TODAY, is_provisional=True
        )

        assert quote.to_message()["change_percent"] is None

    def test_a_zero_previous_close_does_not_divide_by_zero(self) -> None:
        quote = stream.Quote(
            symbol="ODD",
            close=Decimal("10"),
            trade_date=TODAY,
            is_provisional=False,
            previous_close=Decimal("0"),
        )

        assert quote.to_message()["change_percent"] is None


class TestDeduplication:
    """The per-connection sent-quote tracking."""

    def test_a_repeated_quote_is_suppressed(self) -> None:
        subscription = stream.Subscription()
        quote = _quotes("NVDA")[0]

        assert subscription.changed(quote) is True
        assert subscription.changed(quote) is False

    def test_a_new_close_is_sent(self) -> None:
        subscription = stream.Subscription()
        subscription.changed(_quotes("NVDA")[0])

        updated = stream.Quote(
            symbol="NVDA", close=Decimal("101.00"), trade_date=TODAY, is_provisional=False
        )

        assert subscription.changed(updated) is True

    def test_the_same_close_on_a_new_session_is_sent(self) -> None:
        """A flat day is still a new session, and the client needs the date."""
        subscription = stream.Subscription()
        subscription.changed(_quotes("NVDA")[0])

        next_session = stream.Quote(
            symbol="NVDA",
            close=Decimal("100.50"),
            trade_date=date(2026, 7, 31),
            is_provisional=False,
        )

        assert subscription.changed(next_session) is True


class TestTokenCompatibility:
    """The stream accepts the same tokens the HTTP API issues."""

    def test_an_issued_access_token_is_the_expected_shape(self, settings: Settings) -> None:
        pair = create_token_pair(settings.security, uuid.uuid4())

        assert pair.access_token.count(".") == 2
        assert pair.expires_in == settings.security.access_token_ttl_minutes * 60


async def _noop_loader(postgres: object, symbols: list[str]) -> list[Any]:
    """Quote loader that returns nothing, for protocol-only tests."""
    return []
