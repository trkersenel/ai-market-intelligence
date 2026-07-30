"""WebSocket price stream.

Pushes quote updates to subscribed clients instead of making them poll. A
dashboard watching fourteen tickers would otherwise issue fourteen requests every
few seconds, most of which return the same bar it already has.

Two things are worth being explicit about.

**What this streams.** The platform's data source is end-of-day bars, so the
"live" price is the latest stored close, provisional or not. Broadcasting it as
though it were a tick feed would be a lie the frontend could not detect, so every
message carries ``is_provisional`` and the session it belongs to. Wiring a real
tick source later means replacing the poller, not the protocol.

**Authentication.** Browsers cannot set headers on a WebSocket handshake, so the
token arrives as a query parameter -- the standard workaround, and the reason
access tokens are short-lived. URLs reach server logs and browser history far more
readily than headers do.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Annotated, Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from app.api.deps import get_current_user_from_token
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.db.postgres import PostgresDatabase
from app.repositories.company import TickerRepository
from app.repositories.price import DailyPriceRepository

logger = get_logger(__name__)

router = APIRouter(tags=["stream"])

#: Seconds between database polls. Matched to the data: end-of-day bars change
#: at most a few times an hour during ingestion, so polling faster would burn
#: queries to send messages identical to the last one.
POLL_INTERVAL_SECONDS = 15.0

#: Symbols one connection may follow. Bounded because each is a query per poll,
#: and an unbounded subscription list is a trivial way to make one socket
#: expensive for everybody.
MAX_SUBSCRIPTIONS = 50


@dataclass
class Quote:
    """One symbol's latest stored bar."""

    symbol: str
    close: Decimal
    trade_date: date
    is_provisional: bool
    previous_close: Decimal | None = None

    def to_message(self) -> dict[str, Any]:
        """Serialise for the wire.

        Decimals become strings rather than floats: JSON numbers are IEEE
        doubles, and a price that survived NUMERIC all the way through the
        backend should not lose precision in its last two metres.
        """
        change = None
        if self.previous_close is not None and self.previous_close != 0:
            change = (self.close - self.previous_close) / self.previous_close * 100

        return {
            "symbol": self.symbol,
            "close": str(self.close),
            "trade_date": self.trade_date.isoformat(),
            "change_percent": None if change is None else str(round(change, 4)),
            # Never omitted. A client must be able to tell a settled close from a
            # mid-session snapshot, or it will chart them identically.
            "is_provisional": self.is_provisional,
        }


@dataclass
class Subscription:
    """What one connection is following, and what it has already been told."""

    symbols: set[str] = field(default_factory=set)
    #: Last value sent per symbol, so unchanged bars are not resent. Without
    #: this the stream would emit an identical payload every poll and the client
    #: would have to deduplicate.
    last_sent: dict[str, tuple[str, str]] = field(default_factory=dict)

    def changed(self, quote: Quote) -> bool:
        """Whether this quote differs from the last one sent for its symbol."""
        signature = (str(quote.close), quote.trade_date.isoformat())
        if self.last_sent.get(quote.symbol) == signature:
            return False
        self.last_sent[quote.symbol] = signature
        return True


@router.websocket("/prices")
async def stream_prices(
    websocket: WebSocket,
    token: Annotated[str | None, Query(description="JWT access token.")] = None,
) -> None:
    """Stream quote updates for the symbols a client subscribes to.

    Protocol, after the handshake:

    - client sends ``{"action": "subscribe", "symbols": ["NVDA", "MU"]}``
    - client sends ``{"action": "unsubscribe", "symbols": ["MU"]}``
    - client sends ``{"action": "ping"}`` and receives ``{"type": "pong"}``
    - server sends ``{"type": "quote", "data": {...}}`` when a value changes
    - server sends ``{"type": "error", "message": "..."}`` for a bad frame

    Closes with 1008 if the token is missing or invalid.
    """
    postgres: PostgresDatabase | None = getattr(websocket.app.state, "postgres", None)
    if postgres is None:  # pragma: no cover - lifespan wiring bug
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    try:
        user = await get_current_user_from_token(token, websocket.app.state, postgres)
    except AppError as exc:
        # Accept then close, rather than rejecting the handshake: browsers report
        # a rejected handshake as an opaque network error, so the client cannot
        # tell "your token expired" from "the server is down".
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": exc.message})
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    subscription = Subscription()
    logger.info("stream_connected", user_id=str(user.id))

    pump = asyncio.create_task(_broadcast(websocket, postgres, subscription))
    try:
        await _receive(websocket, subscription)
    except WebSocketDisconnect:
        logger.info("stream_disconnected", user_id=str(user.id))
    finally:
        pump.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pump


async def _receive(websocket: WebSocket, subscription: Subscription) -> None:
    """Handle client frames until it disconnects."""
    while True:
        try:
            frame = await websocket.receive_json()
        except (ValueError, TypeError):
            await websocket.send_json({"type": "error", "message": "Malformed JSON frame."})
            continue

        if not isinstance(frame, dict):
            await websocket.send_json({"type": "error", "message": "Frame must be an object."})
            continue

        action = frame.get("action")
        if action == "ping":
            await websocket.send_json({"type": "pong"})
            continue

        symbols = frame.get("symbols")
        if action not in {"subscribe", "unsubscribe"} or not isinstance(symbols, list):
            await websocket.send_json(
                {"type": "error", "message": "Expected subscribe/unsubscribe with symbols."}
            )
            continue

        requested = {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
        if action == "subscribe":
            if len(subscription.symbols | requested) > MAX_SUBSCRIPTIONS:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"At most {MAX_SUBSCRIPTIONS} symbols per connection.",
                    }
                )
                continue
            subscription.symbols |= requested
        else:
            subscription.symbols -= requested
            for symbol in requested:
                subscription.last_sent.pop(symbol, None)

        await websocket.send_json({"type": "subscribed", "symbols": sorted(subscription.symbols)})


async def _broadcast(
    websocket: WebSocket, postgres: PostgresDatabase, subscription: Subscription
) -> None:
    """Poll for changed quotes and push them to this connection."""
    while True:
        try:
            if subscription.symbols:
                quotes = await _load_quotes(postgres, sorted(subscription.symbols))
                for quote in quotes:
                    if subscription.changed(quote):
                        await websocket.send_json(
                            {
                                "type": "quote",
                                "data": quote.to_message(),
                                "sent_at": datetime.now(UTC).isoformat(),
                            }
                        )
        except WebSocketDisconnect:
            return
        except Exception:
            # One failed poll must not kill a long-lived connection: the next
            # tick may well succeed, and dropping the socket would make a
            # transient database blip look like a client-side bug.
            logger.warning("stream_poll_failed", exc_info=True)

        await asyncio.sleep(POLL_INTERVAL_SECONDS)


async def _load_quotes(postgres: PostgresDatabase, symbols: list[str]) -> list[Quote]:
    """Read the latest two bars for each subscribed symbol.

    Opens a short-lived session per poll rather than holding one for the
    connection's lifetime. A socket idle for an hour must not pin a pooled
    connection for that hour.
    """
    quotes: list[Quote] = []
    async with postgres.session() as session:
        tickers = TickerRepository(session)
        prices = DailyPriceRepository(session)

        for listing in await tickers.get_many_by_symbols(symbols):
            recent = await prices.get_recent(listing.id, sessions=2)
            if not recent:
                continue
            latest = recent[-1]
            quotes.append(
                Quote(
                    symbol=listing.symbol,
                    close=latest.close,
                    trade_date=latest.trade_date,
                    is_provisional=latest.is_provisional,
                    previous_close=recent[-2].close if len(recent) > 1 else None,
                )
            )
    return quotes
