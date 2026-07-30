"""Registration, login and token refresh.

The recurring theme is that failures must not be distinguishable. A login
endpoint that says "no such account" for one address and "wrong password" for
another is an account enumeration oracle: an attacker learns which of a list of
harvested emails are registered here, which is exactly what they need before
credential stuffing.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from app.core.config import SecuritySettings
from app.core.exceptions import AuthenticationError, ConflictError
from app.core.logging import get_logger
from app.core.security import (
    TokenPair,
    TokenType,
    create_token_pair,
    decode_token,
    hash_password,
    needs_rehash,
    verify_password,
)
from app.models.user import User
from app.repositories.user import UserRepository, WatchlistRepository

logger = get_logger(__name__)

#: The watchlist every new account starts with. An empty dashboard is a worse
#: first impression than a populated one, and this gives the frontend something
#: real to render immediately.
DEFAULT_WATCHLIST_NAME = "My Watchlist"
DEFAULT_WATCHLIST_SYMBOLS = ("NVDA", "AMD", "MU", "TSM")


@dataclass(frozen=True)
class AuthenticatedUser:
    """A user together with freshly issued tokens."""

    user: User
    tokens: TokenPair


class AuthService:
    """Owns account creation and credential verification."""

    def __init__(
        self,
        *,
        users: UserRepository,
        watchlists: WatchlistRepository,
        settings: SecuritySettings,
    ) -> None:
        """Wire the service to its collaborators.

        Args:
            users: Account repository.
            watchlists: Used to seed a starter watchlist on registration.
            settings: Token signing configuration.
        """
        self._users = users
        self._watchlists = watchlists
        self._settings = settings

    async def register(
        self, *, email: str, password: str, full_name: str | None = None
    ) -> AuthenticatedUser:
        """Create an account and return it with tokens.

        Raises:
            ConflictError: If the address is already registered.

        Notes:
            Registration *does* reveal whether an address is taken, unavoidably:
            the user has to be told they cannot have that address. Login does
            not, which is where the enumeration risk actually lives.
        """
        if await self._users.email_exists(email):
            msg = "An account with this email address already exists."
            raise ConflictError(msg, details={"field": "email"})

        user = User(
            email=email,
            hashed_password=hash_password(password),
            full_name=full_name,
        )
        self._users.add(user)
        await self._users.flush()

        await self._seed_default_watchlist(user.id)

        logger.info("user_registered", user_id=str(user.id))
        return AuthenticatedUser(user=user, tokens=create_token_pair(self._settings, user.id))

    async def authenticate(self, *, email: str, password: str) -> AuthenticatedUser:
        """Verify credentials and issue tokens.

        Raises:
            AuthenticationError: For an unknown address, a wrong password, or a
                deactivated account -- all with the same message, so the response
                cannot be used to enumerate accounts.

        Notes:
            The password is verified against a dummy hash when the account does
            not exist. Skipping it would make a miss return in microseconds and a
            hit in ~50ms, and that timing difference is itself an enumeration
            oracle.
        """
        user = await self._users.get_by_email(email)

        if user is None:
            _burn_time(password)
            raise AuthenticationError(_INVALID_CREDENTIALS)

        if not verify_password(password, user.hashed_password):
            logger.info("login_failed", user_id=str(user.id), reason="bad_password")
            raise AuthenticationError(_INVALID_CREDENTIALS)

        if not user.is_active:
            logger.info("login_failed", user_id=str(user.id), reason="inactive")
            raise AuthenticationError(_INVALID_CREDENTIALS)

        # Transparently upgrade the stored hash if the cost parameters have been
        # raised since it was written, so old accounts do not stay on old
        # settings forever.
        if needs_rehash(user.hashed_password):
            user.hashed_password = hash_password(password)
            logger.info("password_hash_upgraded", user_id=str(user.id))

        user.last_login_at = datetime.now(UTC)
        logger.info("login_succeeded", user_id=str(user.id))
        return AuthenticatedUser(user=user, tokens=create_token_pair(self._settings, user.id))

    async def refresh(self, refresh_token: str) -> AuthenticatedUser:
        """Exchange a valid refresh token for a new pair.

        Raises:
            AuthenticationError: If the token is invalid, expired, of the wrong
                type, or belongs to an account that no longer exists or is
                deactivated.

        Notes:
            The account is re-read on every refresh rather than trusted from the
            token. Otherwise a deactivated user would keep minting valid access
            tokens until their refresh token expired, up to two weeks later.
        """
        claims = decode_token(self._settings, refresh_token, expected=TokenType.REFRESH)

        user = await self._users.get(claims.subject)
        if user is None or not user.is_active:
            logger.info("refresh_rejected", user_id=str(claims.subject))
            raise AuthenticationError(_INVALID_CREDENTIALS)

        return AuthenticatedUser(user=user, tokens=create_token_pair(self._settings, user.id))

    async def _seed_default_watchlist(self, user_id: uuid.UUID) -> None:
        """Create the starter watchlist for a new account."""
        from app.models.user import Watchlist  # noqa: PLC0415 - avoids a cycle

        watchlist = Watchlist(
            user_id=user_id,
            name=DEFAULT_WATCHLIST_NAME,
            description="Tracked by default. Rename or delete it freely.",
            is_default=True,
        )
        self._watchlists.add(watchlist)
        await self._watchlists.flush()


#: One message for every credential failure.
_INVALID_CREDENTIALS = "Incorrect email or password."

#: A real Argon2id hash of a throwaway value, used to spend the same CPU time on
#: a missing account as on a real one. Computed once at import.
_DUMMY_HASH = hash_password("timing-equalisation-placeholder")


def _burn_time(password: str) -> None:
    """Spend password-verification time on an account that does not exist."""
    verify_password(password, _DUMMY_HASH)
