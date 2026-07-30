"""Password hashing and JWT issuance.

Two decisions here are security-relevant enough to state explicitly.

**Argon2id, not bcrypt.** Argon2id is memory-hard: cracking it requires RAM per
guess, which is what defeats GPU and ASIC attacks that make bcrypt's pure CPU
cost cheap to parallelise. It won the Password Hashing Competition and is OWASP's
current first recommendation. bcrypt remains acceptable, but there is no reason
to choose it for a new system.

**The token subject is the user's UUID, never the email.** An email address is
mutable and is a personal identifier; putting it in a JWT means every issued
token leaks it to anything that can read the payload -- and JWT payloads are
base64, not encrypted. It also breaks silently the moment a user changes address.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from app.core.config import SecuritySettings
from app.core.exceptions import AuthenticationError
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Argon2id parameters. These are the argon2-cffi defaults, which track OWASP's
#: guidance: ~64 MiB of memory and 3 passes per hash. Deliberately not lowered --
#: the cost is the security property, and a login taking 50ms is not a problem
#: that needs solving.
_HASHER = PasswordHasher()


class TokenType(StrEnum):
    """Distinguishes an access token from a refresh token.

    Carried inside the token and checked on decode. Without it, a refresh token
    -- which is long-lived by design -- would be accepted as an access token,
    silently turning a 30-minute credential into a 14-day one.
    """

    # Values are discriminators, not credentials.
    ACCESS = "access"
    REFRESH = "refresh"


@dataclass(frozen=True)
class TokenPair:
    """An issued access/refresh pair."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"  # noqa: S105 - the OAuth2 scheme name
    expires_in: int = 0


@dataclass(frozen=True)
class TokenClaims:
    """The verified contents of a decoded token."""

    subject: uuid.UUID
    token_type: TokenType
    issued_at: datetime
    expires_at: datetime


def hash_password(password: str) -> str:
    """Return an Argon2id hash of ``password``.

    The salt is generated per call and embedded in the returned string, so no
    separate salt column is needed and two identical passwords never share a
    hash.
    """
    return _HASHER.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    """Check ``password`` against a stored hash.

    Returns ``False`` for a mismatch and for a malformed hash alike. Callers
    must not distinguish the two in what they return to the client: "that hash
    is corrupt" is information about the account's existence.
    """
    try:
        return _HASHER.verify(hashed, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(hashed: str) -> bool:
    """Whether a stored hash was made with weaker parameters than current.

    Lets the platform upgrade a user's hash transparently on their next
    successful login, so raising the cost parameters does not leave old accounts
    permanently on the old settings.
    """
    try:
        return _HASHER.check_needs_rehash(hashed)
    except InvalidHashError:
        return True


def create_token_pair(settings: SecuritySettings, subject: uuid.UUID) -> TokenPair:
    """Issue an access and refresh token for a user."""
    access_ttl = timedelta(minutes=settings.access_token_ttl_minutes)
    return TokenPair(
        access_token=_encode(settings, subject, TokenType.ACCESS, access_ttl),
        refresh_token=_encode(
            settings,
            subject,
            TokenType.REFRESH,
            timedelta(days=settings.refresh_token_ttl_days),
        ),
        expires_in=int(access_ttl.total_seconds()),
    )


def decode_token(settings: SecuritySettings, token: str, *, expected: TokenType) -> TokenClaims:
    """Verify a token's signature, expiry and type.

    Args:
        settings: Signing key and algorithm.
        token: The encoded JWT.
        expected: The type the caller requires.

    Returns:
        The verified claims.

    Raises:
        AuthenticationError: If the signature is invalid, the token has expired,
            the payload is malformed, or the type does not match ``expected``.

    Notes:
        Every failure raises the same exception with a generic message. Telling a
        caller *why* verification failed helps an attacker far more than it helps
        a legitimate client, which only needs to know it must authenticate again.
    """
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            settings.secret_key.get_secret_value(),
            algorithms=[settings.algorithm],
            options={"require": ["exp", "iat", "sub", "type"]},
        )
    except jwt.PyJWTError as exc:
        logger.info("token_rejected", reason=type(exc).__name__)
        msg = "Could not validate credentials."
        raise AuthenticationError(msg) from exc

    if payload.get("type") != expected.value:
        # A refresh token presented as an access token is the case that matters:
        # accepting it would extend a 30-minute credential to 14 days.
        logger.info("token_rejected", reason="wrong_type", presented=payload.get("type"))
        msg = "Could not validate credentials."
        raise AuthenticationError(msg)

    try:
        subject = uuid.UUID(str(payload["sub"]))
    except (ValueError, KeyError) as exc:
        msg = "Could not validate credentials."
        raise AuthenticationError(msg) from exc

    return TokenClaims(
        subject=subject,
        token_type=expected,
        issued_at=datetime.fromtimestamp(payload["iat"], tz=UTC),
        expires_at=datetime.fromtimestamp(payload["exp"], tz=UTC),
    )


def _encode(
    settings: SecuritySettings,
    subject: uuid.UUID,
    token_type: TokenType,
    ttl: timedelta,
) -> str:
    """Sign a token for ``subject``."""
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "sub": str(subject),
            "type": token_type.value,
            "iat": int(now.timestamp()),
            "exp": int((now + ttl).timestamp()),
            # A per-token id, so a future revocation list has something to key
            # on without needing to store the whole token.
            "jti": str(uuid.uuid4()),
        },
        settings.secret_key.get_secret_value(),
        algorithm=settings.algorithm,
    )
