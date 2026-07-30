"""Tests for password hashing, tokens and the authentication service.

Security code is where a test suite earns its keep, because the failure modes are
silent: a system that accepts refresh tokens as access tokens works perfectly in
every manual check and quietly turns a 30-minute credential into a two-week one.
Most of these tests assert on things that must *not* happen.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.core.config import SecuritySettings
from app.core.exceptions import AuthenticationError, ConflictError
from app.core.security import (
    TokenType,
    create_token_pair,
    decode_token,
    hash_password,
    needs_rehash,
    verify_password,
)
from app.services.auth_service import AuthService

PASSWORD = "correct-horse-battery-staple"


def _settings(**overrides: object) -> SecuritySettings:
    values: dict[str, object] = {
        "secret_key": "test-signing-key-not-for-production",
        "access_token_ttl_minutes": 30,
        "refresh_token_ttl_days": 14,
    }
    values.update(overrides)
    return SecuritySettings(**values)  # type: ignore[arg-type]


class TestPasswordHashing:
    """Argon2id hashing."""

    def test_a_correct_password_verifies(self) -> None:
        assert verify_password(PASSWORD, hash_password(PASSWORD))

    def test_a_wrong_password_does_not(self) -> None:
        assert not verify_password("wrong", hash_password(PASSWORD))

    def test_the_hash_is_not_the_password(self) -> None:
        hashed = hash_password(PASSWORD)

        assert PASSWORD not in hashed
        assert hashed.startswith("$argon2id$")

    def test_identical_passwords_produce_different_hashes(self) -> None:
        """A per-hash salt: two users with the same password must not collide.

        Shared hashes would let an attacker who cracks one account unlock every
        other account using that password, and would reveal which users share one.
        """
        assert hash_password(PASSWORD) != hash_password(PASSWORD)

    def test_a_malformed_hash_returns_false_rather_than_raising(self) -> None:
        """A corrupt row must fail the login, not 500 the endpoint."""
        assert not verify_password(PASSWORD, "not-a-hash")

    def test_a_current_hash_does_not_need_rehashing(self) -> None:
        assert not needs_rehash(hash_password(PASSWORD))

    def test_a_malformed_hash_is_treated_as_needing_rehash(self) -> None:
        assert needs_rehash("garbage")


class TestTokens:
    """JWT issuance and verification."""

    def test_an_access_token_round_trips(self) -> None:
        settings = _settings()
        subject = uuid.uuid4()

        pair = create_token_pair(settings, subject)
        claims = decode_token(settings, pair.access_token, expected=TokenType.ACCESS)

        assert claims.subject == subject
        assert claims.token_type is TokenType.ACCESS

    def test_a_refresh_token_is_rejected_as_an_access_token(self) -> None:
        """The bug this guards is invisible in manual testing.

        Both tokens verify against the same key, so without the type claim a
        refresh token would authorise API calls for its full two-week lifetime.
        """
        settings = _settings()
        pair = create_token_pair(settings, uuid.uuid4())

        with pytest.raises(AuthenticationError):
            decode_token(settings, pair.refresh_token, expected=TokenType.ACCESS)

    def test_an_access_token_is_rejected_as_a_refresh_token(self) -> None:
        settings = _settings()
        pair = create_token_pair(settings, uuid.uuid4())

        with pytest.raises(AuthenticationError):
            decode_token(settings, pair.access_token, expected=TokenType.REFRESH)

    def test_a_token_signed_with_another_key_is_rejected(self) -> None:
        pair = create_token_pair(_settings(secret_key="key-one"), uuid.uuid4())

        with pytest.raises(AuthenticationError):
            decode_token(
                _settings(secret_key="key-two"), pair.access_token, expected=TokenType.ACCESS
            )

    def test_an_expired_token_is_rejected(self) -> None:
        settings = _settings()
        expired = jwt.encode(
            {
                "sub": str(uuid.uuid4()),
                "type": "access",
                "iat": int((datetime.now(UTC) - timedelta(hours=2)).timestamp()),
                "exp": int((datetime.now(UTC) - timedelta(hours=1)).timestamp()),
            },
            settings.secret_key.get_secret_value(),
            algorithm=settings.algorithm,
        )

        with pytest.raises(AuthenticationError):
            decode_token(settings, expired, expected=TokenType.ACCESS)

    def test_a_token_without_a_type_claim_is_rejected(self) -> None:
        """`require` must actually be enforced, or an old token shape slips through."""
        settings = _settings()
        untyped = jwt.encode(
            {
                "sub": str(uuid.uuid4()),
                "iat": int(datetime.now(UTC).timestamp()),
                "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
            },
            settings.secret_key.get_secret_value(),
            algorithm=settings.algorithm,
        )

        with pytest.raises(AuthenticationError):
            decode_token(settings, untyped, expected=TokenType.ACCESS)

    def test_an_unsigned_token_is_rejected(self) -> None:
        """The `alg: none` attack: a payload with no signature at all."""
        settings = _settings()
        unsigned = jwt.encode(
            {
                "sub": str(uuid.uuid4()),
                "type": "access",
                "iat": int(datetime.now(UTC).timestamp()),
                "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
            },
            key="",
            algorithm="none",
        )

        with pytest.raises(AuthenticationError):
            decode_token(settings, unsigned, expected=TokenType.ACCESS)

    def test_garbage_is_rejected(self) -> None:
        with pytest.raises(AuthenticationError):
            decode_token(_settings(), "not.a.token", expected=TokenType.ACCESS)

    def test_the_payload_carries_no_email(self) -> None:
        """JWT payloads are base64, not encrypted: anything in them is public."""
        settings = _settings()
        pair = create_token_pair(settings, uuid.uuid4())

        payload = jwt.decode(
            pair.access_token,
            settings.secret_key.get_secret_value(),
            algorithms=[settings.algorithm],
        )

        assert set(payload) == {"sub", "type", "iat", "exp", "jti"}
        assert "@" not in payload["sub"]

    def test_the_access_token_expires_sooner_than_the_refresh_token(self) -> None:
        settings = _settings()
        pair = create_token_pair(settings, uuid.uuid4())

        access = decode_token(settings, pair.access_token, expected=TokenType.ACCESS)
        refresh = decode_token(settings, pair.refresh_token, expected=TokenType.REFRESH)

        assert access.expires_at < refresh.expires_at

    def test_two_tokens_for_one_user_are_distinguishable(self) -> None:
        """The `jti` gives a future revocation list something to key on."""
        settings = _settings()
        subject = uuid.uuid4()

        first = create_token_pair(settings, subject).access_token
        second = create_token_pair(settings, subject).access_token
        key = settings.secret_key.get_secret_value()

        assert (
            jwt.decode(first, key, algorithms=[settings.algorithm])["jti"]
            != jwt.decode(second, key, algorithms=[settings.algorithm])["jti"]
        )


# --- Auth service ----------------------------------------------------------


class FakeUser:
    """Stands in for the User model."""

    def __init__(self, email: str, password: str, *, is_active: bool = True) -> None:
        self.id = uuid.uuid4()
        self.email = email.lower()
        self.hashed_password = hash_password(password)
        self.full_name: str | None = None
        self.is_active = is_active
        self.last_login_at: datetime | None = None


class FakeUserRepository:
    """In-memory account store."""

    def __init__(self, users: list[FakeUser] | None = None) -> None:
        self.users = users or []
        self.added: list[object] = []

    async def email_exists(self, email: str) -> bool:
        return any(user.email == email.strip().lower() for user in self.users)

    async def get_by_email(self, email: str) -> FakeUser | None:
        wanted = email.strip().lower()
        return next((user for user in self.users if user.email == wanted), None)

    async def get(self, user_id: uuid.UUID) -> FakeUser | None:
        return next((user for user in self.users if user.id == user_id), None)

    def add(self, user: object) -> object:
        self.added.append(user)
        self.users.append(user)  # type: ignore[arg-type]
        return user

    async def flush(self) -> None:
        return None


class FakeWatchlistRepository:
    """Records the watchlists seeded on registration."""

    def __init__(self) -> None:
        self.added: list[object] = []

    def add(self, watchlist: object) -> object:
        self.added.append(watchlist)
        return watchlist

    async def flush(self) -> None:
        return None


def _service(
    users: FakeUserRepository | None = None,
) -> tuple[AuthService, FakeUserRepository, FakeWatchlistRepository]:
    user_repo = users or FakeUserRepository()
    watchlists = FakeWatchlistRepository()
    service = AuthService(
        users=user_repo,  # type: ignore[arg-type]
        watchlists=watchlists,  # type: ignore[arg-type]
        settings=_settings(),
    )
    return service, user_repo, watchlists


class TestRegistration:
    """Account creation."""

    async def test_a_new_account_is_created_with_tokens(self) -> None:
        service, users, _ = _service()

        result = await service.register(email="a@example.test", password=PASSWORD)

        assert result.user.email == "a@example.test"
        assert result.tokens.access_token
        assert result.tokens.refresh_token
        assert len(users.added) == 1

    async def test_the_password_is_not_stored_in_the_clear(self) -> None:
        service, _, _ = _service()

        result = await service.register(email="a@example.test", password=PASSWORD)

        assert result.user.hashed_password != PASSWORD
        assert verify_password(PASSWORD, result.user.hashed_password)

    async def test_a_duplicate_email_is_rejected(self) -> None:
        service, _, _ = _service(FakeUserRepository([FakeUser("taken@example.test", PASSWORD)]))

        with pytest.raises(ConflictError):
            await service.register(email="taken@example.test", password=PASSWORD)

    async def test_duplicate_detection_is_case_insensitive(self) -> None:
        service, _, _ = _service(FakeUserRepository([FakeUser("Taken@Example.test", PASSWORD)]))

        with pytest.raises(ConflictError):
            await service.register(email="taken@example.TEST", password=PASSWORD)

    async def test_a_default_watchlist_is_seeded(self) -> None:
        """A new account should not land on an empty dashboard."""
        service, _, watchlists = _service()

        await service.register(email="a@example.test", password=PASSWORD)

        assert len(watchlists.added) == 1


class TestAuthentication:
    """Login, and the ways it must not leak information."""

    async def test_correct_credentials_succeed(self) -> None:
        user = FakeUser("a@example.test", PASSWORD)
        service, _, _ = _service(FakeUserRepository([user]))

        result = await service.authenticate(email="a@example.test", password=PASSWORD)

        assert result.user.id == user.id
        assert result.tokens.access_token

    async def test_a_wrong_password_is_rejected(self) -> None:
        service, _, _ = _service(FakeUserRepository([FakeUser("a@example.test", PASSWORD)]))

        with pytest.raises(AuthenticationError):
            await service.authenticate(email="a@example.test", password="wrong")

    async def test_an_unknown_account_and_a_wrong_password_are_indistinguishable(
        self,
    ) -> None:
        """An enumeration oracle: differing messages reveal which emails exist."""
        service, _, _ = _service(FakeUserRepository([FakeUser("a@example.test", PASSWORD)]))

        with pytest.raises(AuthenticationError) as wrong_password:
            await service.authenticate(email="a@example.test", password="wrong")
        with pytest.raises(AuthenticationError) as unknown_account:
            await service.authenticate(email="nobody@example.test", password=PASSWORD)

        assert wrong_password.value.message == unknown_account.value.message
        assert wrong_password.value.status_code == unknown_account.value.status_code

    async def test_a_deactivated_account_is_rejected_with_the_same_message(self) -> None:
        """Otherwise "your account is disabled" confirms the address is registered."""
        service, _, _ = _service(
            FakeUserRepository([FakeUser("a@example.test", PASSWORD, is_active=False)])
        )

        with pytest.raises(AuthenticationError) as error:
            await service.authenticate(email="a@example.test", password=PASSWORD)

        assert error.value.message == "Incorrect email or password."

    async def test_a_missing_account_still_verifies_a_hash(self) -> None:
        """Timing equalisation.

        Returning early on an unknown address makes a miss microseconds and a hit
        tens of milliseconds, which is measurable over a network and is itself an
        enumeration oracle. The dummy verification closes that gap.
        """
        service, _, _ = _service(FakeUserRepository([FakeUser("a@example.test", PASSWORD)]))

        started = time.perf_counter()
        with pytest.raises(AuthenticationError):
            await service.authenticate(email="nobody@example.test", password=PASSWORD)
        missing = time.perf_counter() - started

        # Argon2id at default cost is tens of milliseconds; a short-circuit
        # return would be three orders of magnitude faster than this bound.
        assert missing > 0.005

    async def test_login_records_the_timestamp(self) -> None:
        user = FakeUser("a@example.test", PASSWORD)
        service, _, _ = _service(FakeUserRepository([user]))

        await service.authenticate(email="a@example.test", password=PASSWORD)

        assert user.last_login_at is not None

    async def test_the_email_lookup_is_case_insensitive(self) -> None:
        service, _, _ = _service(FakeUserRepository([FakeUser("a@example.test", PASSWORD)]))

        result = await service.authenticate(email="A@Example.TEST", password=PASSWORD)

        assert result.user.email == "a@example.test"


class TestRefresh:
    """Token rotation."""

    async def test_a_valid_refresh_token_yields_a_new_pair(self) -> None:
        user = FakeUser("a@example.test", PASSWORD)
        service, _, _ = _service(FakeUserRepository([user]))
        original = create_token_pair(_settings(), user.id)

        result = await service.refresh(original.refresh_token)

        assert result.user.id == user.id
        assert result.tokens.access_token

    async def test_an_access_token_cannot_be_used_to_refresh(self) -> None:
        user = FakeUser("a@example.test", PASSWORD)
        service, _, _ = _service(FakeUserRepository([user]))
        pair = create_token_pair(_settings(), user.id)

        with pytest.raises(AuthenticationError):
            await service.refresh(pair.access_token)

    async def test_a_deactivated_account_cannot_refresh(self) -> None:
        """The account is re-read, so deactivation takes effect immediately.

        Trusting the token's claims instead would let a disabled user keep
        minting access tokens for up to two weeks.
        """
        user = FakeUser("a@example.test", PASSWORD, is_active=False)
        service, _, _ = _service(FakeUserRepository([user]))
        pair = create_token_pair(_settings(), user.id)

        with pytest.raises(AuthenticationError):
            await service.refresh(pair.refresh_token)

    async def test_a_deleted_account_cannot_refresh(self) -> None:
        service, _, _ = _service(FakeUserRepository([]))
        pair = create_token_pair(_settings(), uuid.uuid4())

        with pytest.raises(AuthenticationError):
            await service.refresh(pair.refresh_token)
