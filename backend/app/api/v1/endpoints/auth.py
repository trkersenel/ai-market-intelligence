"""Authentication endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import AuthServiceDep, CurrentUserDep
from app.schemas.auth import (
    AuthResponse,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
    user_response,
)

router = APIRouter(tags=["auth"])


@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account",
    description=(
        "Creates the account, seeds a default watchlist, and returns tokens so "
        "the client does not have to log in immediately afterwards."
    ),
    responses={status.HTTP_409_CONFLICT: {"description": "Email already registered."}},
)
async def register(payload: RegisterRequest, service: AuthServiceDep) -> AuthResponse:
    """Register a new account.

    Raises:
        ConflictError: If the email address is already in use.
    """
    result = await service.register(
        email=payload.email,
        password=payload.password,
        full_name=payload.full_name,
    )
    return AuthResponse(
        user=user_response(result.user),
        tokens=TokenResponse(**vars(result.tokens)),
    )


@router.post(
    "/login",
    response_model=AuthResponse,
    summary="Exchange credentials for tokens",
    description=(
        "Returns the same error for an unknown address, a wrong password and a "
        "deactivated account, so the response cannot be used to discover which "
        "addresses are registered."
    ),
    responses={status.HTTP_401_UNAUTHORIZED: {"description": "Invalid credentials."}},
)
async def login(payload: LoginRequest, service: AuthServiceDep) -> AuthResponse:
    """Authenticate and issue tokens.

    Raises:
        AuthenticationError: On any credential failure.
    """
    result = await service.authenticate(email=payload.email, password=payload.password)
    return AuthResponse(
        user=user_response(result.user),
        tokens=TokenResponse(**vars(result.tokens)),
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Exchange a refresh token for a new pair",
    description=(
        "The account is re-read on every refresh, so deactivating a user takes "
        "effect immediately rather than when their refresh token expires."
    ),
    responses={status.HTTP_401_UNAUTHORIZED: {"description": "Invalid refresh token."}},
)
async def refresh(payload: RefreshRequest, service: AuthServiceDep) -> TokenResponse:
    """Rotate tokens.

    Raises:
        AuthenticationError: If the token is invalid, expired or of the wrong type.
    """
    result = await service.refresh(payload.refresh_token)
    return TokenResponse(**vars(result.tokens))


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get the authenticated account",
    responses={status.HTTP_401_UNAUTHORIZED: {"description": "Missing or invalid token."}},
)
async def me(current_user: CurrentUserDep) -> UserResponse:
    """Return the account the presented token belongs to."""
    return user_response(current_user)
