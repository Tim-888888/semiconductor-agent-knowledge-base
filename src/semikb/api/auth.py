"""Demo JWT authentication with an OIDC-compatible boundary for later replacement."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from fastapi import Depends, Header, HTTPException, status

from semikb.config import Settings, get_settings
from semikb.contracts.models import ActorScope


def create_demo_token(scope: ActorScope, settings: Settings) -> str:
    expires_at = datetime.now(UTC) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": scope.user_id, "scope": scope.model_dump(), "exp": expires_at}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def get_actor_scope(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> ActorScope:
    """Resolve an actor without trusting role or scope data submitted in a request body."""

    if not authorization:
        if settings.demo_mode:
            return ActorScope()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token required.")
    prefix, _, token = authorization.partition(" ")
    if prefix.lower() != "bearer" or not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authorization header.")
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return ActorScope.model_validate(payload["scope"])
    except (jwt.InvalidTokenError, KeyError) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token.") from exc
