"""Shared, credential-safe resilience primitives for external providers."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import httpx
from pydantic import BaseModel, Field


class ProviderFailureKind(StrEnum):
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    UPSTREAM_5XX = "upstream_5xx"
    TRANSPORT = "transport"
    INVALID_RESPONSE = "invalid_response"
    REJECTED = "rejected"
    CONFIGURATION = "configuration"
    STREAM_INTERRUPTED = "stream_interrupted"


class ProviderAttemptAudit(BaseModel):
    """Safe attempt evidence; request bodies, endpoints, and credentials are excluded."""

    schema_version: Literal["semikb-provider-attempt-v1"] = "semikb-provider-attempt-v1"
    provider: str = Field(min_length=1, max_length=64)
    operation: str = Field(min_length=1, max_length=96)
    attempt: int = Field(ge=1, le=10)
    max_attempts: int = Field(ge=1, le=10)
    outcome: Literal["succeeded", "retrying", "failed"]
    failure_kind: ProviderFailureKind | None = None
    status_code: int | None = Field(default=None, ge=100, le=599)
    retryable: bool = False
    retry_after_seconds: float | None = Field(default=None, ge=0, le=60)
    latency_ms: float = Field(ge=0)


@dataclass(frozen=True, slots=True)
class ProviderRetryPolicy:
    max_attempts: int = 2
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 2.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_attempts <= 10:
            raise ValueError("max_attempts must be between 1 and 10")
        if self.backoff_base_seconds < 0 or self.backoff_max_seconds < 0:
            raise ValueError("backoff values must not be negative")

    def delay(self, attempt: int, retry_after_seconds: float | None = None) -> float:
        exponential = self.backoff_base_seconds * (2 ** max(attempt - 1, 0))
        requested = retry_after_seconds if retry_after_seconds is not None else exponential
        return min(max(requested, 0.0), self.backoff_max_seconds)


class ProviderCallFailure(RuntimeError):
    """Internal classified failure without upstream response content."""

    def __init__(
        self,
        failure_kind: ProviderFailureKind,
        safe_message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(safe_message)
        self.failure_kind = failure_kind
        self.safe_message = safe_message
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds


class ProviderRetriesExhausted(RuntimeError):
    """Carries safe attempt evidence when an operation cannot complete."""

    def __init__(
        self,
        provider: str,
        operation: str,
        failure: ProviderCallFailure,
        attempts: tuple[ProviderAttemptAudit, ...],
    ) -> None:
        super().__init__(f"Provider {provider!r} operation {operation!r} failed.")
        self.provider = provider
        self.operation = operation
        self.failure = failure
        self.attempts = attempts


@dataclass(frozen=True, slots=True)
class ProviderCallResult[T]:
    value: T
    attempts: tuple[ProviderAttemptAudit, ...]


def failure_from_response(response: httpx.Response, safe_message: str) -> ProviderCallFailure:
    status = response.status_code
    if status == 429:
        return ProviderCallFailure(
            ProviderFailureKind.RATE_LIMITED,
            safe_message,
            status_code=status,
            retryable=True,
            retry_after_seconds=_retry_after_seconds(response),
        )
    if status >= 500:
        return ProviderCallFailure(
            ProviderFailureKind.UPSTREAM_5XX,
            safe_message,
            status_code=status,
            retryable=True,
            retry_after_seconds=_retry_after_seconds(response),
        )
    return ProviderCallFailure(
        ProviderFailureKind.REJECTED,
        safe_message,
        status_code=status,
        retryable=False,
    )


def failure_from_exception(exc: Exception, safe_message: str) -> ProviderCallFailure:
    if isinstance(exc, ProviderCallFailure):
        return exc
    if isinstance(exc, (httpx.TimeoutException, TimeoutError, asyncio.TimeoutError)):
        return ProviderCallFailure(
            ProviderFailureKind.TIMEOUT,
            safe_message,
            retryable=True,
        )
    if isinstance(exc, httpx.HTTPError):
        return ProviderCallFailure(
            ProviderFailureKind.TRANSPORT,
            safe_message,
            retryable=True,
        )
    return ProviderCallFailure(
        ProviderFailureKind.TRANSPORT,
        safe_message,
        retryable=True,
    )


def run_with_retry[T](
    provider: str,
    operation: str,
    policy: ProviderRetryPolicy,
    call: Callable[[], T],
    *,
    sleeper: Callable[[float], None] = time.sleep,
) -> ProviderCallResult[T]:
    attempts: list[ProviderAttemptAudit] = []
    for attempt in range(1, policy.max_attempts + 1):
        started = time.perf_counter()
        try:
            value = call()
        except Exception as exc:
            failure = failure_from_exception(exc, "The provider request failed.")
            will_retry = failure.retryable and attempt < policy.max_attempts
            delay = policy.delay(attempt, failure.retry_after_seconds) if will_retry else None
            attempts.append(
                _attempt(
                    provider,
                    operation,
                    attempt,
                    policy.max_attempts,
                    "retrying" if will_retry else "failed",
                    started,
                    failure=failure,
                    retry_after_seconds=delay,
                )
            )
            if not will_retry:
                raise ProviderRetriesExhausted(
                    provider,
                    operation,
                    failure,
                    tuple(attempts),
                ) from exc
            sleeper(delay or 0.0)
        else:
            attempts.append(
                _attempt(
                    provider,
                    operation,
                    attempt,
                    policy.max_attempts,
                    "succeeded",
                    started,
                )
            )
            return ProviderCallResult(value=value, attempts=tuple(attempts))
    raise AssertionError("Provider retry loop terminated unexpectedly.")


async def run_with_retry_async[T](
    provider: str,
    operation: str,
    policy: ProviderRetryPolicy,
    call: Callable[[], Awaitable[T]],
    *,
    sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> ProviderCallResult[T]:
    attempts: list[ProviderAttemptAudit] = []
    for attempt in range(1, policy.max_attempts + 1):
        started = time.perf_counter()
        try:
            value = await call()
        except Exception as exc:
            failure = failure_from_exception(exc, "The provider request failed.")
            will_retry = failure.retryable and attempt < policy.max_attempts
            delay = policy.delay(attempt, failure.retry_after_seconds) if will_retry else None
            attempts.append(
                _attempt(
                    provider,
                    operation,
                    attempt,
                    policy.max_attempts,
                    "retrying" if will_retry else "failed",
                    started,
                    failure=failure,
                    retry_after_seconds=delay,
                )
            )
            if not will_retry:
                raise ProviderRetriesExhausted(
                    provider,
                    operation,
                    failure,
                    tuple(attempts),
                ) from exc
            await sleeper(delay or 0.0)
        else:
            attempts.append(
                _attempt(
                    provider,
                    operation,
                    attempt,
                    policy.max_attempts,
                    "succeeded",
                    started,
                )
            )
            return ProviderCallResult(value=value, attempts=tuple(attempts))
    raise AssertionError("Provider retry loop terminated unexpectedly.")


def invalid_response(safe_message: str) -> ProviderCallFailure:
    return ProviderCallFailure(
        ProviderFailureKind.INVALID_RESPONSE,
        safe_message,
        retryable=False,
    )


def _attempt(
    provider: str,
    operation: str,
    attempt: int,
    max_attempts: int,
    outcome: Literal["succeeded", "retrying", "failed"],
    started: float,
    *,
    failure: ProviderCallFailure | None = None,
    retry_after_seconds: float | None = None,
) -> ProviderAttemptAudit:
    return ProviderAttemptAudit(
        provider=provider,
        operation=operation,
        attempt=attempt,
        max_attempts=max_attempts,
        outcome=outcome,
        failure_kind=failure.failure_kind if failure else None,
        status_code=failure.status_code if failure else None,
        retryable=failure.retryable if failure else False,
        retry_after_seconds=retry_after_seconds,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after", "").strip()
    if not raw:
        return None
    try:
        return min(max(float(raw), 0.0), 60.0)
    except ValueError:
        return None
