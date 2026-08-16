"""OpenAI-compatible LLM gateway with explicit primary/fallback routing."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import httpx

from semikb.config import Settings
from semikb_provider_resilience import (
    ProviderAttemptAudit,
    ProviderCallFailure,
    ProviderFailureKind,
    ProviderRetriesExhausted,
    ProviderRetryPolicy,
    failure_from_exception,
    run_with_retry,
    run_with_retry_async,
)


class LLMConfigurationError(ValueError):
    """Raised when a selected provider is missing required configuration."""


class LLMProviderError(RuntimeError):
    """Provider failure without credentials or endpoint details in the message."""

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status_code: int | None = None,
        provider_attempts: tuple[ProviderAttemptAudit, ...] = (),
    ) -> None:
        super().__init__(f"LLM provider '{provider}' failed: {message}")
        self.provider = provider
        self.safe_message = message
        self.status_code = status_code
        self.content_started = False
        self.provider_attempts = provider_attempts


@dataclass(frozen=True, slots=True)
class LLMProviderConfig:
    provider: str
    base_url: str
    model: str
    api_key: str = field(repr=False)
    max_tokens_field: str = "max_tokens"
    reasoning_effort: str | None = None
    verbosity: str | None = None


@dataclass(frozen=True, slots=True)
class LLMCompletion:
    content: str
    provider: str
    requested_model: str
    reported_model: str
    fallback_used: bool
    attempted_providers: tuple[str, ...]
    usage: dict[str, Any]
    provider_attempts: tuple[ProviderAttemptAudit, ...] = ()


@dataclass(frozen=True, slots=True)
class LLMStreamProbe:
    """Credential-safe evidence that a Provider emitted OpenAI-compatible deltas."""

    provider: str
    requested_model: str
    reported_model: str
    content: str = field(repr=False)
    content_sha256: str
    content_length: int
    event_count: int
    content_delta_count: int
    reasoning_delta_count: int
    first_event_ms: float
    first_content_delta_ms: float
    total_ms: float
    finish_reason: str | None
    done_received: bool
    termination: str
    usage: dict[str, Any]


def resolve_provider_config(settings: Settings, provider: str) -> LLMProviderConfig:
    """Resolve provider settings while retaining the former generic Qwen variables."""

    normalized = provider.strip().lower()
    if normalized == "closeai":
        config = LLMProviderConfig(
            provider="closeai",
            base_url=settings.closeai_base_url,
            api_key=settings.closeai_api_key,
            model=settings.closeai_model,
            max_tokens_field="max_completion_tokens",
            reasoning_effort=settings.closeai_reasoning_effort,
            verbosity=settings.closeai_verbosity,
        )
    elif normalized == "qwen":
        config = LLMProviderConfig(
            provider="qwen",
            base_url=settings.qwen_api_base_url or settings.llm_api_base_url,
            api_key=settings.qwen_api_key or settings.llm_api_key,
            model=settings.qwen_model or settings.llm_model,
        )
    else:
        raise LLMConfigurationError(f"Unsupported LLM provider: {provider}")

    missing = [
        name
        for name, value in (
            ("base URL", config.base_url),
            ("API key", config.api_key),
            ("model", config.model),
        )
        if not value
    ]
    if missing:
        raise LLMConfigurationError(
            f"LLM provider '{config.provider}' is missing: {', '.join(missing)}"
        )
    return config


class OpenAICompatibleLLMGateway:
    """Call the configured primary provider and fail over without leaking secrets."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self.last_attempts: tuple[ProviderAttemptAudit, ...] = ()
        self._retry_policy = ProviderRetryPolicy(
            max_attempts=settings.provider_max_attempts,
            backoff_base_seconds=settings.provider_backoff_base_seconds,
            backoff_max_seconds=settings.provider_backoff_max_seconds,
        )

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        response_json: bool = False,
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "structured_response",
        temperature: float | None = None,
        max_output_tokens: int = 1024,
        allow_fallback: bool = True,
    ) -> LLMCompletion:
        if not messages:
            raise ValueError("messages must not be empty")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")

        provider_names = self._provider_names(allow_fallback)
        attempted: list[str] = []
        last_error: Exception | None = None
        provider_attempts: list[ProviderAttemptAudit] = []
        self.last_attempts = ()
        for index, provider_name in enumerate(provider_names):
            attempted.append(provider_name.strip().lower())
            try:
                config = resolve_provider_config(self.settings, provider_name)
                result = await run_with_retry_async(
                    config.provider,
                    "chat_completion",
                    self._retry_policy,
                    lambda: self._resilient_async_completion(
                        config,
                        messages,
                        response_json=response_json,
                        response_schema=response_schema,
                        schema_name=schema_name,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        fallback_used=index > 0,
                        attempted_providers=tuple(attempted),
                    ),
                )
                provider_attempts.extend(result.attempts)
                self.last_attempts = tuple(provider_attempts)
                return replace(result.value, provider_attempts=self.last_attempts)
            except ProviderRetriesExhausted as exc:
                provider_attempts.extend(exc.attempts)
                last_error = exc
            except LLMConfigurationError as exc:
                last_error = exc

        attempted_text = ", ".join(attempted)
        self.last_attempts = tuple(provider_attempts)
        raise LLMProviderError(
            attempted_text,
            "all configured providers were unavailable",
            provider_attempts=self.last_attempts,
        ) from last_error

    async def probe_stream(
        self,
        provider: str,
        messages: list[dict[str, Any]],
        *,
        max_output_tokens: int = 128,
    ) -> LLMStreamProbe:
        """Probe one Provider directly; fallback would hide which endpoint was tested."""

        if not messages:
            raise ValueError("messages must not be empty")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        config = resolve_provider_config(self.settings, provider)
        self.last_attempts = ()
        try:
            result = await run_with_retry_async(
                config.provider,
                "stream_probe",
                self._retry_policy,
                lambda: self._resilient_stream_probe(
                    config,
                    messages,
                    max_output_tokens=max_output_tokens,
                ),
            )
        except ProviderRetriesExhausted as exc:
            self.last_attempts = exc.attempts
            raise LLMProviderError(
                config.provider,
                exc.failure.safe_message,
                status_code=exc.failure.status_code,
                provider_attempts=exc.attempts,
            ) from exc
        self.last_attempts = result.attempts
        return result.value

    async def stream_complete(
        self,
        messages: list[dict[str, Any]],
        *,
        on_content_delta: Callable[[str, str, str], None],
        max_output_tokens: int = 1024,
        allow_fallback: bool = True,
    ) -> LLMCompletion:
        """Consume a production stream and expose only visible content deltas."""

        if not messages:
            raise ValueError("messages must not be empty")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")

        attempted: list[str] = []
        last_error: Exception | None = None
        provider_attempts: list[ProviderAttemptAudit] = []
        self.last_attempts = ()
        for index, provider_name in enumerate(self._provider_names(allow_fallback)):
            attempted.append(provider_name.strip().lower())
            try:
                config = resolve_provider_config(self.settings, provider_name)
                result = await run_with_retry_async(
                    config.provider,
                    "stream_completion",
                    self._retry_policy,
                    lambda: self._resilient_stream_completion(
                        config,
                        messages,
                        on_content_delta=on_content_delta,
                        max_output_tokens=max_output_tokens,
                        fallback_used=index > 0,
                        attempted_providers=tuple(attempted),
                    ),
                )
                provider_attempts.extend(result.attempts)
                self.last_attempts = tuple(provider_attempts)
                return replace(result.value, provider_attempts=self.last_attempts)
            except ProviderRetriesExhausted as exc:
                provider_attempts.extend(exc.attempts)
                last_error = exc
                if exc.failure.failure_kind is ProviderFailureKind.STREAM_INTERRUPTED:
                    self.last_attempts = tuple(provider_attempts)
                    error = LLMProviderError(
                        config.provider,
                        exc.failure.safe_message,
                        status_code=exc.failure.status_code,
                        provider_attempts=self.last_attempts,
                    )
                    error.content_started = True
                    raise error from exc
            except LLMConfigurationError as exc:
                last_error = exc

        attempted_text = ", ".join(attempted)
        self.last_attempts = tuple(provider_attempts)
        raise LLMProviderError(
            attempted_text,
            "all configured providers were unavailable",
            provider_attempts=self.last_attempts,
        ) from last_error

    def complete_sync(
        self,
        messages: list[dict[str, Any]],
        *,
        response_json: bool = False,
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "structured_response",
        temperature: float | None = None,
        max_output_tokens: int = 1024,
        allow_fallback: bool = True,
    ) -> LLMCompletion:
        """Synchronous entry point for retrieval workers and FastAPI thread-pool routes."""

        if not messages:
            raise ValueError("messages must not be empty")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")

        attempted: list[str] = []
        last_error: Exception | None = None
        provider_attempts: list[ProviderAttemptAudit] = []
        self.last_attempts = ()
        for index, provider_name in enumerate(self._provider_names(allow_fallback)):
            attempted.append(provider_name.strip().lower())
            try:
                config = resolve_provider_config(self.settings, provider_name)
                result = run_with_retry(
                    config.provider,
                    "chat_completion_sync",
                    self._retry_policy,
                    lambda: self._resilient_sync_completion(
                        config,
                        messages,
                        response_json=response_json,
                        response_schema=response_schema,
                        schema_name=schema_name,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        fallback_used=index > 0,
                        attempted_providers=tuple(attempted),
                    ),
                )
                provider_attempts.extend(result.attempts)
                self.last_attempts = tuple(provider_attempts)
                return replace(result.value, provider_attempts=self.last_attempts)
            except ProviderRetriesExhausted as exc:
                provider_attempts.extend(exc.attempts)
                last_error = exc
            except LLMConfigurationError as exc:
                last_error = exc

        attempted_text = ", ".join(attempted)
        self.last_attempts = tuple(provider_attempts)
        raise LLMProviderError(
            attempted_text,
            "all configured providers were unavailable",
            provider_attempts=self.last_attempts,
        ) from last_error

    def _provider_names(self, allow_fallback: bool) -> list[str]:
        provider_names = [self.settings.llm_primary_provider]
        fallback = self.settings.llm_fallback_provider.strip().lower()
        if allow_fallback and fallback and fallback != provider_names[0].strip().lower():
            provider_names.append(fallback)
        return provider_names

    async def _resilient_async_completion(self, config: LLMProviderConfig, *args, **kwargs):
        try:
            return await self._complete_with_provider(config, *args, **kwargs)
        except (LLMProviderError, httpx.HTTPError) as exc:
            raise _llm_call_failure(exc) from exc

    async def _resilient_stream_probe(self, config: LLMProviderConfig, *args, **kwargs):
        try:
            return await self._probe_stream_with_provider(config, *args, **kwargs)
        except httpx.TimeoutException as exc:
            raise ProviderCallFailure(
                ProviderFailureKind.TIMEOUT,
                "stream request timed out",
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderCallFailure(
                ProviderFailureKind.TRANSPORT,
                "stream network error",
                retryable=True,
            ) from exc
        except LLMProviderError as exc:
            raise _llm_call_failure(exc) from exc

    async def _resilient_stream_completion(
        self,
        config: LLMProviderConfig,
        *args,
        **kwargs,
    ):
        try:
            return await self._stream_complete_with_provider(config, *args, **kwargs)
        except (LLMProviderError, httpx.HTTPError) as exc:
            if isinstance(exc, LLMProviderError) and exc.content_started:
                raise ProviderCallFailure(
                    ProviderFailureKind.STREAM_INTERRUPTED,
                    exc.safe_message,
                    status_code=exc.status_code,
                    retryable=False,
                ) from exc
            raise _llm_call_failure(exc) from exc

    def _resilient_sync_completion(self, config: LLMProviderConfig, *args, **kwargs):
        try:
            return self._complete_sync_with_provider(config, *args, **kwargs)
        except (LLMProviderError, httpx.HTTPError) as exc:
            raise _llm_call_failure(exc) from exc

    async def _complete_with_provider(
        self,
        config: LLMProviderConfig,
        messages: list[dict[str, Any]],
        *,
        response_json: bool,
        response_schema: dict[str, Any] | None,
        schema_name: str,
        temperature: float | None,
        max_output_tokens: int,
        fallback_used: bool,
        attempted_providers: tuple[str, ...],
    ) -> LLMCompletion:
        payload = self._build_payload(
            config,
            messages,
            response_json=response_json,
            response_schema=response_schema,
            schema_name=schema_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        endpoint = f"{config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(
            timeout=self.settings.llm_timeout_seconds,
            transport=self.transport,
        ) as client:
            response = await client.post(endpoint, headers=headers, json=payload)

        return self._parse_completion(
            response,
            config,
            fallback_used=fallback_used,
            attempted_providers=attempted_providers,
        )

    async def _probe_stream_with_provider(
        self,
        config: LLMProviderConfig,
        messages: list[dict[str, Any]],
        *,
        max_output_tokens: int,
    ) -> LLMStreamProbe:
        payload = self._build_payload(
            config,
            messages,
            response_json=False,
            max_output_tokens=max_output_tokens,
            stream=True,
        )
        endpoint = f"{config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }

        started = time.perf_counter()
        event_count = 0
        content_chunks: list[str] = []
        content_delta_count = 0
        reasoning_delta_count = 0
        first_event_ms: float | None = None
        first_content_delta_ms: float | None = None
        reported_model = config.model
        finish_reason: str | None = None
        done_received = False
        usage: dict[str, Any] = {}
        pending_data: list[str] = []

        def elapsed_ms() -> float:
            return round((time.perf_counter() - started) * 1000, 2)

        def consume_event(raw_data: str) -> None:
            nonlocal event_count
            nonlocal content_delta_count
            nonlocal reasoning_delta_count
            nonlocal first_event_ms
            nonlocal first_content_delta_ms
            nonlocal reported_model
            nonlocal finish_reason
            nonlocal done_received
            nonlocal usage

            if raw_data.strip() == "[DONE]":
                done_received = True
                return
            try:
                body = json.loads(raw_data)
            except (TypeError, ValueError) as exc:
                raise LLMProviderError(config.provider, "invalid streaming JSON event") from exc
            if not isinstance(body, dict):
                raise LLMProviderError(config.provider, "invalid streaming event shape")

            event_count += 1
            if first_event_ms is None:
                first_event_ms = elapsed_ms()
            if body.get("model"):
                reported_model = str(body["model"])
            raw_usage = body.get("usage")
            if isinstance(raw_usage, dict):
                usage = raw_usage

            choices = body.get("choices")
            if not isinstance(choices, list) or not choices:
                return
            choice = choices[0]
            if not isinstance(choice, dict):
                return
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                return
            reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                reasoning_delta_count += 1
            content = delta.get("content")
            if isinstance(content, str) and content:
                if first_content_delta_ms is None:
                    first_content_delta_ms = elapsed_ms()
                content_delta_count += 1
                content_chunks.append(content)

        async with httpx.AsyncClient(
            timeout=self.settings.llm_timeout_seconds,
            transport=self.transport,
        ) as client:
            async with client.stream("POST", endpoint, headers=headers, json=payload) as response:
                if response.is_error:
                    raise LLMProviderError(
                        config.provider,
                        f"HTTP {response.status_code}",
                        status_code=response.status_code,
                    )
                content_type = response.headers.get("content-type", "").lower()
                if "text/event-stream" not in content_type:
                    raise LLMProviderError(config.provider, "invalid streaming content type")

                async for line in response.aiter_lines():
                    if line == "":
                        if pending_data:
                            consume_event("\n".join(pending_data))
                            pending_data.clear()
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        pending_data.append(line[5:].lstrip())
                if pending_data:
                    consume_event("\n".join(pending_data))

        content = "".join(content_chunks)
        if not done_received and finish_reason is None:
            raise LLMProviderError(config.provider, "stream ended before a terminal signal")
        if not content.strip() or first_content_delta_ms is None:
            raise LLMProviderError(config.provider, "stream contained no visible content delta")

        return LLMStreamProbe(
            provider=config.provider,
            requested_model=config.model,
            reported_model=reported_model,
            content=content,
            content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            content_length=len(content),
            event_count=event_count,
            content_delta_count=content_delta_count,
            reasoning_delta_count=reasoning_delta_count,
            first_event_ms=first_event_ms or first_content_delta_ms,
            first_content_delta_ms=first_content_delta_ms,
            total_ms=elapsed_ms(),
            finish_reason=finish_reason,
            done_received=done_received,
            termination="done_marker" if done_received else "finish_reason_eof",
            usage=usage,
        )

    async def _stream_complete_with_provider(
        self,
        config: LLMProviderConfig,
        messages: list[dict[str, Any]],
        *,
        on_content_delta: Callable[[str, str, str], None],
        max_output_tokens: int,
        fallback_used: bool,
        attempted_providers: tuple[str, ...],
    ) -> LLMCompletion:
        payload = self._build_payload(
            config,
            messages,
            response_json=False,
            max_output_tokens=max_output_tokens,
            stream=True,
        )
        endpoint = f"{config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        content_chunks: list[str] = []
        reported_model = config.model
        finish_reason: str | None = None
        done_received = False
        usage: dict[str, Any] = {}
        pending_data: list[str] = []

        def provider_error(message: str, *, status_code: int | None = None) -> LLMProviderError:
            error = LLMProviderError(config.provider, message, status_code=status_code)
            error.content_started = bool(content_chunks)
            return error

        def consume_event(raw_data: str) -> None:
            nonlocal reported_model, finish_reason, done_received, usage
            if raw_data.strip() == "[DONE]":
                done_received = True
                return
            try:
                body = json.loads(raw_data)
            except (TypeError, ValueError) as exc:
                raise provider_error("invalid streaming JSON event") from exc
            if not isinstance(body, dict):
                raise provider_error("invalid streaming event shape")
            if body.get("model"):
                reported_model = str(body["model"])
            raw_usage = body.get("usage")
            if isinstance(raw_usage, dict):
                usage = raw_usage
            choices = body.get("choices")
            if not isinstance(choices, list) or not choices:
                return
            choice = choices[0]
            if not isinstance(choice, dict):
                return
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                return
            content = delta.get("content")
            if isinstance(content, str) and content:
                content_chunks.append(content)
                on_content_delta(content, config.provider, reported_model)

        try:
            async with httpx.AsyncClient(
                timeout=self.settings.llm_timeout_seconds,
                transport=self.transport,
            ) as client:
                async with client.stream("POST", endpoint, headers=headers, json=payload) as response:
                    if response.is_error:
                        raise provider_error(
                            f"HTTP {response.status_code}",
                            status_code=response.status_code,
                        )
                    content_type = response.headers.get("content-type", "").lower()
                    if "text/event-stream" not in content_type:
                        raise provider_error("invalid streaming content type")
                    async for line in response.aiter_lines():
                        if line == "":
                            if pending_data:
                                consume_event("\n".join(pending_data))
                                pending_data.clear()
                            continue
                        if line.startswith(":"):
                            continue
                        if line.startswith("data:"):
                            pending_data.append(line[5:].lstrip())
                    if pending_data:
                        consume_event("\n".join(pending_data))
        except httpx.TimeoutException as exc:
            raise provider_error("stream request timed out") from exc
        except httpx.HTTPError as exc:
            raise provider_error("stream network error") from exc

        content = "".join(content_chunks)
        if not done_received and finish_reason is None:
            raise provider_error("stream ended before a terminal signal")
        if not content.strip():
            raise provider_error("stream contained no visible content delta")
        return LLMCompletion(
            content=content,
            provider=config.provider,
            requested_model=config.model,
            reported_model=reported_model,
            fallback_used=fallback_used,
            attempted_providers=attempted_providers,
            usage=usage,
        )

    def _complete_sync_with_provider(
        self,
        config: LLMProviderConfig,
        messages: list[dict[str, Any]],
        *,
        response_json: bool,
        response_schema: dict[str, Any] | None,
        schema_name: str,
        temperature: float | None,
        max_output_tokens: int,
        fallback_used: bool,
        attempted_providers: tuple[str, ...],
    ) -> LLMCompletion:
        payload = self._build_payload(
            config,
            messages,
            response_json=response_json,
            response_schema=response_schema,
            schema_name=schema_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        endpoint = f"{config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(
            timeout=self.settings.llm_timeout_seconds,
            transport=self.transport,
        ) as client:
            response = client.post(endpoint, headers=headers, json=payload)
        return self._parse_completion(
            response,
            config,
            fallback_used=fallback_used,
            attempted_providers=attempted_providers,
        )

    @staticmethod
    def _build_payload(
        config: LLMProviderConfig,
        messages: list[dict[str, Any]],
        *,
        response_json: bool,
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "structured_response",
        temperature: float | None = None,
        max_output_tokens: int,
        stream: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            config.max_tokens_field: max_output_tokens,
        }
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": response_schema,
                },
            }
        elif response_json:
            payload["response_format"] = {"type": "json_object"}
        if temperature is not None:
            payload["temperature"] = temperature
        if stream:
            payload["stream"] = True
        if config.reasoning_effort:
            payload["reasoning_effort"] = config.reasoning_effort
        if config.verbosity:
            payload["verbosity"] = config.verbosity
        return payload

    @staticmethod
    def _parse_completion(
        response: httpx.Response,
        config: LLMProviderConfig,
        *,
        fallback_used: bool,
        attempted_providers: tuple[str, ...],
    ) -> LLMCompletion:
        if response.is_error:
            raise LLMProviderError(
                config.provider,
                f"HTTP {response.status_code}",
                status_code=response.status_code,
            )
        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMProviderError(config.provider, "invalid Chat Completions response") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError(config.provider, "empty completion content")

        usage = body.get("usage")
        return LLMCompletion(
            content=content,
            provider=config.provider,
            requested_model=config.model,
            reported_model=str(body.get("model") or config.model),
            fallback_used=fallback_used,
            attempted_providers=attempted_providers,
            usage=usage if isinstance(usage, dict) else {},
        )


def _llm_call_failure(exc: Exception) -> ProviderCallFailure:
    if isinstance(exc, LLMProviderError):
        status = exc.status_code
        if status == 429:
            return ProviderCallFailure(
                ProviderFailureKind.RATE_LIMITED,
                exc.safe_message,
                status_code=status,
                retryable=True,
            )
        if status is not None and status >= 500:
            return ProviderCallFailure(
                ProviderFailureKind.UPSTREAM_5XX,
                exc.safe_message,
                status_code=status,
                retryable=True,
            )
        if status is not None and status >= 400:
            return ProviderCallFailure(
                ProviderFailureKind.REJECTED,
                exc.safe_message,
                status_code=status,
                retryable=False,
            )
        normalized = exc.safe_message.lower()
        if "timed out" in normalized or "timeout" in normalized:
            return ProviderCallFailure(
                ProviderFailureKind.TIMEOUT,
                exc.safe_message,
                retryable=True,
            )
        if "network" in normalized:
            return ProviderCallFailure(
                ProviderFailureKind.TRANSPORT,
                exc.safe_message,
                retryable=True,
            )
        if any(token in normalized for token in ("invalid", "empty", "no visible")):
            return ProviderCallFailure(
                ProviderFailureKind.INVALID_RESPONSE,
                exc.safe_message,
                retryable=False,
            )
        return ProviderCallFailure(
            ProviderFailureKind.STREAM_INTERRUPTED,
            exc.safe_message,
            retryable=False,
        )
    return failure_from_exception(exc, "LLM provider transport failure.")
