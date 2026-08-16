from __future__ import annotations

import json

import httpx
import pytest

from semikb.agent_runtime.llm_gateway import (
    LLMConfigurationError,
    LLMProviderError,
    OpenAICompatibleLLMGateway,
    resolve_provider_config,
)
from semikb.config import Settings


def llm_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "llm_primary_provider": "closeai",
        "llm_fallback_provider": "qwen",
        "closeai_base_url": "https://closeai.invalid/v1",
        "closeai_api_key": "closeai-secret",
        "closeai_model": "gpt-5.6-luna",
        "qwen_api_base_url": "https://qwen.invalid/v1",
        "qwen_api_key": "qwen-secret",
        "qwen_model": "qwen-flash",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_qwen_provider_accepts_legacy_generic_settings() -> None:
    settings = llm_settings(
        qwen_api_base_url="",
        qwen_api_key="",
        qwen_model="",
        llm_api_base_url="https://legacy.invalid/v1",
        llm_api_key="legacy-secret",
        llm_model="qwen-flash",
    )

    config = resolve_provider_config(settings, "qwen")

    assert config.base_url == "https://legacy.invalid/v1"
    assert config.model == "qwen-flash"


def test_unknown_provider_is_rejected_without_exposing_credentials() -> None:
    with pytest.raises(LLMConfigurationError, match="Unsupported LLM provider: other"):
        resolve_provider_config(llm_settings(), "other")


def test_provider_config_repr_redacts_api_key() -> None:
    config = resolve_provider_config(llm_settings(), "closeai")

    assert "closeai-secret" not in repr(config)


def test_sync_gateway_uses_same_luna_parameter_contract() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "gpt-5.6-luna-test",
                "choices": [{"message": {"content": "hypothetical passage"}}],
            },
        )

    gateway = OpenAICompatibleLLMGateway(
        llm_settings(),
        transport=httpx.MockTransport(handler),
    )
    result = gateway.complete_sync(
        [{"role": "user", "content": "Generate HyDE"}],
        max_output_tokens=64,
        allow_fallback=False,
    )

    assert captured["max_completion_tokens"] == 64
    assert "temperature" not in captured
    assert result.reported_model == "gpt-5.6-luna-test"


@pytest.mark.asyncio
async def test_closeai_uses_luna_compatible_parameters() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "gpt-5.6-luna-2026-07-09",
                "choices": [{"message": {"content": '{"answer":"ok"}'}}],
                "usage": {"total_tokens": 12},
            },
        )

    gateway = OpenAICompatibleLLMGateway(
        llm_settings(),
        transport=httpx.MockTransport(handler),
    )
    result = await gateway.complete(
        [{"role": "user", "content": "Return JSON"}],
        response_json=True,
        max_output_tokens=80,
    )

    assert captured["max_completion_tokens"] == 80
    assert captured["reasoning_effort"] == "none"
    assert captured["verbosity"] == "low"
    assert captured["response_format"] == {"type": "json_object"}
    assert "temperature" not in captured
    assert "max_tokens" not in captured
    assert result.provider == "closeai"
    assert result.fallback_used is False


@pytest.mark.asyncio
async def test_structured_completion_sends_strict_schema_and_zero_temperature() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "structured-model",
                "choices": [{"message": {"content": '{"route":"chat_direct"}'}}],
            },
        )

    schema = {
        "type": "object",
        "properties": {"route": {"type": "string", "enum": ["chat_direct"]}},
        "required": ["route"],
        "additionalProperties": False,
    }
    gateway = OpenAICompatibleLLMGateway(
        llm_settings(),
        transport=httpx.MockTransport(handler),
    )

    await gateway.complete(
        [{"role": "user", "content": "route"}],
        response_schema=schema,
        schema_name="route_schema",
        temperature=0,
        allow_fallback=False,
    )

    assert captured["temperature"] == 0
    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "route_schema", "strict": True, "schema": schema},
    }


@pytest.mark.asyncio
async def test_primary_failure_falls_back_with_qwen_parameter_shape() -> None:
    payloads: list[tuple[str, dict[str, object]]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append((request.url.host, payload))
        if request.url.host == "closeai.invalid":
            return httpx.Response(503, json={"error": {"message": "unavailable"}})
        return httpx.Response(
            200,
            json={
                "model": "qwen-flash",
                "choices": [{"message": {"content": "fallback ok"}}],
            },
        )

    gateway = OpenAICompatibleLLMGateway(
        llm_settings(),
        transport=httpx.MockTransport(handler),
    )
    result = await gateway.complete(
        [{"role": "user", "content": "Hello"}],
        max_output_tokens=32,
    )

    assert [host for host, _ in payloads] == [
        "closeai.invalid",
        "closeai.invalid",
        "qwen.invalid",
    ]
    assert payloads[2][1]["max_tokens"] == 32
    assert "reasoning_effort" not in payloads[2][1]
    assert result.provider == "qwen"
    assert result.fallback_used is True
    assert result.attempted_providers == ("closeai", "qwen")
    assert [attempt.outcome for attempt in result.provider_attempts] == [
        "retrying",
        "failed",
        "succeeded",
    ]


class ChunkedAsyncStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


class FailingAfterContentStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        raise httpx.ReadError("stream interrupted")


@pytest.mark.asyncio
async def test_stream_probe_parses_network_splits_and_ignores_reasoning_content() -> None:
    captured: dict[str, object] = {}
    body = (
        'data: {"model":"gpt-5.6-luna-stream","choices":[{"delta":{"reasoning_content":"hidden"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"流"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"式输出"},"finish_reason":"stop"}]}\n\n'
        'data: [DONE]\n\n'
    ).encode()
    split_at = body.index("流".encode()) + 1

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream; charset=utf-8"},
            stream=ChunkedAsyncStream([body[:split_at], body[split_at:]]),
        )

    gateway = OpenAICompatibleLLMGateway(
        llm_settings(),
        transport=httpx.MockTransport(handler),
    )
    result = await gateway.probe_stream(
        "closeai",
        [{"role": "user", "content": "stream"}],
        max_output_tokens=32,
    )

    assert captured["stream"] is True
    assert captured["max_completion_tokens"] == 32
    assert result.content == "流式输出"
    assert result.content_delta_count == 2
    assert result.reasoning_delta_count == 1
    assert result.reported_model == "gpt-5.6-luna-stream"
    assert result.done_received is True
    assert "hidden" not in repr(result)


@pytest.mark.asyncio
async def test_stream_probe_rejects_a_stream_without_done_marker() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
        )

    gateway = OpenAICompatibleLLMGateway(
        llm_settings(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMProviderError, match="ended before a terminal signal"):
        await gateway.probe_stream(
            "qwen",
            [{"role": "user", "content": "stream"}],
        )


@pytest.mark.asyncio
async def test_stream_probe_accepts_finish_reason_followed_by_clean_eof() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices":[{"delta":{"content":"complete"},'
                b'"finish_reason":"stop"}]}\n\n'
            ),
        )

    gateway = OpenAICompatibleLLMGateway(
        llm_settings(),
        transport=httpx.MockTransport(handler),
    )
    result = await gateway.probe_stream(
        "closeai",
        [{"role": "user", "content": "stream"}],
    )

    assert result.done_received is False
    assert result.finish_reason == "stop"
    assert result.termination == "finish_reason_eof"


@pytest.mark.asyncio
async def test_stream_probe_redacts_network_failures() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            "upstream contained closeai-secret",
            request=request,
        )

    gateway = OpenAICompatibleLLMGateway(
        llm_settings(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMProviderError, match="stream network error") as raised:
        await gateway.probe_stream(
            "closeai",
            [{"role": "user", "content": "stream"}],
        )

    assert "closeai-secret" not in str(raised.value)
    assert "closeai.invalid" not in str(raised.value)


@pytest.mark.asyncio
async def test_production_stream_forwards_visible_deltas_and_metadata() -> None:
    body = (
        'data: {"model":"gpt-stream","choices":[{"delta":{"reasoning_content":"hidden"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"{\\"type\\":\\"unknown\\","}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"\\"text\\":\\"待确认\\"}"},"finish_reason":"stop"}]}\n\n'
        'data: [DONE]\n\n'
    ).encode()

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ChunkedAsyncStream([body[:73], body[73:]]),
        )

    deltas: list[tuple[str, str, str]] = []
    gateway = OpenAICompatibleLLMGateway(
        llm_settings(),
        transport=httpx.MockTransport(handler),
    )
    result = await gateway.stream_complete(
        [{"role": "user", "content": "stream"}],
        on_content_delta=lambda delta, provider, model: deltas.append(
            (delta, provider, model)
        ),
        allow_fallback=False,
    )

    assert "".join(item[0] for item in deltas) == '{"type":"unknown","text":"待确认"}'
    assert all(item[1] == "closeai" for item in deltas)
    assert result.reported_model == "gpt-stream"
    assert "hidden" not in result.content


@pytest.mark.asyncio
async def test_stream_failure_after_visible_content_never_retries_or_falls_back() -> None:
    hosts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=FailingAfterContentStream(),
        )

    deltas: list[str] = []
    gateway = OpenAICompatibleLLMGateway(
        llm_settings(provider_backoff_base_seconds=0, provider_backoff_max_seconds=0),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMProviderError) as captured:
        await gateway.stream_complete(
            [{"role": "user", "content": "stream"}],
            on_content_delta=lambda delta, _provider, _model: deltas.append(delta),
        )

    assert deltas == ["partial"]
    assert hosts == ["closeai.invalid"]
    assert captured.value.content_started is True
    assert captured.value.provider_attempts[0].failure_kind == "stream_interrupted"
