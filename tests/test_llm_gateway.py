from __future__ import annotations

import json

import httpx
import pytest

from semikb.agent_runtime.llm_gateway import (
    LLMConfigurationError,
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

    assert [host for host, _ in payloads] == ["closeai.invalid", "qwen.invalid"]
    assert payloads[1][1]["max_tokens"] == 32
    assert "reasoning_effort" not in payloads[1][1]
    assert result.provider == "qwen"
    assert result.fallback_used is True
    assert result.attempted_providers == ("closeai", "qwen")
