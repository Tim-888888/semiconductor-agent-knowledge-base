from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from semikb.agent_runtime.web_search import AliyunWebSearchGateway, WebSearchUnavailable
from semikb.config import Settings


def settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "aliyun_web_mcp_api_key": "secret",
        "web_allowed_domains": "arxiv.org",
        "provider_max_attempts": 2,
        "provider_backoff_base_seconds": 0,
        "provider_backoff_max_seconds": 0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.mark.asyncio
async def test_web_search_retries_timeout_and_keeps_only_allowed_urls() -> None:
    calls = 0

    async def client_call(_query: str):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("secret upstream detail")
        return SimpleNamespace(
            content=[
                SimpleNamespace(
                    text=(
                        "Paper https://arxiv.org/abs/2601.00001 "
                        "mirror https://untrusted.example/paper"
                    )
                )
            ]
        )

    gateway = AliyunWebSearchGateway(settings(), client_call=client_call)
    results = await gateway.search("最新半导体论文")

    assert calls == 2
    assert [item["url"] for item in results] == ["https://arxiv.org/abs/2601.00001"]
    assert [attempt.outcome for attempt in gateway.last_attempts] == ["retrying", "succeeded"]
    assert "secret" not in str(gateway.last_attempts)


@pytest.mark.asyncio
async def test_web_search_invalid_response_fails_without_retry() -> None:
    calls = 0

    async def client_call(_query: str):
        nonlocal calls
        calls += 1
        return SimpleNamespace(content="invalid")

    gateway = AliyunWebSearchGateway(settings(), client_call=client_call)

    with pytest.raises(WebSearchUnavailable) as captured:
        await gateway.search("最新半导体论文")

    assert calls == 1
    assert captured.value.provider_attempts[0].failure_kind == "invalid_response"
    assert captured.value.provider_attempts[0].retryable is False
