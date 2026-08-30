from __future__ import annotations

import json
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
async def test_web_search_retries_timeout_and_keeps_public_results_without_domain_filter() -> None:
    calls = 0

    async def client_call(_query: str):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("secret upstream detail")
        return SimpleNamespace(
            content=[
                SimpleNamespace(
                    text=json.dumps(
                        {
                            "pages": [
                                {
                                    "title": "Paper",
                                    "url": "https://arxiv.org/abs/2601.00001",
                                    "snippet": "abstract",
                                    "hostname": "arxiv.org",
                                },
                                {
                                    "title": "Vendor",
                                    "url": "https://vendor.example/process",
                                    "snippet": "process flow",
                                    "hostname": "vendor.example",
                                },
                            ]
                        }
                    )
                )
            ]
        )

    gateway = AliyunWebSearchGateway(settings(), client_call=client_call)
    results = await gateway.search("最新半导体论文")

    assert calls == 2
    assert [item["url"] for item in results] == [
        "https://arxiv.org/abs/2601.00001",
        "https://vendor.example/process",
    ]
    assert results[0]["source_priority"] == "preferred"
    assert results[1]["source_priority"] == "standard"
    assert [attempt.outcome for attempt in gateway.last_attempts] == ["retrying", "succeeded"]
    assert "secret" not in str(gateway.last_attempts)


@pytest.mark.asyncio
async def test_web_search_rewrites_once_after_empty_content_and_rejects_private_urls() -> None:
    queries: list[str] = []

    async def client_call(query: str):
        queries.append(query)
        pages = [] if len(queries) == 1 else [
            {
                "title": "Valid source",
                "url": "https://example.org/semiconductor",
                "snippet": "manufacturing overview",
                "hostname": "example.org",
            },
            {
                "title": "Private",
                "url": "http://127.0.0.1/admin",
                "snippet": "must not be returned",
                "hostname": "127.0.0.1",
            },
        ]
        return SimpleNamespace(
            content=[SimpleNamespace(text=json.dumps({"pages": pages}))]
        )

    gateway = AliyunWebSearchGateway(settings(), client_call=client_call)
    results = await gateway.search("半导体制造概览")

    assert len(queries) == 2
    assert queries[1].endswith("权威资料 详细说明")
    assert [item["url"] for item in results] == ["https://example.org/semiconductor"]
    assert gateway.last_audit["used_rewritten_query"] is True
    assert gateway.last_audit["normalized_count_by_attempt"] == [0, 1]


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
