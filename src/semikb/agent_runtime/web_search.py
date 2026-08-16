"""Controlled Alibaba Cloud Web Search MCP gateway."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse

from semikb.config import Settings
from semikb_provider_resilience import (
    ProviderAttemptAudit,
    ProviderCallFailure,
    ProviderFailureKind,
    ProviderRetriesExhausted,
    ProviderRetryPolicy,
    invalid_response,
    run_with_retry_async,
)


class WebSearchUnavailable(RuntimeError):
    """Raised when external search is unavailable without exposing Provider details."""

    def __init__(
        self,
        message: str,
        *,
        provider_attempts: tuple[ProviderAttemptAudit, ...] = (),
    ) -> None:
        super().__init__(message)
        self.provider_attempts = provider_attempts


class AliyunWebSearchGateway:
    """Keeps MCP protocol details outside LangGraph and retrieval business logic."""

    def __init__(
        self,
        settings: Settings,
        *,
        client_call: Callable[[str], Awaitable[Any]] | None = None,
    ) -> None:
        self.settings = settings
        self._client_call = client_call
        self.last_attempts: tuple[ProviderAttemptAudit, ...] = ()
        self._retry_policy = ProviderRetryPolicy(
            max_attempts=settings.provider_max_attempts,
            backoff_base_seconds=settings.provider_backoff_base_seconds,
            backoff_max_seconds=settings.provider_backoff_max_seconds,
        )

    def should_search(self, query: str) -> bool:
        trigger_words = ("最新", "外部", "公开资料", "厂商", "论文", "web", "联网")
        return bool(self.settings.aliyun_web_mcp_api_key) and any(
            word in query.lower() for word in trigger_words
        )

    async def search(self, query: str) -> list[dict[str, str]]:
        """Run one bounded, read-only MCP search and retain safe attempt evidence."""

        if not self.settings.aliyun_web_mcp_api_key:
            raise WebSearchUnavailable("Alibaba Cloud Web Search MCP is not configured.")
        self.last_attempts = ()
        try:
            result = await run_with_retry_async(
                "aliyun-web-mcp",
                "web_search",
                self._retry_policy,
                lambda: self._search_once(query),
            )
        except ProviderRetriesExhausted as exc:
            self.last_attempts = exc.attempts
            raise WebSearchUnavailable(
                "External Web search is temporarily unavailable; internal evidence remains usable.",
                provider_attempts=exc.attempts,
            ) from exc
        self.last_attempts = result.attempts
        return result.value

    async def _search_once(self, query: str) -> list[dict[str, str]]:
        try:
            async with asyncio.timeout(self.settings.web_search_timeout_seconds):
                result = (
                    await self._client_call(query)
                    if self._client_call is not None
                    else await self._invoke_mcp(query)
                )
        except ImportError as exc:
            raise ProviderCallFailure(
                ProviderFailureKind.CONFIGURATION,
                "The optional MCP runtime is not installed.",
                retryable=False,
            ) from exc

        content_items = getattr(result, "content", None)
        if not isinstance(content_items, list):
            raise invalid_response("Web Search MCP returned an invalid result shape.")
        normalized: list[dict[str, str]] = []
        for content in content_items:
            text = getattr(content, "text", "")
            if not isinstance(text, str) or not text:
                continue
            urls = re.findall(r"https?://[^\s)\\\"]+", text)
            for url in urls:
                normalized.append({"source_type": "external", "content": text, "url": url})
        return [item for item in normalized if self._allowed(item["url"])]

    async def _invoke_mcp(self, query: str):
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        headers = {"Authorization": f"Bearer {self.settings.aliyun_web_mcp_api_key}"}
        async with streamablehttp_client(
            self.settings.aliyun_web_mcp_url,
            headers=headers,
        ) as streams:
            read_stream, write_stream = streams[0], streams[1]
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return await session.call_tool(
                    self.settings.aliyun_web_mcp_tool_name,
                    {"query": query},
                )

    def _allowed(self, url: str) -> bool:
        if not self.settings.allowed_domains:
            return False
        host = urlparse(url).hostname or ""
        return any(
            host == domain or host.endswith(f".{domain}")
            for domain in self.settings.allowed_domains
        )
