"""Controlled Alibaba Cloud Web Search MCP gateway."""

from __future__ import annotations

import asyncio
import ipaddress
import json
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
        self.last_audit: dict[str, Any] = {}
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

    async def search(self, query: str) -> list[dict[str, Any]]:
        """Run bounded MCP searches and normalize structured result pages."""

        if not self.settings.aliyun_web_mcp_api_key:
            raise WebSearchUnavailable("Alibaba Cloud Web Search MCP is not configured.")
        self.last_attempts = ()
        self.last_audit = {}
        variants = [query.strip()]
        if self.settings.web_search_content_retries:
            variants.append(self._rewrite_query(query))
        variants = list(dict.fromkeys(item for item in variants if item))
        attempts: list[ProviderAttemptAudit] = []
        normalized_count_by_attempt: list[int] = []

        for variant_index, variant in enumerate(variants):
            try:
                result = await run_with_retry_async(
                    "aliyun-web-mcp",
                    "web_search",
                    self._retry_policy,
                    lambda variant=variant: self._search_once(variant),
                )
            except ProviderRetriesExhausted as exc:
                attempts.extend(exc.attempts)
                self.last_attempts = tuple(attempts)
                self.last_audit = {
                    "schema_version": "semikb-web-search-audit-v1",
                    "query_variant_count": variant_index + 1,
                    "normalized_count_by_attempt": normalized_count_by_attempt,
                    "final_status": "provider_unavailable",
                }
                raise WebSearchUnavailable(
                    "External Web search is temporarily unavailable.",
                    provider_attempts=tuple(attempts),
                ) from exc
            attempts.extend(result.attempts)
            normalized_count_by_attempt.append(len(result.value))
            if result.value:
                self.last_attempts = tuple(attempts)
                self.last_audit = {
                    "schema_version": "semikb-web-search-audit-v1",
                    "query_variant_count": variant_index + 1,
                    "normalized_count_by_attempt": normalized_count_by_attempt,
                    "returned_count": len(result.value),
                    "used_rewritten_query": variant_index > 0,
                    "final_status": "usable_results",
                }
                return result.value

        self.last_attempts = tuple(attempts)
        self.last_audit = {
            "schema_version": "semikb-web-search-audit-v1",
            "query_variant_count": len(variants),
            "normalized_count_by_attempt": normalized_count_by_attempt,
            "returned_count": 0,
            "used_rewritten_query": len(variants) > 1,
            "final_status": "no_usable_results",
        }
        return []

    async def _search_once(self, query: str) -> list[dict[str, Any]]:
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

        content_items = (
            result.get("content") if isinstance(result, dict) else getattr(result, "content", None)
        )
        if not isinstance(content_items, list):
            raise invalid_response("Web Search MCP returned an invalid result shape.")
        normalized: list[dict[str, Any]] = []
        structured = (
            result.get("structuredContent")
            if isinstance(result, dict)
            else getattr(result, "structuredContent", None)
        )
        if structured is not None:
            normalized.extend(self._normalize_payload(structured))
        for content in content_items:
            text = content.get("text", "") if isinstance(content, dict) else getattr(content, "text", "")
            if not isinstance(text, str) or not text:
                continue
            try:
                normalized.extend(self._normalize_payload(json.loads(text)))
            except (json.JSONDecodeError, TypeError, ValueError):
                normalized.extend(self._normalize_text(text))

        deduplicated: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for item in normalized:
            url = str(item.get("url", "")).strip()
            if not self._safe_external_url(url) or url in seen_urls:
                continue
            seen_urls.add(url)
            host = (urlparse(url).hostname or "").lower()
            item["source_type"] = "external"
            item["source_domain"] = str(item.get("source_domain") or host)
            item["source_priority"] = "preferred" if self._preferred(host) else "standard"
            item["content"] = self._clean_text(item.get("content"), limit=1600)
            item["title"] = self._clean_text(item.get("title"), limit=300)
            deduplicated.append(item)
            if len(deduplicated) >= self.settings.web_search_max_results:
                break
        return deduplicated

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

    def _preferred(self, host: str) -> bool:
        return any(
            host == domain or host.endswith(f".{domain}")
            for domain in self.settings.allowed_domains
        )

    def _normalize_payload(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for value in payload for item in self._normalize_payload(value)]
        if not isinstance(payload, dict):
            return []
        pages = payload.get("pages")
        if isinstance(pages, list):
            return [self._normalize_page(page) for page in pages if isinstance(page, dict)]
        for key in ("result", "data", "output"):
            if key in payload:
                nested = self._normalize_payload(payload[key])
                if nested:
                    return nested
        if payload.get("url"):
            return [self._normalize_page(payload)]
        return []

    def _normalize_text(self, text: str) -> list[dict[str, Any]]:
        markdown_links = re.findall(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", text)
        if markdown_links:
            return [
                {"title": title, "url": url, "content": text}
                for title, url in markdown_links
            ]
        return [
            {"title": "", "url": url.rstrip(".,;"), "content": text}
            for url in re.findall(r"https?://[^\s)\\\"]+", text)
        ]

    @staticmethod
    def _normalize_page(page: dict[str, Any]) -> dict[str, Any]:
        return {
            "title": page.get("title", ""),
            "url": page.get("url", ""),
            "content": page.get("snippet") or page.get("content") or page.get("text") or "",
            "source_domain": page.get("hostname", ""),
        }

    @staticmethod
    def _safe_external_url(url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        if parsed.username or parsed.password:
            return False
        host = parsed.hostname.casefold().rstrip(".")
        if host == "localhost" or host.endswith(".localhost"):
            return False
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return True
        return not (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        )

    @staticmethod
    def _clean_text(value: Any, *, limit: int) -> str:
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(value or "")).strip()[:limit]

    @staticmethod
    def _rewrite_query(query: str) -> str:
        return f"{query.strip()} 权威资料 详细说明".strip()
