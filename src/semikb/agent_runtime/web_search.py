"""Controlled Alibaba Cloud Web Search MCP gateway."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from semikb.config import Settings


class WebSearchUnavailable(RuntimeError):
    """Raised when an external search is requested without a usable MCP configuration."""


class AliyunWebSearchGateway:
    """Keeps MCP protocol details outside LangGraph and retrieval business logic."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def should_search(self, query: str) -> bool:
        trigger_words = ("最新", "外部", "公开资料", "厂商", "论文", "web", "联网")
        return bool(self.settings.aliyun_web_mcp_api_key) and any(word in query.lower() for word in trigger_words)

    async def search(self, query: str) -> list[dict[str, str]]:
        """Call configured MCP using the Python MCP client when credentials are available."""

        if not self.settings.aliyun_web_mcp_api_key:
            raise WebSearchUnavailable("Alibaba Cloud Web Search MCP is not configured.")
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as exc:  # pragma: no cover - exercised only with real MCP deployment
            raise WebSearchUnavailable("Install the optional MCP runtime before enabling Web Search.") from exc

        headers = {"Authorization": f"Bearer {self.settings.aliyun_web_mcp_api_key}"}
        async with streamablehttp_client(self.settings.aliyun_web_mcp_url, headers=headers) as streams:
            read_stream, write_stream = streams[0], streams[1]
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(self.settings.aliyun_web_mcp_tool_name, {"query": query})

        normalized: list[dict[str, str]] = []
        for content in result.content:
            text = getattr(content, "text", "")
            if not text:
                continue
            urls = re.findall(r"https?://[^\s)\\\"]+", text)
            for url in urls:
                normalized.append({"source_type": "external", "content": text, "url": url})
        return [item for item in normalized if self._allowed(item["url"])]

    def _allowed(self, url: str) -> bool:
        if not self.settings.allowed_domains:
            return False
        host = urlparse(url).hostname or ""
        return any(host == domain or host.endswith(f".{domain}") for domain in self.settings.allowed_domains)
