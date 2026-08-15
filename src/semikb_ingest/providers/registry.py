"""Explicit provider dependencies without a hidden default service."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from semikb_ingest.errors import IngestError, IngestErrorCode


@runtime_checkable
class ProviderClient(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def provider_version(self) -> str: ...


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, ProviderClient] = {}

    def register(self, provider: ProviderClient) -> None:
        if not isinstance(provider, ProviderClient):
            raise TypeError("Provider must expose provider_name and provider_version.")
        if provider.provider_name in self._providers:
            raise ValueError(f"Provider {provider.provider_name!r} is already registered.")
        self._providers[provider.provider_name] = provider

    def require(self, provider_name: str) -> ProviderClient:
        provider = self._providers.get(provider_name)
        if provider is None:
            raise IngestError(
                IngestErrorCode.PARSER_NOT_CONFIGURED,
                f"Provider {provider_name!r} is not configured for this format.",
            )
        return provider
