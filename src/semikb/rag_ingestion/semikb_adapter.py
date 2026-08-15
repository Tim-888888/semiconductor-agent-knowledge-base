"""Governed composition boundary for the standalone document adapters."""

from __future__ import annotations

from dataclasses import dataclass

from semikb.config import Settings
from semikb_ingest import IngestDispatcher, ParsedDocument, build_dispatcher
from semikb_ingest.assets import ProcessPayloadStore
from semikb_ingest.chunking.structured import StructuredBlockChunker
from semikb_ingest.providers import (
    MinerUPdfClient,
    MinerUPdfConfig,
    ProviderRegistry,
    QwenVisionClient,
    QwenVisionConfig,
)
from semikb_ingest.routing import ResolvedRoute


@dataclass(slots=True)
class ParsedIngestSession:
    """Own temporary extracted bytes until the business service persists them."""

    document: ParsedDocument
    payload_store: ProcessPayloadStore

    def pop_image_bytes(self, asset_id: str) -> bytes:
        image = next(item for item in self.document.images if item.asset_id == asset_id)
        return self.payload_store.pop(image.payload)

    def discard_remaining(self) -> None:
        for image in self.document.images:
            self.payload_store.discard(image.payload.handle)


class SemikbIngestAdapter:
    """Build exact-format parsers without exposing Provider details to jobs or stores."""

    chunker_version = StructuredBlockChunker.chunker_version

    def __init__(
        self,
        settings: Settings,
        provider_registry: ProviderRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._providers = provider_registry or self._build_provider_registry(settings)

    def resolve(
        self,
        filename: str,
        content: bytes,
        declared_media_type: str | None,
    ) -> ResolvedRoute:
        return self._dispatcher(ProcessPayloadStore()).resolve(
            filename,
            content,
            declared_media_type,
        )

    def parse(
        self,
        filename: str,
        content: bytes,
        *,
        correlation_id: str,
        declared_media_type: str | None,
    ) -> ParsedIngestSession:
        payload_store = ProcessPayloadStore()
        dispatcher = self._dispatcher(payload_store)
        document = dispatcher.parse(
            filename,
            content,
            correlation_id=correlation_id,
            declared_media_type=declared_media_type,
        )
        return ParsedIngestSession(document=document, payload_store=payload_store)

    def _dispatcher(self, payload_store: ProcessPayloadStore) -> IngestDispatcher:
        return build_dispatcher(payload_store, self._providers)

    @staticmethod
    def _build_provider_registry(settings: Settings) -> ProviderRegistry:
        registry = ProviderRegistry()
        if settings.mineru_api_base_url and settings.mineru_api_key:
            registry.register(
                MinerUPdfClient(
                    MinerUPdfConfig(
                        base_url=settings.mineru_api_base_url,
                        api_key=settings.mineru_api_key,
                        model_version=settings.mineru_model_version,
                        timeout_seconds=settings.mineru_timeout_seconds,
                        poll_seconds=settings.mineru_poll_seconds,
                    )
                )
            )
        if settings.qwen_api_base_url and settings.qwen_api_key and settings.qwen_vision_model:
            registry.register(
                QwenVisionClient(
                    QwenVisionConfig(
                        base_url=settings.qwen_api_base_url,
                        api_key=settings.qwen_api_key,
                        model=settings.qwen_vision_model,
                        timeout_seconds=settings.qwen_vision_timeout_seconds,
                    )
                )
            )
        return registry
