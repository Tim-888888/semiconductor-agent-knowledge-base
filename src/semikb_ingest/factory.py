"""Composition helpers for the adapter package; no parser has a hidden fallback."""

from __future__ import annotations

from semikb_ingest.assets import ProcessPayloadStore
from semikb_ingest.parsers import ParserRegistry
from semikb_ingest.parsers.docx import DocxStructuredParser
from semikb_ingest.parsers.image import ImageVlmParser
from semikb_ingest.parsers.pdf import PdfMinerUParser
from semikb_ingest.parsers.pptx import PptxStructuredParser
from semikb_ingest.parsers.tabular import CsvStructuredParser, XlsxStructuredParser
from semikb_ingest.parsers.textual import (
    HtmlStructuredParser,
    MarkdownStructuredParser,
    TextStructuredParser,
)
from semikb_ingest.providers import PdfProvider, ProviderRegistry, VisionProvider
from semikb_ingest.service import IngestDispatcher


def build_parser_registry(
    payload_store: ProcessPayloadStore,
    provider_registry: ProviderRegistry | None = None,
) -> ParserRegistry:
    registry = ParserRegistry()
    for parser in (
        MarkdownStructuredParser(),
        TextStructuredParser(),
        HtmlStructuredParser(),
        DocxStructuredParser(payload_store),
        XlsxStructuredParser(),
        CsvStructuredParser(),
        PptxStructuredParser(payload_store),
    ):
        registry.register(parser)

    if provider_registry is not None:
        mineru = provider_registry.get("mineru")
        if mineru is not None:
            if not isinstance(mineru, PdfProvider):
                raise TypeError("The registered mineru provider does not implement PdfProvider.")
            registry.register(PdfMinerUParser(payload_store, mineru))
        vision = provider_registry.get("qwen-vl")
        if vision is not None:
            if not isinstance(vision, VisionProvider):
                raise TypeError(
                    "The registered qwen-vl provider does not implement VisionProvider."
                )
            registry.register(ImageVlmParser(payload_store, vision))
    return registry


def build_dispatcher(
    payload_store: ProcessPayloadStore,
    provider_registry: ProviderRegistry | None = None,
) -> IngestDispatcher:
    return IngestDispatcher(build_parser_registry(payload_store, provider_registry))
