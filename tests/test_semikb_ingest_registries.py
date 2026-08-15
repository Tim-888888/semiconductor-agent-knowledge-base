from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest

from semikb_ingest.errors import IngestError, IngestErrorCode
from semikb_ingest.models import (
    ParsedDocument,
    ParseProvenance,
    SourceFormat,
)
from semikb_ingest.parsers import ParseRequest, ParserRegistry
from semikb_ingest.providers import ProviderRegistry
from semikb_ingest.routing import FormatRouter
from semikb_ingest.service import IngestDispatcher


@dataclass
class FakeParser:
    parser_id: str
    source_format: SourceFormat

    def parse(self, request: ParseRequest) -> ParsedDocument:
        return ParsedDocument(
            source_format=self.source_format,
            normalized_markdown=request.content.decode("utf-8"),
            provenance=ParseProvenance(
                parser_name=self.parser_id,
                parser_version="test-v1",
                provider_name=request.route.provider,
                provider_version="test-v1",
                upstream_project=None,
                upstream_commit=None,
                source_filename=request.filename,
                source_media_type=request.route.declared_media_type or "text/plain",
                source_sha256=request.source_sha256,
                detected_format=request.route.source_format.value,
            ),
        )


@dataclass
class FakeProvider:
    provider_name: str
    provider_version: str


def test_parser_registry_has_no_default_or_cross_format_fallback() -> None:
    registry = ParserRegistry()
    markdown = FakeParser("markdown-structured-v1", SourceFormat.MARKDOWN)
    registry.register(markdown)

    markdown_route = FormatRouter().resolve("sop.md", b"# SOP", "text/markdown")
    pdf_route = FormatRouter().resolve("sop.pdf", b"%PDF-1.7", "application/pdf")

    assert registry.require(markdown_route) is markdown
    with pytest.raises(IngestError) as captured:
        registry.require(pdf_route)
    assert captured.value.code is IngestErrorCode.PARSER_NOT_CONFIGURED


def test_parser_registry_rejects_duplicate_format_registration() -> None:
    registry = ParserRegistry()
    registry.register(FakeParser("markdown-structured-v1", SourceFormat.MARKDOWN))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(FakeParser("other-parser", SourceFormat.MARKDOWN))


def test_provider_registry_requires_an_explicit_named_provider() -> None:
    registry = ProviderRegistry()
    provider = FakeProvider("mineru", "precision-v1")
    registry.register(provider)

    assert registry.require("mineru") is provider
    with pytest.raises(IngestError) as captured:
        registry.require("qwen-vl")
    assert captured.value.code is IngestErrorCode.PARSER_NOT_CONFIGURED


def test_dispatcher_is_the_single_parse_boundary() -> None:
    registry = ParserRegistry()
    registry.register(FakeParser("markdown-structured-v1", SourceFormat.MARKDOWN))
    dispatcher = IngestDispatcher(registry)
    content = b"# Chamber clean"

    parsed = dispatcher.parse(
        "sop.md",
        content,
        declared_media_type="text/markdown",
        correlation_id="job-1",
    )

    assert parsed.source_format is SourceFormat.MARKDOWN
    assert parsed.provenance.source_sha256 == hashlib.sha256(content).hexdigest()


def test_dispatcher_rejects_parser_output_from_another_format() -> None:
    class WrongFormatParser(FakeParser):
        def parse(self, request: ParseRequest) -> ParsedDocument:
            parsed = super().parse(request)
            return parsed.model_copy(update={"source_format": SourceFormat.TEXT})

    registry = ParserRegistry()
    registry.register(WrongFormatParser("markdown-structured-v1", SourceFormat.MARKDOWN))

    with pytest.raises(IngestError) as captured:
        IngestDispatcher(registry).parse(
            "sop.md",
            b"# SOP",
            declared_media_type="text/markdown",
            correlation_id="job-2",
        )
    assert captured.value.code is IngestErrorCode.CONTRACT_VIOLATION


def test_dispatcher_requires_a_correlation_id() -> None:
    registry = ParserRegistry()
    registry.register(FakeParser("markdown-structured-v1", SourceFormat.MARKDOWN))

    with pytest.raises(ValueError, match="correlation_id"):
        IngestDispatcher(registry).parse(
            "sop.md",
            b"# SOP",
            declared_media_type="text/markdown",
            correlation_id=" ",
        )
