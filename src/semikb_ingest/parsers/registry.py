"""Parser contracts that never silently fall back to another format."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from semikb_ingest.errors import IngestError, IngestErrorCode
from semikb_ingest.models import ParsedDocument, SourceFormat
from semikb_ingest.routing import ResolvedRoute


@dataclass(frozen=True, slots=True)
class ParseRequest:
    filename: str
    content: bytes
    source_sha256: str
    route: ResolvedRoute
    correlation_id: str


@runtime_checkable
class DocumentParser(Protocol):
    @property
    def parser_id(self) -> str: ...

    @property
    def source_format(self) -> SourceFormat: ...

    def parse(self, request: ParseRequest) -> ParsedDocument: ...


class ParserRegistry:
    def __init__(self) -> None:
        self._parsers: dict[SourceFormat, DocumentParser] = {}

    def register(self, parser: DocumentParser) -> None:
        if not isinstance(parser, DocumentParser):
            raise TypeError("Parser does not implement the DocumentParser protocol.")
        if parser.source_format in self._parsers:
            raise ValueError(f"A parser for {parser.source_format.value!r} is already registered.")
        self._parsers[parser.source_format] = parser

    def require(self, route: ResolvedRoute) -> DocumentParser:
        parser = self._parsers.get(route.source_format)
        if parser is None or parser.parser_id != route.parser_id:
            raise IngestError(
                IngestErrorCode.PARSER_NOT_CONFIGURED,
                f"Parser {route.parser_id!r} is not configured for this format.",
            )
        return parser
