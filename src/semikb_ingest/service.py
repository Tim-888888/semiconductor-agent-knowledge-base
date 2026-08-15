"""Single public dispatch boundary used by the governed ingestion service."""

from __future__ import annotations

import hashlib

from semikb_ingest.errors import IngestError, IngestErrorCode
from semikb_ingest.models import ParsedDocument
from semikb_ingest.parsers import ParseRequest, ParserRegistry
from semikb_ingest.routing import FormatRouter, ResolvedRoute


class IngestDispatcher:
    """Resolve a format, select its exact parser, and validate the public output."""

    def __init__(
        self,
        parser_registry: ParserRegistry,
        router: FormatRouter | None = None,
    ) -> None:
        self._parsers = parser_registry
        self._router = router or FormatRouter()

    def resolve(
        self,
        filename: str,
        content: bytes,
        declared_media_type: str | None = None,
    ) -> ResolvedRoute:
        return self._router.resolve(filename, content, declared_media_type)

    def parse(
        self,
        filename: str,
        content: bytes,
        *,
        correlation_id: str,
        declared_media_type: str | None = None,
    ) -> ParsedDocument:
        if not correlation_id.strip():
            raise ValueError("A non-empty correlation_id is required for parser audit.")
        route = self.resolve(filename, content, declared_media_type)
        parser = self._parsers.require(route)
        source_sha256 = hashlib.sha256(content).hexdigest()
        result = parser.parse(
            ParseRequest(
                filename=filename,
                content=content,
                source_sha256=source_sha256,
                route=route,
                correlation_id=correlation_id,
            )
        )
        if not isinstance(result, ParsedDocument):
            self._raise_contract_violation("Parser did not return a ParsedDocument.")
        if result.source_format is not route.source_format:
            self._raise_contract_violation("Parser output format does not match its route.")
        if result.provenance.parser_name != route.parser_id:
            self._raise_contract_violation("Parser output provenance does not match its route.")
        if result.provenance.provider_name != route.provider:
            self._raise_contract_violation("Provider output provenance does not match its route.")
        if result.provenance.source_filename != filename:
            self._raise_contract_violation("Parser output filename does not match the input.")
        if result.provenance.source_sha256 != source_sha256:
            self._raise_contract_violation("Parser output source hash does not match the input.")
        if result.provenance.detected_format != route.source_format.value:
            self._raise_contract_violation("Parser detected format does not match its route.")
        return result

    @staticmethod
    def _raise_contract_violation(message: str) -> None:
        raise IngestError(IngestErrorCode.CONTRACT_VIOLATION, message)
