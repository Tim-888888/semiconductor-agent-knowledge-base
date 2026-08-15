"""Document parser protocols and explicit adapter registry."""

from semikb_ingest.parsers.registry import (
    DocumentParser,
    ParseRequest,
    ParserRegistry,
)

__all__ = ["DocumentParser", "ParseRequest", "ParserRegistry"]
