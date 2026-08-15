"""Stable document parsing boundary for the semiconductor knowledge base."""

from semikb_ingest.errors import IngestError, IngestErrorCode
from semikb_ingest.factory import build_dispatcher, build_parser_registry
from semikb_ingest.models import (
    BinaryPayload,
    CaptionSource,
    ChunkDraft,
    ChunkType,
    ImageAssetDraft,
    ParsedDocument,
    ParseMetrics,
    ParseProvenance,
    ParseWarning,
    SourceFormat,
    SourceLocation,
    TableAssetDraft,
)
from semikb_ingest.routing import FormatRouter, ResolvedRoute, RoutingPolicy
from semikb_ingest.service import IngestDispatcher

__all__ = [
    "BinaryPayload",
    "build_dispatcher",
    "build_parser_registry",
    "CaptionSource",
    "ChunkDraft",
    "ChunkType",
    "FormatRouter",
    "ImageAssetDraft",
    "IngestDispatcher",
    "IngestError",
    "IngestErrorCode",
    "ParseMetrics",
    "ParseProvenance",
    "ParseWarning",
    "ParsedDocument",
    "ResolvedRoute",
    "RoutingPolicy",
    "SourceFormat",
    "SourceLocation",
    "TableAssetDraft",
]
