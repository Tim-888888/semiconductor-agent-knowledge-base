"""Shared adapter helpers with no storage or governance dependencies."""

from __future__ import annotations

import html
import re
import time
from collections.abc import Iterable, Sequence

from semikb_ingest.chunking import StructuredBlockChunker
from semikb_ingest.errors import IngestError, IngestErrorCode
from semikb_ingest.models import (
    ImageAssetDraft,
    ParsedDocument,
    ParseMetrics,
    ParseProvenance,
    ParseWarning,
    SourceFormat,
    TableAssetDraft,
)
from semikb_ingest.parsers.registry import ParseRequest
from semikb_ingest.structure import BlockKind, StructuredBlock

KNOWHERE_PROJECT = "Ontos-AI/knowhere"
KNOWHERE_COMMIT = "2e4eb5846249d273b11902ee00f26db949e45b38"


def decode_text(content: bytes) -> tuple[str, str | None]:
    """Decode common UTF and Chinese text encodings without lossy replacement."""

    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(encoding), None if encoding.startswith("utf-8") else encoding
        except UnicodeDecodeError:
            continue
    raise IngestError(
        IngestErrorCode.TEXT_DECODE_FAILED,
        "The document text encoding could not be decoded safely.",
    )


def normalized_text(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"[ \t\f\v]+", " ", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def matrix_to_markdown(rows: Sequence[Sequence[object]]) -> tuple[str, tuple[str, ...]]:
    matrix = [[normalized_text(value) for value in row] for row in rows]
    width = max((len(row) for row in matrix), default=0)
    if not matrix or width == 0:
        return "", ()
    padded = [row + [""] * (width - len(row)) for row in matrix]
    headers = tuple(value or f"Column {index + 1}" for index, value in enumerate(padded[0]))

    def escaped(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", "<br>")

    lines = [
        "| " + " | ".join(escaped(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(escaped(value) for value in row) + " |" for row in padded[1:])
    return "\n".join(lines), headers


def matrix_to_html(rows: Sequence[Sequence[object]]) -> str:
    matrix = [[normalized_text(value) for value in row] for row in rows]
    width = max((len(row) for row in matrix), default=0)
    if not matrix or width == 0:
        return ""
    padded = [row + [""] * (width - len(row)) for row in matrix]
    head = "".join(f"<th>{html.escape(value)}</th>" for value in padded[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
        for row in padded[1:]
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def blocks_to_markdown(blocks: Iterable[StructuredBlock]) -> str:
    rendered: list[str] = []
    for block in blocks:
        if not block.text.strip():
            continue
        if block.kind is BlockKind.HEADING:
            level = int(block.metadata.get("level", len(block.heading_path) or 1))
            rendered.append(f"{'#' * max(1, min(level, 6))} {block.text}")
        elif block.kind is BlockKind.LIST_ITEM:
            rendered.append(f"- {block.text}")
        elif block.kind is BlockKind.CODE:
            language = str(block.metadata.get("language", ""))
            rendered.append(f"```{language}\n{block.text}\n```")
        elif block.kind is BlockKind.IMAGE:
            asset_id = block.image_asset_ids[0] if block.image_asset_ids else "unknown"
            rendered.append(f"![{block.text}](asset:{asset_id})")
        else:
            rendered.append(block.text)
    return "\n\n".join(rendered).strip()


def complete_document(
    *,
    request: ParseRequest,
    source_format: SourceFormat,
    parser_name: str,
    parser_version: str,
    provider_name: str,
    provider_version: str,
    blocks: Sequence[StructuredBlock],
    images: Sequence[ImageAssetDraft] = (),
    tables: Sequence[TableAssetDraft] = (),
    warnings: Sequence[ParseWarning] = (),
    detected_title: str | None = None,
    detected_language: str | None = None,
    normalized_markdown: str | None = None,
    pages: int = 0,
    slides: int = 0,
    sheets: int = 0,
    started_at: float | None = None,
    reference_knowhere: bool = False,
) -> ParsedDocument:
    chunker = StructuredBlockChunker()
    chunks = chunker.chunk(blocks)
    chunk_by_image = {
        asset_id: tuple(chunk.draft_id for chunk in chunks if asset_id in chunk.image_asset_ids)
        for asset_id in {asset.asset_id for asset in images}
    }
    chunk_by_table = {
        asset_id: tuple(chunk.draft_id for chunk in chunks if asset_id in chunk.table_asset_ids)
        for asset_id in {asset.asset_id for asset in tables}
    }
    linked_images = tuple(
        asset.model_copy(update={"related_chunk_draft_ids": chunk_by_image[asset.asset_id]})
        for asset in images
    )
    linked_tables = tuple(
        asset.model_copy(update={"related_chunk_draft_ids": chunk_by_table[asset.asset_id]})
        for asset in tables
    )
    markdown = (
        normalized_markdown if normalized_markdown is not None else blocks_to_markdown(blocks)
    )
    if not chunks and not markdown.strip():
        raise IngestError(
            IngestErrorCode.EMPTY_PARSE_RESULT,
            "The parser did not find indexable document content.",
        )
    duration_ms = 0.0 if started_at is None else (time.monotonic() - started_at) * 1000
    return ParsedDocument(
        source_format=source_format,
        normalized_markdown=markdown.strip(),
        detected_title=detected_title,
        detected_language=detected_language,
        chunks=chunks,
        images=linked_images,
        tables=linked_tables,
        provenance=ParseProvenance(
            parser_name=parser_name,
            parser_version=parser_version,
            provider_name=provider_name,
            provider_version=provider_version,
            upstream_project=KNOWHERE_PROJECT if reference_knowhere else None,
            upstream_commit=KNOWHERE_COMMIT if reference_knowhere else None,
            source_filename=request.filename,
            source_media_type=request.route.declared_media_type or "application/octet-stream",
            source_sha256=request.source_sha256,
            detected_format=source_format.value,
        ),
        warnings=tuple(warnings),
        metrics=ParseMetrics(
            pages=pages,
            slides=slides,
            sheets=sheets,
            chunks=len(chunks),
            images=len(linked_images),
            tables=len(linked_tables),
            duration_ms=duration_ms,
        ),
    )
