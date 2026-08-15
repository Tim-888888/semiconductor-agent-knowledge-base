"""Pure mapping from parser drafts into governed business records."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from semikb.contracts.models import (
    ApprovalStatus,
    Chunk,
    ChunkType,
    DocumentLifecycle,
    DocumentRevision,
    ImageAsset,
    IngestionJob,
    ObjectRef,
    TableAsset,
)
from semikb_ingest.models import ParsedDocument, SourceLocation


def build_governed_records(
    *,
    metadata: dict[str, Any],
    parsed: ParsedDocument,
    source_ref: ObjectRef,
    parsed_ref: ObjectRef,
    image_payloads: list[dict[str, Any]],
    table_payloads: list[dict[str, Any]],
    job: IngestionJob,
    chunker_version: str,
) -> tuple[DocumentRevision, list[Chunk], list[ImageAsset], list[TableAsset]]:
    shared = _shared_scope(metadata)
    provenance = parsed.provenance
    effective_at = (
        datetime.fromisoformat(metadata["effective_at"])
        if metadata.get("effective_at")
        else datetime.now(UTC)
    )
    expires_at = (
        datetime.fromisoformat(metadata["expires_at"])
        if metadata.get("expires_at")
        else None
    )
    document = DocumentRevision(
        document_id=metadata["document_id"],
        revision=metadata["revision"],
        title=metadata["title"],
        document_type=metadata["document_type"],
        approval_status=ApprovalStatus(metadata.get("approval_status", "approved")),
        lifecycle=DocumentLifecycle.STAGED,
        effective_at=effective_at,
        expires_at=expires_at,
        supersedes_revision=metadata.get("supersedes_revision"),
        source_hash=job.source_hash,
        source_ref=source_ref,
        parsed_ref=parsed_ref,
        source_kind=metadata.get("source_kind", "user_upload"),
        source_uri=metadata.get("source_uri", f"upload://{job.filename}"),
        source_license=metadata.get("source_license", "internal"),
        source_id=metadata.get("source_id"),
        source_manifest_version=metadata.get("source_manifest_version"),
        dataset_version=metadata.get("dataset_version"),
        source_license_status=metadata.get("source_license_status"),
        redistribution_policy=metadata.get("redistribution_policy"),
        access_scope_key=metadata.get("access_scope_key", "demo_engineering"),
        parse_contract_version=parsed.contract_version,
        parser_name=provenance.parser_name,
        parser_version=provenance.parser_version,
        provider_name=provenance.provider_name,
        provider_version=provenance.provider_version,
        upstream_project=provenance.upstream_project,
        upstream_commit=provenance.upstream_commit,
        detected_title=parsed.detected_title,
        detected_language=parsed.detected_language,
        parse_warning_codes=[warning.code for warning in parsed.warnings],
        parse_metrics=parsed.metrics.model_dump(mode="json"),
        chunker_version=chunker_version,
        embedding_version=job.embedding_version,
        index_version=job.index_version,
        **shared,
    )
    chunk_id_by_draft = {
        draft.draft_id: scoped_id(
            document.document_id,
            document.revision,
            "",
            index,
        )
        for index, draft in enumerate(parsed.chunks, start=1)
    }
    image_id_by_asset = {
        str(item["source_asset_id"]): str(item["image_id"])
        for item in image_payloads
        if item.get("related_chunk_draft_ids")
    }
    table_id_by_asset = {
        str(item["source_asset_id"]): str(item["table_id"])
        for item in table_payloads
    }
    chunks = _build_chunks(
        parsed,
        document,
        shared,
        chunk_id_by_draft,
        image_id_by_asset,
        table_id_by_asset,
    )
    images = _build_images(
        image_payloads,
        document,
        chunks,
        chunk_id_by_draft,
    )
    _append_unreferenced_image_chunks(chunks, images, document, shared)
    tables = _build_tables(
        table_payloads,
        document,
        chunk_id_by_draft,
    )
    return document, chunks, images, tables


def _build_chunks(
    parsed: ParsedDocument,
    document: DocumentRevision,
    shared: dict[str, Any],
    chunk_id_by_draft: dict[str, str],
    image_id_by_asset: dict[str, str],
    table_id_by_asset: dict[str, str],
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for draft in parsed.chunks:
        chunks.append(
            Chunk(
                chunk_id=chunk_id_by_draft[draft.draft_id],
                document_id=document.document_id,
                revision=document.revision,
                parent_chunk_id=(
                    chunk_id_by_draft.get(draft.parent_draft_id)
                    if draft.parent_draft_id
                    else None
                ),
                chunk_type=ChunkType(draft.chunk_type.value),
                chunk_text=draft.text,
                title_path=list(draft.title_path),
                page_or_section=location_label(draft.location),
                approval_status=document.approval_status,
                lifecycle=DocumentLifecycle.STAGED,
                effective_at=document.effective_at,
                expires_at=document.expires_at,
                access_scope_key=document.access_scope_key,
                image_ids=[image_id_by_asset[item] for item in draft.image_asset_ids],
                table_ids=[table_id_by_asset[item] for item in draft.table_asset_ids],
                metadata={
                    **draft.metadata,
                    "source_uri": document.source_uri,
                    "document_title": document.title,
                    "source_location": draft.location.model_dump(mode="json"),
                    "parse_contract_version": parsed.contract_version,
                    "provider_name": parsed.provenance.provider_name,
                    "provider_version": parsed.provenance.provider_version,
                },
                parser_name=document.parser_name,
                parser_version=document.parser_version,
                upstream_commit=document.upstream_commit,
                chunker_version=document.chunker_version,
                embedding_version=document.embedding_version,
                index_version=document.index_version,
                **shared,
            )
        )
    return chunks


def _build_images(
    payloads: list[dict[str, Any]],
    document: DocumentRevision,
    chunks: list[Chunk],
    chunk_id_by_draft: dict[str, str],
) -> list[ImageAsset]:
    images: list[ImageAsset] = []
    for payload in payloads:
        parent_chunk_id = _first_related_chunk(
            payload.get("related_chunk_draft_ids", []),
            chunk_id_by_draft,
        ) or (chunks[0].chunk_id if chunks else None)
        images.append(
            ImageAsset(
                image_id=payload["image_id"],
                document_id=document.document_id,
                revision=document.revision,
                parent_chunk_id=parent_chunk_id,
                object_ref=ObjectRef.model_validate(payload["object_ref"]),
                image_type=payload["image_type"],
                caption=payload["caption"],
                caption_source=payload.get("caption_source", "human"),
                caption_confidence=payload.get("caption_confidence", 1.0),
                ocr_text=payload.get("ocr_text", ""),
                detection_summary=payload.get("detection_summary", ""),
                source_page=payload.get("source_page", ""),
                source_asset_id=payload.get("source_asset_id"),
                source_location=payload.get("source_location", {}),
                parser_name=payload.get("parser_name", document.parser_name),
                parser_version=payload.get("parser_version", document.parser_version),
                provider_name=payload.get("provider_name"),
                provider_version=payload.get("provider_version"),
                related_case_id=payload.get("related_case_id"),
                demo_source_path=payload.get("source_path"),
                access_scope_key=document.access_scope_key,
                approval_status=document.approval_status,
                lifecycle=DocumentLifecycle.STAGED,
                effective_at=document.effective_at,
                expires_at=document.expires_at,
            )
        )
    return images


def _append_unreferenced_image_chunks(
    chunks: list[Chunk],
    images: list[ImageAsset],
    document: DocumentRevision,
    shared: dict[str, Any],
) -> None:
    referenced = {image_id for chunk in chunks for image_id in chunk.image_ids}
    for image_index, image in enumerate(images, start=1):
        if image.image_id in referenced:
            continue
        image_text = " ".join(
            value
            for value in (image.caption, image.ocr_text, image.detection_summary)
            if value
        ).strip()
        if not image_text:
            continue
        chunks.append(
            Chunk(
                chunk_id=scoped_id(
                    document.document_id,
                    document.revision,
                    "IMAGE",
                    image_index,
                ),
                document_id=document.document_id,
                revision=document.revision,
                parent_chunk_id=image.parent_chunk_id,
                chunk_type=ChunkType.IMAGE_TEXT,
                chunk_text=image_text,
                title_path=[document.title, "图文证据"],
                page_or_section=image.source_page or "图像附件",
                approval_status=document.approval_status,
                lifecycle=DocumentLifecycle.STAGED,
                effective_at=document.effective_at,
                expires_at=document.expires_at,
                access_scope_key=document.access_scope_key,
                image_ids=[image.image_id],
                metadata={
                    "image_type": image.image_type,
                    "related_case_id": image.related_case_id,
                    "source_uri": document.source_uri,
                    "document_title": document.title,
                },
                parser_name=document.parser_name,
                parser_version=document.parser_version,
                upstream_commit=document.upstream_commit,
                chunker_version=document.chunker_version,
                embedding_version=document.embedding_version,
                index_version=document.index_version,
                **shared,
            )
        )


def _build_tables(
    payloads: list[dict[str, Any]],
    document: DocumentRevision,
    chunk_id_by_draft: dict[str, str],
) -> list[TableAsset]:
    tables: list[TableAsset] = []
    for payload in payloads:
        tables.append(
            TableAsset(
                table_id=payload["table_id"],
                document_id=document.document_id,
                revision=document.revision,
                parent_chunk_id=_first_related_chunk(
                    payload.get("related_chunk_draft_ids", []),
                    chunk_id_by_draft,
                ),
                object_ref=ObjectRef.model_validate(payload["object_ref"]),
                title=payload["title"],
                markdown=payload["markdown"],
                html=payload["html"],
                headers=payload["headers"],
                row_count=payload["row_count"],
                column_count=payload["column_count"],
                source_asset_id=payload["source_asset_id"],
                source_page=payload["source_page"],
                source_location=payload["location"],
                parser_name=document.parser_name,
                parser_version=document.parser_version,
                access_scope_key=document.access_scope_key,
                approval_status=document.approval_status,
                lifecycle=DocumentLifecycle.STAGED,
                effective_at=document.effective_at,
                expires_at=document.expires_at,
            )
        )
    return tables


def _first_related_chunk(
    related_drafts: list[str],
    chunk_id_by_draft: dict[str, str],
) -> str | None:
    return next(
        (
            chunk_id_by_draft[draft_id]
            for draft_id in related_drafts
            if draft_id in chunk_id_by_draft
        ),
        None,
    )


def _shared_scope(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key in (
            "fab",
            "product",
            "process_layer",
            "tool_id",
            "chamber",
            "recipe_id",
            "recipe_version",
        )
        if (value := metadata.get(key)) is not None
    }


def location_label(location: SourceLocation) -> str:
    parts: list[str] = []
    if location.section_path:
        parts.append(" > ".join(location.section_path))
    if location.page_number is not None:
        parts.append(f"第 {location.page_number} 页")
    if location.slide_number is not None:
        parts.append(f"Slide {location.slide_number}")
    if location.sheet_name:
        sheet = f"Sheet {location.sheet_name}"
        if location.cell_range:
            sheet = f"{sheet} {location.cell_range}"
        parts.append(sheet)
    elif location.cell_range:
        parts.append(location.cell_range)
    return " / ".join(parts) or "正文"


def scoped_id(
    document_id: str,
    revision: str,
    kind: str,
    index: int,
    *,
    max_length: int = 160,
) -> str:
    marker = f"{kind}-" if kind else ""
    value = f"{document_id}-{revision}-{marker}{index:03d}"
    if len(value) <= max_length:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{value[: max_length - len(digest) - 1]}-{digest}"
