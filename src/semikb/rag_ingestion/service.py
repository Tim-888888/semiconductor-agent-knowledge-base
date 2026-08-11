"""Synthetic ingestion flow with explicit staging and publication semantics."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from semikb.config import Settings
from semikb.contracts.models import (
    ApprovalStatus,
    Chunk,
    ChunkType,
    DocumentLifecycle,
    DocumentRevision,
    ImageAsset,
    IngestionJob,
    IngestionStatus,
    ObjectRef,
)
from semikb.rag_ingestion.chunker import chunk_markdown
from semikb.rag_ingestion.mineru import MinerUPrecisionClient
from semikb.storage.memory import DemoStore


class IngestionService:
    """Runs a document through validation, chunking, staging, and publication."""

    def __init__(self, store: DemoStore, settings: Settings | None = None) -> None:
        self.store = store
        self.settings = settings or Settings()
        self._payloads: dict[str, dict[str, Any]] = {}

    def ingest_payload(
        self,
        payload: dict[str, Any],
        created_by: str = "demo_admin",
        *,
        source_hash: str | None = None,
        filename: str | None = None,
    ) -> IngestionJob:
        source_hash = source_hash or hashlib.sha256(payload["content"].encode("utf-8")).hexdigest()
        key = f"{payload['document_id']}:{payload['revision']}:{source_hash}"
        job = IngestionJob(
            document_id=payload["document_id"],
            revision=payload["revision"],
            filename=filename or f"{payload['document_id']}-{payload['revision']}.md",
            file_type="markdown",
            source_hash=source_hash,
            idempotency_key=key,
            created_by=created_by,
        )
        job = self.store.create_or_get_job(job)
        self._payloads[job.job_id] = payload
        if job.status is IngestionStatus.PUBLISHED:
            return job
        return self._run(job.job_id)

    def ingest_file(
        self,
        filename: str,
        content: bytes,
        metadata: dict[str, Any],
        created_by: str = "demo_admin",
    ) -> IngestionJob:
        """Normalize an uploaded source file before the shared ingestion flow."""

        source_hash = hashlib.sha256(content).hexdigest()
        suffix = Path(filename).suffix.lower()
        if suffix in {".md", ".markdown", ".txt"}:
            normalized_markdown = content.decode("utf-8")
        else:
            normalized_markdown = MinerUPrecisionClient(self.settings).parse_file(
                filename, content, f"{metadata['document_id']}-{metadata['revision']}-{source_hash[:12]}"
            )
        payload = {
            **metadata,
            "content": normalized_markdown,
            "source_filename": filename,
        }
        return self.ingest_payload(payload, created_by, source_hash=source_hash, filename=filename)

    def retry(self, job_id: str) -> IngestionJob:
        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        payload = self._payloads.get(job_id)
        if payload is None:
            raise ValueError("No replayable payload is available for this job.")
        if job.status is not IngestionStatus.FAILED:
            return job
        job.attempt += 1
        job.error_code = None
        job.safe_error_summary = None
        job.failed_stage = None
        return self._run(job_id)

    def seed_demo_corpus(self, fixture_path: Path) -> list[IngestionJob]:
        corpus = json.loads(fixture_path.read_text(encoding="utf-8"))
        return [self.ingest_payload(payload) for payload in corpus["documents"]]

    def _run(self, job_id: str) -> IngestionJob:
        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        payload = self._payloads[job_id]
        try:
            self.store.update_job(job_id, IngestionStatus.VALIDATING, "Validated metadata and source hash.", 10)
            self._validate(payload)
            self.store.update_job(job_id, IngestionStatus.PARSING, "Parsed normalized Markdown content.", 30)
            document, chunks, images = self._build_records(payload, job.source_hash)
            self.store.update_job(job_id, IngestionStatus.QUALITY_CHECK, "Passed required metadata and image checks.", 55)
            self._quality_check(document, chunks, images)
            self.store.update_job(job_id, IngestionStatus.EMBEDDING, "Prepared dense and sparse embedding payloads.", 75)
            self.store.add_document(document, chunks, images)
            job.chunks_count = len(chunks)
            job.images_count = len(images)
            self.store.update_job(job_id, IngestionStatus.STAGED, "Staged document and verified index release.", 90)
            if document.lifecycle is DocumentLifecycle.PUBLISHED:
                self.store.publish_document(document.document_id, document.revision, document.index_version)
                message = "Published to active knowledge index."
            else:
                message = "Processed inactive revision; it remains outside active retrieval."
            return self.store.update_job(job_id, IngestionStatus.PUBLISHED, message, 100)
        except (KeyError, TypeError, ValueError) as exc:
            return self.store.update_job(
                job_id,
                IngestionStatus.FAILED,
                "Document remains unpublished. Fix metadata or parser input before retrying.",
                job.progress,
                error_code=type(exc).__name__.upper(),
            )

    @staticmethod
    def _validate(payload: dict[str, Any]) -> None:
        required = {"document_id", "revision", "title", "content", "document_type"}
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"Missing fields: {', '.join(missing)}")
        if not payload["content"].strip():
            raise ValueError("Content is empty.")

    @staticmethod
    def _quality_check(document: DocumentRevision, chunks: list[Chunk], images: list[ImageAsset]) -> None:
        if document.approval_status is not ApprovalStatus.APPROVED:
            raise ValueError("Only approved synthetic documents can be published.")
        if not chunks:
            raise ValueError("No semantic chunks were generated.")
        for image in images:
            if not image.caption.strip():
                raise ValueError("Image caption is required for text-to-image retrieval.")

    @staticmethod
    def _build_records(
        payload: dict[str, Any], source_hash: str
    ) -> tuple[DocumentRevision, list[Chunk], list[ImageAsset]]:
        lifecycle = DocumentLifecycle(payload.get("lifecycle", "staged"))
        source_filename = payload.get("source_filename") or f"{payload['document_id']}-{payload['revision']}.md"
        source_ref = ObjectRef(
            bucket="semikb-raw",
            object_key=(
                f"documents/{payload['document_id']}/{payload['revision']}/source/"
                f"{source_hash}/{source_filename}"
            ),
            content_type="text/markdown",
            sha256=source_hash,
        )
        shared = {
            key: value
            for key in ("fab", "product", "process_layer", "tool_id", "chamber", "recipe_id", "recipe_version")
            if (value := payload.get(key)) is not None
        }
        document = DocumentRevision(
            document_id=payload["document_id"],
            revision=payload["revision"],
            title=payload["title"],
            document_type=payload["document_type"],
            approval_status=ApprovalStatus(payload.get("approval_status", "approved")),
            lifecycle=lifecycle,
            effective_at=(
                datetime.fromisoformat(payload["effective_at"])
                if payload.get("effective_at")
                else datetime.now(UTC)
            ),
            expires_at=datetime.fromisoformat(payload["expires_at"]) if payload.get("expires_at") else None,
            supersedes_revision=payload.get("supersedes_revision"),
            source_hash=source_hash,
            source_ref=source_ref,
            source_kind=payload.get("source_kind", "user_upload"),
            source_uri=payload.get("source_uri", f"upload://{source_filename}"),
            source_license=payload.get("source_license", "internal"),
            access_scope_key=payload.get("access_scope_key", "demo_engineering"),
            **shared,
        )
        chunks: list[Chunk] = []
        for number, (path, text) in enumerate(chunk_markdown(payload["content"]), start=1):
            chunk_id = f"{document.document_id}-{document.revision}-{number:03d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    revision=document.revision,
                    chunk_text=text,
                    title_path=path,
                    page_or_section=" > ".join(path) or "正文",
                    approval_status=document.approval_status,
                    lifecycle=document.lifecycle,
                    effective_at=document.effective_at,
                    expires_at=document.expires_at,
                    access_scope_key=document.access_scope_key,
                    **shared,
                )
            )
        images: list[ImageAsset] = []
        for index, image_payload in enumerate(payload.get("images", []), start=1):
            image_id = image_payload["image_id"]
            asset_ref = ObjectRef(
                bucket="semikb-derived",
                object_key=image_payload.get(
                    "object_key",
                    f"documents/{document.document_id}/{document.revision}/assets/{image_id}/original.png",
                ),
                content_type=image_payload.get("content_type", "image/png"),
                sha256=image_payload.get("sha256", hashlib.sha256(image_id.encode("utf-8")).hexdigest()),
            )
            image = ImageAsset(
                image_id=image_id,
                document_id=document.document_id,
                revision=document.revision,
                parent_chunk_id=chunks[0].chunk_id if chunks else None,
                object_ref=asset_ref,
                image_type=image_payload["image_type"],
                caption=image_payload["caption"],
                ocr_text=image_payload.get("ocr_text", ""),
                detection_summary=image_payload.get("detection_summary", ""),
                source_page=image_payload.get("source_page", ""),
                related_case_id=image_payload.get("related_case_id"),
                demo_source_path=image_payload.get("source_path"),
                access_scope_key=document.access_scope_key,
                approval_status=document.approval_status,
                lifecycle=document.lifecycle,
                effective_at=document.effective_at,
                expires_at=document.expires_at,
            )
            images.append(image)
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}-{document.revision}-IMAGE-{index:03d}",
                    document_id=document.document_id,
                    revision=document.revision,
                    parent_chunk_id=image.parent_chunk_id,
                    chunk_type=ChunkType.IMAGE_TEXT,
                    chunk_text=" ".join(
                        part for part in (image.caption, image.ocr_text, image.detection_summary) if part
                    ),
                    title_path=[document.title, "图文证据"],
                    page_or_section=image.source_page or "图像附件",
                    approval_status=document.approval_status,
                    lifecycle=document.lifecycle,
                    effective_at=document.effective_at,
                    expires_at=document.expires_at,
                    access_scope_key=document.access_scope_key,
                    image_ids=[image.image_id],
                    metadata={"image_type": image.image_type, "related_case_id": image.related_case_id},
                    **shared,
                )
            )
        return document, chunks, images
