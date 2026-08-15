"""Replayable ingestion flow with explicit staging and publication semantics."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from semikb.config import Settings
from semikb.contracts.models import (
    ApprovalStatus,
    Chunk,
    DocumentLifecycle,
    DocumentRevision,
    ImageAsset,
    IngestionJob,
    IngestionStatus,
    ObjectRef,
    TableAsset,
)
from semikb.rag_ingestion.governed_records import (
    build_governed_records,
    location_label,
    scoped_id,
)
from semikb.rag_ingestion.semikb_adapter import ParsedIngestSession, SemikbIngestAdapter
from semikb.rag_retrieval.encoders import (
    HybridEmbedding,
    HybridEncoder,
    create_hybrid_encoder,
)
from semikb.storage.ingestion import IngestionStore
from semikb_ingest import IngestError, IngestErrorCode
from semikb_ingest.models import CONTRACT_VERSION, ParsedDocument

_CANONICAL_MEDIA_TYPES = {
    ".csv": "text/csv",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".htm": "text/html",
    ".html": "text/html",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".markdown": "text/markdown",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


class IngestionIdempotencyConflictError(ValueError):
    """The same ingestion identity was reused with different replay metadata."""


class IngestionService:
    """Submits replayable jobs and executes their controlled state transitions."""

    def __init__(
        self,
        store: IngestionStore,
        settings: Settings | None = None,
        encoder: HybridEncoder | None = None,
        ingest_adapter: SemikbIngestAdapter | None = None,
    ) -> None:
        self.store = store
        self.settings = settings or Settings()
        self.encoder = encoder or create_hybrid_encoder(self.settings)
        self.ingest_adapter = ingest_adapter or SemikbIngestAdapter(self.settings)

    def submit_payload(
        self,
        payload: dict[str, Any],
        created_by: str = "demo_admin",
    ) -> IngestionJob:
        content = payload.get("content")
        if not isinstance(content, str):
            raise ValueError("Markdown ingestion requires string content.")
        metadata = {key: value for key, value in payload.items() if key != "content"}
        filename = metadata.pop(
            "source_filename",
            f"{payload.get('document_id', 'document')}-{payload.get('revision', 'revision')}.md",
        )
        return self.submit_file(
            filename,
            content.encode("utf-8"),
            metadata,
            created_by,
            content_type="text/markdown",
        )

    def ingest_payload(
        self,
        payload: dict[str, Any],
        created_by: str = "demo_admin",
        *,
        source_hash: str | None = None,
        filename: str | None = None,
    ) -> IngestionJob:
        if source_hash is not None:
            calculated = hashlib.sha256(payload["content"].encode("utf-8")).hexdigest()
            if source_hash != calculated:
                raise ValueError("Provided source hash does not match Markdown content.")
        if filename:
            payload = {**payload, "source_filename": filename}
        job = self.submit_payload(payload, created_by)
        return self.process(job.job_id) if job.status is IngestionStatus.QUEUED else job

    def submit_file(
        self,
        filename: str,
        content: bytes,
        metadata: dict[str, Any],
        created_by: str = "demo_admin",
        *,
        content_type: str | None = None,
    ) -> IngestionJob:
        self._validate_submission(metadata, content)
        source_hash = hashlib.sha256(content).hexdigest()
        suffix = Path(filename).suffix.lower()
        media_type = content_type or _CANONICAL_MEDIA_TYPES.get(
            suffix,
            "application/octet-stream",
        )
        route = self.ingest_adapter.resolve(filename, content, media_type)
        source_ref = self.store.store_source(
            document_id=metadata["document_id"],
            revision=metadata["revision"],
            filename=filename,
            content=content,
            content_type=media_type,
            source_hash=source_hash,
        )
        candidate = IngestionJob(
            document_id=metadata["document_id"],
            revision=metadata["revision"],
            filename=Path(filename).name,
            file_type=route.source_format.value,
            source_hash=source_hash,
            source_ref=source_ref,
            idempotency_key=(
                f"{metadata['document_id']}:{metadata['revision']}:{source_hash}"
            ),
            parse_contract_version=CONTRACT_VERSION,
            parser_name=route.parser_id,
            parser_version="pending",
            provider_name=route.provider,
            chunker_version=self.ingest_adapter.chunker_version,
            embedding_version=(
                "deterministic-demo-v1"
                if self.settings.demo_mode
                else self.settings.embedding_version
            ),
            index_version=self.settings.milvus_index_version,
            created_by=created_by,
        )
        job = self.store.create_or_get_job(candidate)
        replay_payload = {
            "metadata": metadata,
            "filename": Path(filename).name,
            "content_type": media_type,
            "source_ref": source_ref.model_dump(mode="json"),
        }
        existing_payload = self.store.get_replay_payload(job.job_id)
        if existing_payload is not None and existing_payload != replay_payload:
            raise IngestionIdempotencyConflictError(
                "The idempotency key already exists with different ingestion metadata."
            )
        self.store.save_replay_payload(job.job_id, replay_payload)
        return self.store.set_job_artifacts(job.job_id, source_ref=source_ref)

    def ingest_file(
        self,
        filename: str,
        content: bytes,
        metadata: dict[str, Any],
        created_by: str = "demo_admin",
        *,
        content_type: str | None = None,
    ) -> IngestionJob:
        job = self.submit_file(
            filename,
            content,
            metadata,
            created_by,
            content_type=content_type,
        )
        return self.process(job.job_id) if job.status is IngestionStatus.QUEUED else job

    def process(self, job_id: str) -> IngestionJob:
        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status is IngestionStatus.PUBLISHED:
            return job
        if job.status is not IngestionStatus.QUEUED:
            return job
        replay_payload = self.store.get_replay_payload(job_id)
        if replay_payload is None:
            return self.store.update_job(
                job_id,
                IngestionStatus.FAILED,
                "Document remains unpublished because replay metadata is unavailable.",
                job.progress,
                error_code="REPLAY_PAYLOAD_MISSING",
            )
        return self._run(job, replay_payload)

    def prepare_retry(self, job_id: str) -> IngestionJob:
        if self.store.get_replay_payload(job_id) is None:
            raise ValueError("No replayable payload is available for this job.")
        return self.store.prepare_retry(job_id)

    def mark_queue_submission_failed(self, job_id: str) -> IngestionJob:
        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        return self.store.update_job(
            job_id,
            IngestionStatus.FAILED,
            "Task queue unavailable; the document remains unpublished.",
            job.progress,
            error_code="QUEUE_SUBMISSION_FAILED",
        )

    def retry(self, job_id: str) -> IngestionJob:
        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status is not IngestionStatus.FAILED:
            return job
        queued = self.prepare_retry(job_id)
        return self.process(queued.job_id)

    def get_job(self, job_id: str) -> IngestionJob | None:
        return self.store.get_job(job_id)

    def list_jobs(self) -> list[IngestionJob]:
        return self.store.list_jobs()

    def seed_demo_corpus(self, fixture_path: Path) -> list[IngestionJob]:
        corpus = json.loads(fixture_path.read_text(encoding="utf-8"))
        return [self.ingest_payload(payload) for payload in corpus["documents"]]

    def _run(self, job: IngestionJob, replay_payload: dict[str, Any]) -> IngestionJob:
        metadata = replay_payload["metadata"]
        document_id = job.document_id
        revision = job.revision
        parse_session: ParsedIngestSession | None = None
        try:
            self.store.update_job(
                job.job_id,
                IngestionStatus.VALIDATING,
                "Validated metadata, source hash, and replay reference.",
                10,
            )
            self._validate_submission(metadata, b"replay")
            source_ref = ObjectRef.model_validate(replay_payload["source_ref"])
            source_content = self.store.load_object(source_ref)
            if hashlib.sha256(source_content).hexdigest() != job.source_hash:
                raise ValueError("Replay source hash verification failed.")

            self.store.update_job(
                job.job_id,
                IngestionStatus.PARSING,
                "Parsing source through the governed exact-format adapter.",
                30,
            )
            parse_session = self._parse_source(
                job,
                source_content,
                replay_payload.get("content_type"),
            )
            parsed = parse_session.document
            provenance = parsed.provenance
            self.store.set_job_parse_audit(
                job.job_id,
                parse_contract_version=parsed.contract_version,
                parser_name=provenance.parser_name,
                parser_version=provenance.parser_version,
                provider_name=provenance.provider_name,
                provider_version=provenance.provider_version,
                upstream_project=provenance.upstream_project,
                upstream_commit=provenance.upstream_commit,
                chunker_version=self.ingest_adapter.chunker_version,
                warning_codes=[warning.code for warning in parsed.warnings],
                metrics=parsed.metrics.model_dump(mode="json"),
            )
            markdown_bytes = parsed.normalized_markdown.encode("utf-8")
            parsed_ref = self.store.store_parsed_markdown(
                document_id=document_id,
                revision=revision,
                parser_version=f"{provenance.parser_name}-{provenance.parser_version}",
                source_hash=job.source_hash,
                content=markdown_bytes,
            )
            self.store.set_job_artifacts(job.job_id, parsed_ref=parsed_ref)
            image_payloads = self._materialize_images(
                metadata,
                parsed,
                parse_session,
                source_hash=job.source_hash,
            )
            table_payloads = self._materialize_tables(
                metadata,
                parsed,
                source_hash=job.source_hash,
            )
            document, chunks, images, tables = build_governed_records(
                metadata=metadata,
                parsed=parsed,
                source_ref=source_ref,
                parsed_ref=parsed_ref,
                image_payloads=image_payloads,
                table_payloads=table_payloads,
                job=job,
                chunker_version=self.ingest_adapter.chunker_version,
            )
            target_lifecycle = DocumentLifecycle(metadata.get("lifecycle", "staged"))

            self.store.update_job(
                job.job_id,
                IngestionStatus.QUALITY_CHECK,
                "Checking governance, chunk integrity, assets, and parser provenance.",
                55,
            )
            self._quality_check(document, chunks, images, tables, target_lifecycle)

            self.store.update_job(
                job.job_id,
                IngestionStatus.EMBEDDING,
                (
                    "Generating Dense and Sparse representations with "
                    f"{self.encoder.model_name} / {self.encoder.sparse_encoder_version}."
                ),
                75,
            )
            embeddings = self._encode_chunks(chunks)
            if tables:
                self.store.stage_document(
                    document,
                    chunks,
                    images,
                    embeddings,
                    tables=tables,
                )
            else:
                self.store.stage_document(document, chunks, images, embeddings)
            self.store.set_job_counts(
                job.job_id,
                chunks_count=len(chunks),
                images_count=len(images),
                tables_count=len(tables),
            )
            self.store.update_job(
                job.job_id,
                IngestionStatus.STAGED,
                "Staged MongoDB records, private assets, and versioned Milvus rows.",
                90,
            )
            if target_lifecycle is DocumentLifecycle.PUBLISHED:
                if tables:
                    self.store.publish_document(
                        document,
                        chunks,
                        images,
                        embeddings,
                        tables=tables,
                    )
                else:
                    self.store.publish_document(document, chunks, images, embeddings)
                message = "Published the validated revision to the active knowledge index."
            else:
                self.store.finalize_inactive_document(
                    document.document_id,
                    document.revision,
                    target_lifecycle,
                )
                message = (
                    f"Processed revision as {target_lifecycle.value}; it remains outside active retrieval."
                )
            return self.store.update_job(
                job.job_id,
                IngestionStatus.PUBLISHED,
                message,
                100,
            )
        except Exception as exc:
            try:
                self.store.compensate_document(document_id, revision)
            except Exception:
                pass
            latest = self.store.get_job(job.job_id) or job
            error_code, safe_message = self._failure_details(exc)
            return self.store.update_job(
                job.job_id,
                IngestionStatus.FAILED,
                f"Document remains unpublished. {safe_message}",
                latest.progress,
                error_code=error_code,
            )
        finally:
            if parse_session is not None:
                parse_session.discard_remaining()

    def _parse_source(
        self,
        job: IngestionJob,
        content: bytes,
        declared_media_type: str | None,
    ) -> ParsedIngestSession:
        return self.ingest_adapter.parse(
            job.filename,
            content,
            correlation_id=f"{job.document_id}-{job.revision}-{job.source_hash[:12]}",
            declared_media_type=declared_media_type,
        )

    def _materialize_images(
        self,
        metadata: dict[str, Any],
        parsed: ParsedDocument,
        session: ParsedIngestSession,
        *,
        source_hash: str,
    ) -> list[dict[str, Any]]:
        materialized: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        provenance = parsed.provenance
        for index, image_draft in enumerate(parsed.images, start=1):
            image_id = scoped_id(
                metadata["document_id"],
                metadata["revision"],
                "IMAGE",
                index,
            )
            content = session.pop_image_bytes(image_draft.asset_id)
            object_ref = self.store.store_image_asset(
                document_id=metadata["document_id"],
                revision=metadata["revision"],
                image_id=image_id,
                filename=image_draft.payload.filename,
                content=content,
                content_type=image_draft.payload.content_type,
                source_hash=source_hash,
            )
            self.store.load_object(object_ref)
            materialized.append(
                {
                    "image_id": image_id,
                    "source_asset_id": image_draft.asset_id,
                    "image_type": image_draft.image_type,
                    "caption": image_draft.caption,
                    "caption_source": image_draft.caption_source.value,
                    "caption_confidence": image_draft.caption_confidence,
                    "ocr_text": image_draft.ocr_text,
                    "detection_summary": image_draft.detection_summary,
                    "source_location": image_draft.location.model_dump(mode="json"),
                    "source_page": location_label(image_draft.location),
                    "related_chunk_draft_ids": list(
                        image_draft.related_chunk_draft_ids
                    ),
                    "parser_name": provenance.parser_name,
                    "parser_version": provenance.parser_version,
                    "provider_name": provenance.provider_name,
                    "provider_version": provenance.provider_version,
                    "object_ref": object_ref.model_dump(mode="json"),
                }
            )
            seen_ids.add(image_id)

        root = Path(__file__).resolve().parents[3]
        allowed_assets = (root / "data" / "assets").resolve()
        for raw_payload in metadata.get("images", []):
            image_payload = dict(raw_payload)
            image_id = str(image_payload["image_id"])
            if image_id in seen_ids:
                raise IngestError(
                    IngestErrorCode.CONTRACT_VIOLATION,
                    "Two image assets resolved to the same governed identifier.",
                )
            if image_payload.get("object_ref"):
                object_ref = ObjectRef.model_validate(image_payload["object_ref"])
                self.store.load_object(object_ref)
            elif image_payload.get("source_path"):
                source_path = (root / image_payload["source_path"]).resolve()
                if not source_path.is_relative_to(allowed_assets) or not source_path.is_file():
                    raise ValueError("Synthetic image source must be inside data/assets.")
                content = source_path.read_bytes()
                expected_hash = image_payload.get("sha256")
                actual_hash = hashlib.sha256(content).hexdigest()
                if expected_hash and expected_hash != actual_hash:
                    raise ValueError("Synthetic image SHA-256 does not match metadata.")
                object_ref = self.store.store_image_asset(
                    document_id=metadata["document_id"],
                    revision=metadata["revision"],
                    image_id=image_id,
                    filename=source_path.name,
                    content=content,
                    content_type=(
                        image_payload.get("content_type")
                        or mimetypes.guess_type(source_path.name)[0]
                        or "application/octet-stream"
                    ),
                    source_hash=source_hash,
                )
            else:
                raise ValueError("Image metadata requires a stored object reference or source asset.")
            materialized.append(
                {
                    **image_payload,
                    "source_asset_id": image_payload.get("source_asset_id", image_id),
                    "source_location": image_payload.get("source_location", {}),
                    "related_chunk_draft_ids": [],
                    "parser_name": image_payload.get("parser_name", "declared-metadata"),
                    "parser_version": image_payload.get("parser_version", "1"),
                    "object_ref": object_ref.model_dump(mode="json"),
                }
            )
            seen_ids.add(image_id)
        return materialized

    def _materialize_tables(
        self,
        metadata: dict[str, Any],
        parsed: ParsedDocument,
        *,
        source_hash: str,
    ) -> list[dict[str, Any]]:
        materialized: list[dict[str, Any]] = []
        for index, table_draft in enumerate(parsed.tables, start=1):
            table_id = scoped_id(
                metadata["document_id"],
                metadata["revision"],
                "TABLE",
                index,
            )
            artifact = {
                "schema_version": "semikb-table-asset-v1",
                "table_id": table_id,
                "source_asset_id": table_draft.asset_id,
                "title": table_draft.title,
                "headers": list(table_draft.headers),
                "row_count": table_draft.row_count,
                "column_count": table_draft.column_count,
                "location": table_draft.location.model_dump(mode="json"),
                "markdown": table_draft.markdown,
                "html": table_draft.html,
            }
            content = json.dumps(
                artifact,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            object_ref = self.store.store_table_asset(
                document_id=metadata["document_id"],
                revision=metadata["revision"],
                table_id=table_id,
                content=content,
                source_hash=source_hash,
            )
            self.store.load_object(object_ref)
            materialized.append(
                {
                    **artifact,
                    "source_page": location_label(table_draft.location),
                    "related_chunk_draft_ids": list(
                        table_draft.related_chunk_draft_ids
                    ),
                    "object_ref": object_ref.model_dump(mode="json"),
                }
            )
        return materialized

    def _encode_chunks(self, chunks: Sequence[Chunk]) -> list[HybridEmbedding]:
        embeddings: list[HybridEmbedding] = []
        batch_size = self.settings.embedding_batch_size
        texts = [chunk.chunk_text for chunk in chunks]
        for offset in range(0, len(texts), batch_size):
            embeddings.extend(self.encoder.encode(texts[offset : offset + batch_size]))
        if len(embeddings) != len(chunks):
            raise ValueError("Embedding encoder returned an unexpected row count.")
        return embeddings

    @staticmethod
    def _validate_submission(metadata: dict[str, Any], content: bytes) -> None:
        required = {"document_id", "revision", "title", "document_type"}
        missing = sorted(required.difference(metadata))
        if missing:
            raise ValueError(f"Missing fields: {', '.join(missing)}")
        if not content:
            raise ValueError("Source content is empty.")

    @staticmethod
    def _quality_check(
        document: DocumentRevision,
        chunks: list[Chunk],
        images: list[ImageAsset],
        tables: list[TableAsset],
        target_lifecycle: DocumentLifecycle,
    ) -> None:
        if (
            target_lifecycle is DocumentLifecycle.PUBLISHED
            and document.approval_status is not ApprovalStatus.APPROVED
        ):
            raise IngestError(
                IngestErrorCode.QUALITY_GATE_FAILED,
                "Only an approved revision can be published.",
            )
        if not chunks:
            raise IngestError(
                IngestErrorCode.EMPTY_PARSE_RESULT,
                "No semantic chunks were generated.",
            )
        if any(not chunk.chunk_text.strip() for chunk in chunks):
            raise IngestError(
                IngestErrorCode.QUALITY_GATE_FAILED,
                "Empty chunks cannot pass the quality gate.",
            )
        if not document.parser_name or not document.parser_version:
            raise IngestError(
                IngestErrorCode.CONTRACT_VIOLATION,
                "Parser provenance is required for governed publication.",
            )
        image_ids = {image.image_id for image in images}
        table_ids = {table.table_id for table in tables}
        if len(image_ids) != len(images) or len(table_ids) != len(tables):
            raise IngestError(
                IngestErrorCode.CONTRACT_VIOLATION,
                "Asset identifiers must be unique within one revision.",
            )
        for chunk in chunks:
            if not set(chunk.image_ids).issubset(image_ids):
                raise IngestError(
                    IngestErrorCode.CONTRACT_VIOLATION,
                    "A chunk references an image that was not materialized.",
                )
            if not set(chunk.table_ids).issubset(table_ids):
                raise IngestError(
                    IngestErrorCode.CONTRACT_VIOLATION,
                    "A chunk references a table that was not materialized.",
                )
        for image in images:
            if not image.caption.strip():
                raise IngestError(
                    IngestErrorCode.QUALITY_GATE_FAILED,
                    "Image caption is required for text-to-image retrieval.",
                )
            if image.caption_source != "human" and image.caption_confidence < 0.5:
                raise IngestError(
                    IngestErrorCode.QUALITY_GATE_FAILED,
                    "Low-confidence generated image captions require review.",
                )
        for table in tables:
            if not table.markdown.strip() or not table.html.strip():
                raise IngestError(
                    IngestErrorCode.QUALITY_GATE_FAILED,
                    "Table assets require both Markdown and HTML representations.",
                )

    @staticmethod
    def _failure_details(exc: Exception) -> tuple[str, str]:
        if isinstance(exc, IngestError):
            return exc.code.value, exc.safe_message
        return (
            type(exc).__name__.upper(),
            "Review the failed stage, source file, and governed metadata before retrying.",
        )
