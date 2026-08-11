"""Storage boundary used by the ingestion application service."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from semikb.contracts.models import (
    Chunk,
    DocumentLifecycle,
    DocumentRevision,
    ImageAsset,
    IngestionJob,
    IngestionStatus,
    ObjectRef,
)
from semikb.rag_retrieval.encoders import HybridEmbedding


class IngestionStore(Protocol):
    """Persistence operations required by one replayable ingestion run."""

    def create_or_get_job(self, job: IngestionJob) -> IngestionJob: ...

    def get_job(self, job_id: str) -> IngestionJob | None: ...

    def list_jobs(self) -> list[IngestionJob]: ...

    def save_replay_payload(self, job_id: str, payload: dict[str, Any]) -> None: ...

    def get_replay_payload(self, job_id: str) -> dict[str, Any] | None: ...

    def prepare_retry(self, job_id: str) -> IngestionJob: ...

    def update_job(
        self,
        job_id: str,
        stage: IngestionStatus,
        message: str,
        progress: int,
        *,
        error_code: str | None = None,
    ) -> IngestionJob: ...

    def set_job_artifacts(
        self,
        job_id: str,
        *,
        source_ref: ObjectRef | None = None,
        parsed_ref: ObjectRef | None = None,
    ) -> IngestionJob: ...

    def set_job_counts(
        self,
        job_id: str,
        *,
        chunks_count: int,
        images_count: int,
        tables_count: int,
    ) -> IngestionJob: ...

    def store_source(
        self,
        *,
        document_id: str,
        revision: str,
        filename: str,
        content: bytes,
        content_type: str,
        source_hash: str,
    ) -> ObjectRef: ...

    def load_object(self, object_ref: ObjectRef) -> bytes: ...

    def store_parsed_markdown(
        self,
        *,
        document_id: str,
        revision: str,
        parser_version: str,
        source_hash: str,
        content: bytes,
    ) -> ObjectRef: ...

    def store_image_asset(
        self,
        *,
        document_id: str,
        revision: str,
        image_id: str,
        filename: str,
        content: bytes,
        content_type: str,
        source_hash: str,
    ) -> ObjectRef: ...

    def stage_document(
        self,
        document: DocumentRevision,
        chunks: Sequence[Chunk],
        images: Sequence[ImageAsset],
        embeddings: Sequence[HybridEmbedding],
    ) -> None: ...

    def publish_document(
        self,
        document: DocumentRevision,
        chunks: Sequence[Chunk],
        images: Sequence[ImageAsset],
        embeddings: Sequence[HybridEmbedding],
    ) -> None: ...

    def finalize_inactive_document(
        self,
        document_id: str,
        revision: str,
        lifecycle: DocumentLifecycle,
    ) -> None: ...

    def compensate_document(self, document_id: str, revision: str) -> None: ...
