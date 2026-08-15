"""Composite production ingestion store across MongoDB, MinIO, and Milvus."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from semikb.config import Settings
from semikb.contracts.models import (
    Chunk,
    DocumentLifecycle,
    DocumentRevision,
    ImageAsset,
    IngestionJob,
    IngestionStatus,
    ObjectRef,
    TableAsset,
)
from semikb.rag_retrieval.encoders import HybridEmbedding
from semikb.storage.clients import StorageClientFactory
from semikb.storage.milvus_chunks import MilvusChunkRepository
from semikb.storage.minio_artifacts import MinioArtifactRepository
from semikb.storage.mongo_ingestion import MongoIngestionRepository


class ProductionIngestionStore:
    """Coordinates writes while keeping MongoDB as the business-state authority."""

    def __init__(self, settings: Settings, factory: StorageClientFactory | None = None) -> None:
        self._settings = settings
        client_factory = factory or StorageClientFactory(settings)
        self.mongo = MongoIngestionRepository(client_factory, settings.mongodb_database)
        self.artifacts = MinioArtifactRepository(client_factory)
        self.vectors = MilvusChunkRepository(client_factory, settings)

    def create_or_get_job(self, job: IngestionJob) -> IngestionJob:
        return self.mongo.create_or_get_job(job)

    def get_job(self, job_id: str) -> IngestionJob | None:
        return self.mongo.get_job(job_id)

    def list_jobs(self) -> list[IngestionJob]:
        return self.mongo.list_jobs()

    def save_replay_payload(self, job_id: str, payload: dict[str, Any]) -> None:
        self.mongo.save_replay_payload(job_id, payload)

    def get_replay_payload(self, job_id: str) -> dict[str, Any] | None:
        return self.mongo.get_replay_payload(job_id)

    def prepare_retry(self, job_id: str) -> IngestionJob:
        return self.mongo.prepare_retry(job_id)

    def update_job(
        self,
        job_id: str,
        stage: IngestionStatus,
        message: str,
        progress: int,
        *,
        error_code: str | None = None,
    ) -> IngestionJob:
        return self.mongo.update_job(
            job_id,
            stage,
            message,
            progress,
            error_code=error_code,
        )

    def set_job_artifacts(
        self,
        job_id: str,
        *,
        source_ref: ObjectRef | None = None,
        parsed_ref: ObjectRef | None = None,
    ) -> IngestionJob:
        return self.mongo.set_job_artifacts(
            job_id,
            source_ref=source_ref,
            parsed_ref=parsed_ref,
        )

    def set_job_counts(
        self,
        job_id: str,
        *,
        chunks_count: int,
        images_count: int,
        tables_count: int,
    ) -> IngestionJob:
        return self.mongo.set_job_counts(
            job_id,
            chunks_count=chunks_count,
            images_count=images_count,
            tables_count=tables_count,
        )

    def set_job_parse_audit(self, job_id: str, **kwargs: Any) -> IngestionJob:
        return self.mongo.set_job_parse_audit(job_id, **kwargs)

    def store_source(self, **kwargs: Any) -> ObjectRef:
        return self.artifacts.store_source(**kwargs)

    def load_object(self, object_ref: ObjectRef) -> bytes:
        return self.artifacts.load_object(object_ref)

    def store_parsed_markdown(self, **kwargs: Any) -> ObjectRef:
        return self.artifacts.store_parsed_markdown(**kwargs)

    def store_image_asset(self, **kwargs: Any) -> ObjectRef:
        return self.artifacts.store_image_asset(**kwargs)

    def store_table_asset(self, **kwargs: Any) -> ObjectRef:
        return self.artifacts.store_table_asset(**kwargs)

    def stage_document(
        self,
        document: DocumentRevision,
        chunks: Sequence[Chunk],
        images: Sequence[ImageAsset],
        embeddings: Sequence[HybridEmbedding],
        tables: Sequence[TableAsset] = (),
    ) -> None:
        self.mongo.stage_document(document, chunks, images, tables)
        try:
            self.vectors.upsert_chunks(
                chunks,
                embeddings,
                lifecycle=DocumentLifecycle.STAGED,
            )
        except Exception:
            chunk_ids = self.mongo.compensate_document(document.document_id, document.revision)
            self.vectors.delete_chunks(document.index_version, chunk_ids)
            raise

    def publish_document(
        self,
        document: DocumentRevision,
        chunks: Sequence[Chunk],
        images: Sequence[ImageAsset],
        embeddings: Sequence[HybridEmbedding],
        tables: Sequence[TableAsset] = (),
    ) -> None:
        previous_superseded = False
        try:
            self.mongo.publish_document(document)
            self.vectors.upsert_chunks(
                chunks,
                embeddings,
                lifecycle=DocumentLifecycle.PUBLISHED,
            )
            self.vectors.activate_alias(document.index_version)
            self.mongo.record_release(
                document,
                len(chunks),
                embedding_dim=self._settings.embedding_dim,
                embedding_output_type=self._settings.embedding_output_type,
                sparse_encoder_version=self._settings.sparse_encoder_version,
                normalization=(
                    "dense_l2_sparse_provider_raw"
                    if self._settings.embedding_output_type == "dense&sparse"
                    else "dense_l2_sparse_lexical_l2"
                ),
            )
            superseded_chunk_ids = self.mongo.supersede_previous(document)
            previous_superseded = bool(document.supersedes_revision)
            self.vectors.delete_chunks(document.index_version, superseded_chunk_ids)
        except Exception:
            if previous_superseded:
                self.mongo.restore_previous(document)
            chunk_ids = self.mongo.compensate_document(document.document_id, document.revision)
            self.vectors.delete_chunks(document.index_version, chunk_ids)
            self.mongo.record_release_failure(document)
            raise

    def finalize_inactive_document(
        self,
        document_id: str,
        revision: str,
        lifecycle: DocumentLifecycle,
    ) -> None:
        index_version, chunk_ids = self.mongo.get_revision_vector_refs(document_id, revision)
        self.mongo.finalize_inactive_document(document_id, revision, lifecycle)
        if index_version:
            self.vectors.delete_chunks(index_version, chunk_ids)

    def compensate_document(self, document_id: str, revision: str) -> None:
        index_version, _ = self.mongo.get_revision_vector_refs(document_id, revision)
        chunk_ids = self.mongo.compensate_document(document_id, revision)
        if index_version:
            self.vectors.delete_chunks(index_version, chunk_ids)
