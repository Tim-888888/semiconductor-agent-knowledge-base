"""MongoDB authority for ingestion jobs, events, and governed document records."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from pymongo import ReplaceOne, ReturnDocument

from semikb.contracts.models import (
    Chunk,
    DocumentLifecycle,
    DocumentRevision,
    ImageAsset,
    IngestionEvent,
    IngestionJob,
    IngestionStatus,
    ObjectRef,
    TableAsset,
)
from semikb.storage.clients import StorageClientFactory

_REVISION_COLLECTIONS = (
    "document_catalog",
    "chunk_catalog",
    "image_assets",
    "table_assets",
)


def _model_document(model: Any) -> dict[str, Any]:
    return model.model_dump(mode="python")


def _without_mongo_id(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    return {key: value for key, value in document.items() if key != "_id"}


class MongoIngestionRepository:
    """Keeps business state in MongoDB without requiring cross-collection transactions."""

    def __init__(self, factory: StorageClientFactory, database_name: str) -> None:
        self._factory = factory
        self._database_name = database_name

    def create_or_get_job(self, job: IngestionJob) -> IngestionJob:
        queued_event = IngestionEvent(
            job_id=job.job_id,
            stage=IngestionStatus.QUEUED,
            message="Ingestion job accepted.",
            attempt=job.attempt,
            progress=0,
        )
        job.events = [queued_event]
        with self._factory.mongodb() as client:
            database = client[self._database_name]
            stored = database.ingestion_jobs.find_one_and_update(
                {"idempotency_key": job.idempotency_key},
                {"$setOnInsert": _model_document(job)},
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            if stored and stored.get("job_id") == job.job_id:
                self._mirror_event(database, queued_event)
        return IngestionJob.model_validate(_without_mongo_id(stored))

    def get_job(self, job_id: str) -> IngestionJob | None:
        with self._factory.mongodb() as client:
            stored = client[self._database_name].ingestion_jobs.find_one({"job_id": job_id})
        clean = _without_mongo_id(stored)
        return IngestionJob.model_validate(clean) if clean else None

    def list_jobs(self) -> list[IngestionJob]:
        with self._factory.mongodb() as client:
            records = list(
                client[self._database_name]
                .ingestion_jobs.find({}, {"replay_payload": 0})
                .sort("created_at", -1)
            )
        return [IngestionJob.model_validate(_without_mongo_id(record)) for record in records]

    def save_replay_payload(self, job_id: str, payload: dict[str, Any]) -> None:
        with self._factory.mongodb() as client:
            result = client[self._database_name].ingestion_jobs.update_one(
                {"job_id": job_id},
                {"$set": {"replay_payload": payload}},
            )
        if result.matched_count != 1:
            raise KeyError(job_id)

    def get_replay_payload(self, job_id: str) -> dict[str, Any] | None:
        with self._factory.mongodb() as client:
            record = client[self._database_name].ingestion_jobs.find_one(
                {"job_id": job_id},
                {"replay_payload": 1},
            )
        return record.get("replay_payload") if record else None

    def prepare_retry(self, job_id: str) -> IngestionJob:
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status is not IngestionStatus.FAILED:
            return job
        job.attempt += 1
        job.status = IngestionStatus.QUEUED
        job.current_stage = IngestionStatus.QUEUED
        job.progress = 0
        job.error_code = None
        job.safe_error_summary = None
        job.failed_stage = None
        job.finished_at = None
        event = IngestionEvent(
            job_id=job.job_id,
            stage=IngestionStatus.QUEUED,
            message="Ingestion retry accepted.",
            attempt=job.attempt,
            progress=0,
        )
        job.events.append(event)
        self._save_job(job, event)
        return job

    def update_job(
        self,
        job_id: str,
        stage: IngestionStatus,
        message: str,
        progress: int,
        *,
        error_code: str | None = None,
    ) -> IngestionJob:
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        previous_stage = job.current_stage
        job.status = stage
        job.current_stage = stage
        job.progress = progress
        if stage is IngestionStatus.VALIDATING and job.started_at is None:
            job.started_at = datetime.now(UTC)
        if stage is IngestionStatus.FAILED:
            job.error_code = error_code or "INGESTION_FAILED"
            job.safe_error_summary = message
            job.failed_stage = previous_stage
            job.finished_at = datetime.now(UTC)
        elif stage in {IngestionStatus.STAGED, IngestionStatus.PUBLISHED} and progress == 100:
            job.finished_at = datetime.now(UTC)
        event = IngestionEvent(
            job_id=job.job_id,
            stage=stage,
            message=message,
            attempt=job.attempt,
            progress=progress,
        )
        job.events.append(event)
        self._save_job(job, event)
        return job

    def set_job_artifacts(
        self,
        job_id: str,
        *,
        source_ref: ObjectRef | None = None,
        parsed_ref: ObjectRef | None = None,
    ) -> IngestionJob:
        values: dict[str, Any] = {}
        if source_ref is not None:
            values["source_ref"] = _model_document(source_ref)
        if parsed_ref is not None:
            values["parsed_ref"] = _model_document(parsed_ref)
        return self._set_job_values(job_id, values)

    def set_job_counts(
        self,
        job_id: str,
        *,
        chunks_count: int,
        images_count: int,
        tables_count: int,
    ) -> IngestionJob:
        return self._set_job_values(
            job_id,
            {
                "chunks_count": chunks_count,
                "images_count": images_count,
                "tables_count": tables_count,
            },
        )

    def set_job_parse_audit(
        self,
        job_id: str,
        *,
        parse_contract_version: str,
        parser_name: str,
        parser_version: str,
        provider_name: str | None,
        provider_version: str | None,
        upstream_project: str | None,
        upstream_commit: str | None,
        chunker_version: str,
        warning_codes: Sequence[str],
        metrics: dict[str, object],
    ) -> IngestionJob:
        return self._set_job_values(
            job_id,
            {
                "parse_contract_version": parse_contract_version,
                "parser_name": parser_name,
                "parser_version": parser_version,
                "provider_name": provider_name,
                "provider_version": provider_version,
                "upstream_project": upstream_project,
                "upstream_commit": upstream_commit,
                "chunker_version": chunker_version,
                "parse_warning_codes": list(warning_codes),
                "parse_metrics": dict(metrics),
            },
        )

    def stage_document(
        self,
        document: DocumentRevision,
        chunks: Sequence[Chunk],
        images: Sequence[ImageAsset],
        tables: Sequence[TableAsset] = (),
    ) -> None:
        with self._factory.mongodb() as client:
            database = client[self._database_name]
            existing = database.document_catalog.find_one(
                {"document_id": document.document_id, "revision": document.revision},
                {"source_hash": 1},
            )
            if existing and existing.get("source_hash") != document.source_hash:
                raise ValueError(
                    "The document revision already exists with a different source hash."
                )
            database.document_catalog.replace_one(
                {"document_id": document.document_id, "revision": document.revision},
                _model_document(document),
                upsert=True,
            )
            database.chunk_catalog.delete_many(
                {"document_id": document.document_id, "revision": document.revision}
            )
            if chunks:
                database.chunk_catalog.bulk_write(
                    [
                        ReplaceOne(
                            {"chunk_id": chunk.chunk_id},
                            _model_document(chunk),
                            upsert=True,
                        )
                        for chunk in chunks
                    ]
                )
            database.image_assets.delete_many(
                {"document_id": document.document_id, "revision": document.revision}
            )
            if images:
                database.image_assets.bulk_write(
                    [
                        ReplaceOne(
                            {"image_id": image.image_id},
                            _model_document(image),
                            upsert=True,
                        )
                        for image in images
                    ]
                )
            database.table_assets.delete_many(
                {"document_id": document.document_id, "revision": document.revision}
            )
            if tables:
                database.table_assets.bulk_write(
                    [
                        ReplaceOne(
                            {"table_id": table.table_id},
                            _model_document(table),
                            upsert=True,
                        )
                        for table in tables
                    ]
                )

    def publish_document(self, document: DocumentRevision) -> None:
        now = datetime.now(UTC)
        with self._factory.mongodb() as client:
            database = client[self._database_name]
            selector = {"document_id": document.document_id, "revision": document.revision}
            result = database.document_catalog.update_one(
                selector,
                {"$set": {"lifecycle": DocumentLifecycle.PUBLISHED.value, "published_at": now}},
            )
            if result.matched_count != 1:
                raise KeyError(f"{document.document_id}:{document.revision}")
            database.chunk_catalog.update_many(
                selector,
                {"$set": {"lifecycle": DocumentLifecycle.PUBLISHED.value}},
            )
            database.image_assets.update_many(
                selector,
                {"$set": {"lifecycle": DocumentLifecycle.PUBLISHED.value}},
            )
            database.table_assets.update_many(
                selector,
                {"$set": {"lifecycle": DocumentLifecycle.PUBLISHED.value}},
            )

    def supersede_previous(self, document: DocumentRevision) -> list[str]:
        if not document.supersedes_revision:
            return []
        selector = {
            "document_id": document.document_id,
            "revision": document.supersedes_revision,
        }
        with self._factory.mongodb() as client:
            database = client[self._database_name]
            chunk_ids = [
                record["chunk_id"]
                for record in database.chunk_catalog.find(selector, {"chunk_id": 1})
            ]
            for collection_name in _REVISION_COLLECTIONS:
                database[collection_name].update_many(
                    selector,
                    {"$set": {"lifecycle": DocumentLifecycle.SUPERSEDED.value}},
                )
            return chunk_ids

    def restore_previous(self, document: DocumentRevision) -> None:
        if not document.supersedes_revision:
            return
        selector = {
            "document_id": document.document_id,
            "revision": document.supersedes_revision,
        }
        with self._factory.mongodb() as client:
            database = client[self._database_name]
            for collection_name in _REVISION_COLLECTIONS:
                database[collection_name].update_many(
                    selector,
                    {"$set": {"lifecycle": DocumentLifecycle.PUBLISHED.value}},
                )

    def get_revision_vector_refs(
        self,
        document_id: str,
        revision: str,
    ) -> tuple[str | None, list[str]]:
        selector = {"document_id": document_id, "revision": revision}
        with self._factory.mongodb() as client:
            database = client[self._database_name]
            document = database.document_catalog.find_one(selector, {"index_version": 1})
            chunk_ids = [
                record["chunk_id"]
                for record in database.chunk_catalog.find(selector, {"chunk_id": 1})
            ]
        return (document.get("index_version") if document else None, chunk_ids)

    def finalize_inactive_document(
        self,
        document_id: str,
        revision: str,
        lifecycle: DocumentLifecycle,
    ) -> None:
        selector = {"document_id": document_id, "revision": revision}
        with self._factory.mongodb() as client:
            database = client[self._database_name]
            for collection_name in _REVISION_COLLECTIONS:
                database[collection_name].update_many(
                    selector,
                    {"$set": {"lifecycle": lifecycle.value}},
                )

    def compensate_document(self, document_id: str, revision: str) -> list[str]:
        selector = {"document_id": document_id, "revision": revision}
        with self._factory.mongodb() as client:
            database = client[self._database_name]
            chunk_ids = [
                record["chunk_id"]
                for record in database.chunk_catalog.find(selector, {"chunk_id": 1})
            ]
            for collection_name in _REVISION_COLLECTIONS:
                database[collection_name].update_many(
                    selector,
                    {"$set": {"lifecycle": DocumentLifecycle.QUARANTINED.value}},
                )
            return chunk_ids

    def record_release(
        self,
        document: DocumentRevision,
        chunks_count: int,
        *,
        embedding_dim: int,
        embedding_output_type: str,
        sparse_encoder_version: str,
        normalization: str,
    ) -> None:
        with self._factory.mongodb() as client:
            client[self._database_name].index_releases.update_one(
                {"index_version": document.index_version},
                {
                    "$set": {
                        "status": "active",
                        "alias": "semikb_chunks_active",
                        "collection": f"semikb_chunks_{document.index_version.replace('-', '_')}",
                        "embedding_version": document.embedding_version,
                        "embedding_dim": embedding_dim,
                        "embedding_output_type": embedding_output_type,
                        "sparse_encoder_version": sparse_encoder_version,
                        "normalization": normalization,
                        "parse_contract_version": document.parse_contract_version,
                        "parser_name": document.parser_name,
                        "parser_version": document.parser_version,
                        "provider_name": document.provider_name,
                        "provider_version": document.provider_version,
                        "upstream_commit": document.upstream_commit,
                        "chunker_version": document.chunker_version,
                        "last_document_id": document.document_id,
                        "last_revision": document.revision,
                        "last_chunks_count": chunks_count,
                        "published_at": datetime.now(UTC),
                    }
                },
                upsert=True,
            )

    def record_release_failure(self, document: DocumentRevision) -> None:
        now = datetime.now(UTC)
        with self._factory.mongodb() as client:
            client[self._database_name].index_releases.update_one(
                {"index_version": document.index_version},
                {
                    "$set": {
                        "last_failure_at": now,
                        "last_failed_document_id": document.document_id,
                        "last_failed_revision": document.revision,
                    },
                    "$setOnInsert": {
                        "status": "failed",
                        "failed_at": now,
                    },
                },
                upsert=True,
            )

    def _set_job_values(self, job_id: str, values: dict[str, Any]) -> IngestionJob:
        if not values:
            job = self.get_job(job_id)
            if job is None:
                raise KeyError(job_id)
            return job
        with self._factory.mongodb() as client:
            stored = client[self._database_name].ingestion_jobs.find_one_and_update(
                {"job_id": job_id},
                {"$set": values},
                return_document=ReturnDocument.AFTER,
            )
        if stored is None:
            raise KeyError(job_id)
        return IngestionJob.model_validate(_without_mongo_id(stored))

    def _save_job(self, job: IngestionJob, event: IngestionEvent) -> None:
        values = _model_document(job)
        values.pop("events", None)
        with self._factory.mongodb() as client:
            database = client[self._database_name]
            result = database.ingestion_jobs.update_one(
                {"job_id": job.job_id},
                {"$set": values, "$push": {"events": _model_document(event)}},
            )
            if result.matched_count != 1:
                raise KeyError(job.job_id)
            self._mirror_event(database, event)

    @staticmethod
    def _mirror_event(database: Any, event: IngestionEvent) -> None:
        document = _model_document(event)
        database.ingestion_job_events.replace_one(
            {"_id": event.event_id},
            {"_id": event.event_id, **document},
            upsert=True,
        )
