"""Persistence adapters for corpus standardization jobs and immutable artifacts."""

from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Any

from pymongo import ReturnDocument

from semikb.contracts.corpus import CorpusStandardizationJob
from semikb.contracts.models import ObjectRef
from semikb.storage.clients import StorageClientFactory
from semikb.storage.minio_artifacts import MinioArtifactRepository


def _clean(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    return {key: value for key, value in document.items() if key != "_id"}


class CorpusStandardizationConflictError(RuntimeError):
    """The same idempotency identity was reused with different request content."""


class MongoCorpusStandardizationRepository:
    def __init__(self, factory: StorageClientFactory, database_name: str) -> None:
        self._factory = factory
        self._database_name = database_name

    def create_or_get(self, job: CorpusStandardizationJob) -> CorpusStandardizationJob:
        document = job.model_dump(mode="python")
        with self._factory.mongodb() as client:
            stored = client[self._database_name].corpus_standardization_jobs.find_one_and_update(
                {"idempotency_key": job.idempotency_key},
                {"$setOnInsert": document},
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
        result = CorpusStandardizationJob.model_validate(_clean(stored))
        if result.request_fingerprint != job.request_fingerprint:
            raise CorpusStandardizationConflictError(
                "The corpus snapshot version already exists with different metadata or files."
            )
        return result

    def save(self, job: CorpusStandardizationJob) -> CorpusStandardizationJob:
        with self._factory.mongodb() as client:
            result = client[self._database_name].corpus_standardization_jobs.replace_one(
                {"job_id": job.job_id},
                job.model_dump(mode="python"),
            )
        if result.matched_count != 1:
            raise KeyError(job.job_id)
        return job

    def get(self, job_id: str) -> CorpusStandardizationJob | None:
        with self._factory.mongodb() as client:
            stored = client[self._database_name].corpus_standardization_jobs.find_one(
                {"job_id": job_id}
            )
        clean = _clean(stored)
        return CorpusStandardizationJob.model_validate(clean) if clean else None

    def list(self) -> list[CorpusStandardizationJob]:
        with self._factory.mongodb() as client:
            rows = list(
                client[self._database_name]
                .corpus_standardization_jobs.find({})
                .sort("created_at", -1)
            )
        return [CorpusStandardizationJob.model_validate(_clean(row)) for row in rows]


class DemoCorpusStandardizationRepository:
    def __init__(self) -> None:
        self._lock = RLock()
        self.jobs: dict[str, CorpusStandardizationJob] = {}
        self.keys: dict[str, str] = {}
        self.objects: dict[tuple[str, str], bytes] = {}

    def create_or_get(self, job: CorpusStandardizationJob) -> CorpusStandardizationJob:
        with self._lock:
            existing_id = self.keys.get(job.idempotency_key)
            if existing_id:
                existing = self.jobs[existing_id]
                if existing.request_fingerprint != job.request_fingerprint:
                    raise CorpusStandardizationConflictError(
                        "The corpus snapshot version already exists with different metadata or files."
                    )
                return deepcopy(existing)
            self.jobs[job.job_id] = deepcopy(job)
            self.keys[job.idempotency_key] = job.job_id
            return deepcopy(job)

    def save(self, job: CorpusStandardizationJob) -> CorpusStandardizationJob:
        with self._lock:
            if job.job_id not in self.jobs:
                raise KeyError(job.job_id)
            self.jobs[job.job_id] = deepcopy(job)
            return deepcopy(job)

    def get(self, job_id: str) -> CorpusStandardizationJob | None:
        job = self.jobs.get(job_id)
        return deepcopy(job) if job else None

    def list(self) -> list[CorpusStandardizationJob]:
        return sorted(
            (deepcopy(job) for job in self.jobs.values()),
            key=lambda item: item.created_at,
            reverse=True,
        )

    def store_raw(self, **kwargs: Any) -> ObjectRef:
        return self._store(bucket="semikb-raw", **kwargs)

    def store_derived(self, **kwargs: Any) -> ObjectRef:
        return self._store(bucket="semikb-derived", **kwargs)

    def _store(
        self,
        *,
        bucket: str,
        corpus_id: str,
        snapshot_hash: str,
        category: str,
        relative_path: str,
        content: bytes,
        content_type: str,
    ) -> ObjectRef:
        import hashlib
        from urllib.parse import quote

        safe_path = "/".join(quote(part, safe="._-") for part in relative_path.split("/"))
        key = f"corpora/{corpus_id}/{snapshot_hash}/{category}/{safe_path}"
        ref = ObjectRef(
            bucket=bucket,
            object_key=key,
            content_type=content_type,
            sha256=hashlib.sha256(content).hexdigest(),
        )
        self.objects[(bucket, key)] = bytes(content)
        return ref

    def load_object(self, object_ref: ObjectRef) -> bytes:
        return self.objects[(object_ref.bucket, object_ref.object_key)]


class ProductionCorpusStandardizationStore:
    def __init__(self, settings: Any, factory: StorageClientFactory | None = None) -> None:
        client_factory = factory or StorageClientFactory(settings)
        self.jobs = MongoCorpusStandardizationRepository(
            client_factory,
            settings.mongodb_database,
        )
        self.artifacts = MinioArtifactRepository(client_factory)

    def create_or_get(self, job: CorpusStandardizationJob) -> CorpusStandardizationJob:
        return self.jobs.create_or_get(job)

    def save(self, job: CorpusStandardizationJob) -> CorpusStandardizationJob:
        return self.jobs.save(job)

    def get(self, job_id: str) -> CorpusStandardizationJob | None:
        return self.jobs.get(job_id)

    def list(self) -> list[CorpusStandardizationJob]:
        return self.jobs.list()

    def store_raw(self, **kwargs: Any) -> ObjectRef:
        return self.artifacts.store_corpus_raw(**kwargs)

    def store_derived(self, **kwargs: Any) -> ObjectRef:
        return self.artifacts.store_corpus_derived(**kwargs)

    def load_object(self, object_ref: ObjectRef) -> bytes:
        return self.artifacts.load_object(object_ref)

