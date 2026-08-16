"""MongoDB and in-memory persistence for corpus publication batches."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from semikb.contracts.corpus_publication import (
    CorpusPublicationBatch,
    CorpusPublicationItemStatus,
    CorpusPublicationStatus,
)
from semikb.storage.clients import StorageClientFactory


class CorpusPublicationConflictError(RuntimeError):
    pass


def _clean(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    return {key: value for key, value in document.items() if key != "_id"}


class MongoCorpusPublicationRepository:
    def __init__(self, factory: StorageClientFactory, database_name: str) -> None:
        self._factory = factory
        self._database_name = database_name

    def create_or_get(self, batch: CorpusPublicationBatch) -> CorpusPublicationBatch:
        selector = {"review.request_id": batch.review.request_id}
        with self._factory.mongodb() as client:
            collection = client[self._database_name].corpus_publication_batches
            try:
                collection.insert_one(batch.model_dump(mode="python"))
                return batch
            except DuplicateKeyError:
                stored = collection.find_one(selector, {"_id": 0})
        existing = CorpusPublicationBatch.model_validate(stored)
        if existing.request_fingerprint != batch.request_fingerprint:
            raise CorpusPublicationConflictError(
                "The publication request ID already exists with different reviewed content."
            )
        return existing

    def save(self, batch: CorpusPublicationBatch) -> CorpusPublicationBatch:
        with self._factory.mongodb() as client:
            result = client[self._database_name].corpus_publication_batches.replace_one(
                {"batch_id": batch.batch_id},
                batch.model_dump(mode="python"),
            )
        if result.matched_count != 1:
            raise KeyError(batch.batch_id)
        return batch

    def get(self, batch_id: str) -> CorpusPublicationBatch | None:
        with self._factory.mongodb() as client:
            stored = client[self._database_name].corpus_publication_batches.find_one(
                {"batch_id": batch_id}, {"_id": 0}
            )
        return CorpusPublicationBatch.model_validate(stored) if stored else None

    def list(self) -> list[CorpusPublicationBatch]:
        with self._factory.mongodb() as client:
            rows = list(
                client[self._database_name]
                .corpus_publication_batches.find({}, {"_id": 0})
                .sort("created_at", -1)
            )
        return [CorpusPublicationBatch.model_validate(row) for row in rows]

    def claim(self, batch_id: str, execution_id: str | None) -> CorpusPublicationBatch | None:
        now = datetime.now(UTC)
        with self._factory.mongodb() as client:
            stored = client[self._database_name].corpus_publication_batches.find_one_and_update(
                {"batch_id": batch_id, "status": CorpusPublicationStatus.QUEUED.value},
                {
                    "$set": {
                        "status": CorpusPublicationStatus.PREFLIGHT.value,
                        "worker_task_id": execution_id,
                        "started_at": now,
                        "finished_at": None,
                        "error_code": None,
                        "safe_error_summary": None,
                    }
                },
                projection={"_id": 0},
                return_document=ReturnDocument.AFTER,
            )
        return CorpusPublicationBatch.model_validate(_clean(stored)) if stored else None

    def prepare_retry(self, batch_id: str) -> CorpusPublicationBatch:
        with self._factory.mongodb() as client:
            stored = client[self._database_name].corpus_publication_batches.find_one_and_update(
                {"batch_id": batch_id, "status": CorpusPublicationStatus.FAILED.value},
                {
                    "$set": {
                        "status": CorpusPublicationStatus.QUEUED.value,
                        "progress": 0,
                        "worker_task_id": None,
                        "started_at": None,
                        "finished_at": None,
                        "error_code": None,
                        "safe_error_summary": None,
                        "items.$[failed].status": CorpusPublicationItemStatus.PENDING.value,
                        "items.$[failed].error_code": None,
                        "items.$[failed].safe_error_summary": None,
                    },
                    "$inc": {"attempt": 1},
                },
                array_filters=[{"failed.status": CorpusPublicationItemStatus.FAILED.value}],
                projection={"_id": 0},
                return_document=ReturnDocument.AFTER,
            )
        if stored is None:
            existing = self.get(batch_id)
            if existing is None:
                raise KeyError(batch_id)
            return existing
        return CorpusPublicationBatch.model_validate(_clean(stored))


class DemoCorpusPublicationRepository:
    def __init__(self) -> None:
        self.batches: dict[str, CorpusPublicationBatch] = {}
        self.request_ids: dict[str, str] = {}

    def create_or_get(self, batch: CorpusPublicationBatch) -> CorpusPublicationBatch:
        existing_id = self.request_ids.get(batch.review.request_id)
        if existing_id:
            existing = self.batches[existing_id]
            if existing.request_fingerprint != batch.request_fingerprint:
                raise CorpusPublicationConflictError(
                    "The publication request ID already exists with different reviewed content."
                )
            return deepcopy(existing)
        self.batches[batch.batch_id] = deepcopy(batch)
        self.request_ids[batch.review.request_id] = batch.batch_id
        return deepcopy(batch)

    def save(self, batch: CorpusPublicationBatch) -> CorpusPublicationBatch:
        if batch.batch_id not in self.batches:
            raise KeyError(batch.batch_id)
        self.batches[batch.batch_id] = deepcopy(batch)
        return deepcopy(batch)

    def get(self, batch_id: str) -> CorpusPublicationBatch | None:
        batch = self.batches.get(batch_id)
        return deepcopy(batch) if batch else None

    def list(self) -> list[CorpusPublicationBatch]:
        return sorted(
            (deepcopy(batch) for batch in self.batches.values()),
            key=lambda batch: batch.created_at,
            reverse=True,
        )

    def claim(self, batch_id: str, execution_id: str | None) -> CorpusPublicationBatch | None:
        batch = self.batches.get(batch_id)
        if batch is None or batch.status is not CorpusPublicationStatus.QUEUED:
            return None
        batch.status = CorpusPublicationStatus.PREFLIGHT
        batch.worker_task_id = execution_id
        batch.started_at = datetime.now(UTC)
        self.batches[batch_id] = deepcopy(batch)
        return deepcopy(batch)

    def prepare_retry(self, batch_id: str) -> CorpusPublicationBatch:
        batch = self.batches.get(batch_id)
        if batch is None:
            raise KeyError(batch_id)
        if batch.status is not CorpusPublicationStatus.FAILED:
            return deepcopy(batch)
        batch.status = CorpusPublicationStatus.QUEUED
        batch.progress = 0
        batch.attempt += 1
        batch.worker_task_id = None
        batch.started_at = None
        batch.finished_at = None
        batch.error_code = None
        batch.safe_error_summary = None
        for item in batch.items:
            if item.status is CorpusPublicationItemStatus.FAILED:
                item.status = CorpusPublicationItemStatus.PENDING
                item.error_code = None
                item.safe_error_summary = None
        self.batches[batch_id] = deepcopy(batch)
        return deepcopy(batch)
