"""Persistence boundary for immutable evaluation datasets and run history."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from pymongo import MongoClient, ReturnDocument

from semikb.config import Settings
from semikb.contracts.models import (
    EvaluationDataset,
    EvaluationRun,
    EvaluationStatus,
)


class EvaluationRepository(Protocol):
    def save_evaluation_dataset(self, dataset: EvaluationDataset) -> EvaluationDataset: ...

    def get_evaluation_dataset(self, dataset_version: str) -> EvaluationDataset | None: ...

    def list_evaluation_datasets(self) -> list[EvaluationDataset]: ...

    def save_evaluation_run(self, run: EvaluationRun) -> EvaluationRun: ...

    def get_evaluation_run(self, evaluation_run_id: str) -> EvaluationRun | None: ...

    def list_evaluation_runs(self) -> list[EvaluationRun]: ...

    def claim_evaluation_run(
        self,
        evaluation_run_id: str,
        execution_id: str | None = None,
    ) -> EvaluationRun | None: ...

    def prepare_evaluation_retry(self, evaluation_run_id: str) -> EvaluationRun: ...


def _without_mongo_id(document: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in document.items() if key != "_id"}


class MongoEvaluationRepository:
    """Use MongoDB as the authority for datasets and asynchronous run state."""

    def __init__(self, settings: Settings, *, client: MongoClient | None = None) -> None:
        if not settings.mongodb_uri:
            raise ValueError("MONGODB_URI is required for production evaluation.")
        self.client = client or MongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
        self.database = self.client[settings.mongodb_database]
        self.datasets = self.database["evaluation_datasets"]
        self.runs = self.database["evaluation_runs"]

    def save_evaluation_dataset(self, dataset: EvaluationDataset) -> EvaluationDataset:
        self.datasets.update_one(
            {"dataset_version": dataset.dataset_version},
            {"$setOnInsert": dataset.model_dump(mode="python")},
            upsert=True,
        )
        stored = self.get_evaluation_dataset(dataset.dataset_version)
        if stored is None:
            raise RuntimeError("The evaluation dataset could not be read after persistence.")
        if stored.dataset_hash != dataset.dataset_hash:
            raise ValueError(
                f"Dataset version {dataset.dataset_version!r} already has a different hash."
            )
        return stored

    def get_evaluation_dataset(self, dataset_version: str) -> EvaluationDataset | None:
        document = self.datasets.find_one({"dataset_version": dataset_version}, {"_id": 0})
        return EvaluationDataset.model_validate(document) if document else None

    def list_evaluation_datasets(self) -> list[EvaluationDataset]:
        cursor = self.datasets.find({}, {"_id": 0}).sort("created_at", -1)
        return [EvaluationDataset.model_validate(document) for document in cursor]

    def save_evaluation_run(self, run: EvaluationRun) -> EvaluationRun:
        self.runs.replace_one(
            {"evaluation_run_id": run.evaluation_run_id},
            run.model_dump(mode="python"),
            upsert=True,
        )
        return run

    def get_evaluation_run(self, evaluation_run_id: str) -> EvaluationRun | None:
        document = self.runs.find_one(
            {"evaluation_run_id": evaluation_run_id},
            {"_id": 0},
        )
        return EvaluationRun.model_validate(document) if document else None

    def list_evaluation_runs(self) -> list[EvaluationRun]:
        cursor = self.runs.find({}, {"_id": 0}).sort("created_at", -1)
        return [EvaluationRun.model_validate(document) for document in cursor]

    def claim_evaluation_run(
        self,
        evaluation_run_id: str,
        execution_id: str | None = None,
    ) -> EvaluationRun | None:
        claimable: list[dict[str, object]] = [
            {"status": EvaluationStatus.QUEUED.value}
        ]
        if execution_id:
            claimable.append(
                {
                    "status": EvaluationStatus.RUNNING.value,
                    "worker_task_id": execution_id,
                }
            )
        document = self.runs.find_one_and_update(
            {
                "evaluation_run_id": evaluation_run_id,
                "$or": claimable,
            },
            {
                "$set": {
                    "status": EvaluationStatus.RUNNING.value,
                    "started_at": datetime.now(UTC),
                    "finished_at": None,
                    "safe_error_summary": None,
                    "worker_task_id": execution_id,
                }
            },
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
        return EvaluationRun.model_validate(_without_mongo_id(document)) if document else None

    def prepare_evaluation_retry(self, evaluation_run_id: str) -> EvaluationRun:
        document = self.runs.find_one_and_update(
            {
                "evaluation_run_id": evaluation_run_id,
                "status": EvaluationStatus.FAILED.value,
            },
            {
                "$set": {
                    "status": EvaluationStatus.QUEUED.value,
                    "started_at": None,
                    "finished_at": None,
                    "safe_error_summary": None,
                    "worker_task_id": None,
                },
                "$inc": {"attempt": 1},
            },
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
        if document:
            return EvaluationRun.model_validate(_without_mongo_id(document))
        existing = self.get_evaluation_run(evaluation_run_id)
        if existing is None:
            raise KeyError(evaluation_run_id)
        return existing
