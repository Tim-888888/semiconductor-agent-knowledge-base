"""Desired MongoDB indexes expressed separately so they can be verified before migration."""

from __future__ import annotations

from dataclasses import dataclass

from pymongo import ASCENDING, DESCENDING


@dataclass(frozen=True, slots=True)
class MongoIndexSpec:
    name: str
    keys: tuple[tuple[str, int], ...]
    unique: bool = False


def _index(name: str, *fields: str, unique: bool = False) -> MongoIndexSpec:
    return MongoIndexSpec(
        name=name,
        keys=tuple((field, ASCENDING) for field in fields),
        unique=unique,
    )


def _checkpoint_index(name: str, *, writes: bool = False) -> MongoIndexSpec:
    keys = [
        ("thread_id", ASCENDING),
        ("checkpoint_ns", ASCENDING),
        ("checkpoint_id", DESCENDING),
    ]
    if writes:
        keys.extend((("task_id", ASCENDING), ("idx", ASCENDING)))
    return MongoIndexSpec(name=name, keys=tuple(keys), unique=True)


MONGO_INDEX_SPECS: dict[str, tuple[MongoIndexSpec, ...]] = {
    "document_catalog": (
        _index("document_id_revision", "document_id", "revision", unique=True),
        _index(
            "lifecycle_approval_created_at",
            "lifecycle",
            "approval_status",
            "created_at",
        ),
        _index("source_manifest_ref", "source_id", "source_manifest_version"),
    ),
    "source_manifests": (
        _index("source_id_manifest_version", "source_id", "manifest_version", unique=True),
        _index("status_created_at", "status", "created_at"),
        _index("source_hash", "source_hash"),
    ),
    "document_lifecycle_operations": (
        _index("operation_id", "operation_id", unique=True),
        _index("request_id", "request_id", unique=True),
        _index(
            "document_revision_created_at",
            "selector.document_id",
            "selector.revision",
            "created_at",
        ),
        _index("status_updated_at", "status", "updated_at"),
    ),
    "chunk_catalog": (
        _index("chunk_id", "chunk_id", unique=True),
        _index("document_id_revision", "document_id", "revision"),
    ),
    "image_assets": (
        _index("image_id", "image_id", unique=True),
        _index("document_id_revision", "document_id", "revision"),
    ),
    "table_assets": (
        _index("table_id", "table_id", unique=True),
        _index("document_id_revision", "document_id", "revision"),
    ),
    "ingestion_jobs": (
        _index("job_id", "job_id", unique=True),
        _index("idempotency_key", "idempotency_key", unique=True),
        _index("status_created_at", "status", "created_at"),
    ),
    "ingestion_job_events": (
        _index("job_id_created_at", "job_id", "created_at"),
    ),
    "corpus_standardization_jobs": (
        _index("job_id", "job_id", unique=True),
        _index("idempotency_key", "idempotency_key", unique=True),
        _index(
            "corpus_snapshot_version",
            "metadata.corpus_id",
            "metadata.snapshot_version",
            unique=True,
        ),
        _index("status_created_at", "status", "created_at"),
    ),
    "corpus_publication_batches": (
        _index("batch_id", "batch_id", unique=True),
        _index("review_request_id", "review.request_id", unique=True),
        _index(
            "standardization_job_status_created_at",
            "review.standardization_job_id",
            "status",
            "created_at",
        ),
    ),
    "index_releases": (
        _index("index_version", "index_version", unique=True),
    ),
    "retrieval_traces": (
        _index("trace_id", "trace_id", unique=True),
        _index("actor_user_id_created_at", "actor_user_id", "created_at"),
    ),
    "evaluation_datasets": (
        _index("dataset_version", "dataset_version", unique=True),
        _index("purpose_created_at", "purpose", "created_at"),
    ),
    "evaluation_release_freezes": (
        _index("freeze_id", "freeze_id", unique=True),
        _index("release_version", "release_version", unique=True),
        _index("holdout_hash_status", "holdout_dataset_hash", "status"),
    ),
    "evaluation_runs": (
        _index("evaluation_run_id", "evaluation_run_id", unique=True),
        _index("created_at", "created_at"),
    ),
    "agent_threads": (
        _index("thread_id", "thread_id", unique=True),
        _index("actor_user_id_updated_at", "actor_scope.user_id", "updated_at"),
    ),
    "agent_message_requests": (
        _index(
            "actor_thread_request",
            "actor_user_id",
            "thread_id",
            "request_id",
            unique=True,
        ),
        _index("status_updated_at", "status", "updated_at"),
    ),
    "checkpoints": (
        _checkpoint_index("thread_checkpoint_namespace_id"),
    ),
    "checkpoint_writes": (
        _checkpoint_index("thread_checkpoint_namespace_task_write", writes=True),
    ),
    "long_term_memories": (
        _index("namespace_str_key", "namespace_str", "key", unique=True),
    ),
    "audit_events": (
        _index("event_id", "event_id", unique=True),
        _index("actor_user_id_created_at", "actor_user_id", "created_at"),
        _index(
            "resource_created_at",
            "details.resource_type",
            "details.resource_id",
            "created_at",
        ),
    ),
}


def compare_index_information(
    expected: tuple[MongoIndexSpec, ...],
    actual: dict[str, dict[str, object]],
) -> list[str]:
    """Return safe structural differences without including indexed document values."""

    differences: list[str] = []
    for spec in expected:
        current = actual.get(spec.name)
        if current is None:
            differences.append(f"missing index: {spec.name}")
            continue
        actual_keys = tuple(tuple(item) for item in current.get("key", []))
        actual_unique = bool(current.get("unique", False))
        if actual_keys != spec.keys:
            differences.append(f"index keys differ: {spec.name}")
        if actual_unique != spec.unique:
            differences.append(f"index uniqueness differs: {spec.name}")
    expected_names = {spec.name for spec in expected}
    unexpected_names = sorted(set(actual).difference(expected_names, {"_id_"}))
    differences.extend(f"unexpected index: {name}" for name in unexpected_names)
    return differences
