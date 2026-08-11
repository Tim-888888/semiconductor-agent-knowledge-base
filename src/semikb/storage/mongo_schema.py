"""Desired MongoDB indexes expressed separately so they can be verified before migration."""

from __future__ import annotations

from dataclasses import dataclass

from pymongo import ASCENDING


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


MONGO_INDEX_SPECS: dict[str, tuple[MongoIndexSpec, ...]] = {
    "document_catalog": (
        _index("document_id_revision", "document_id", "revision", unique=True),
    ),
    "chunk_catalog": (
        _index("chunk_id", "chunk_id", unique=True),
        _index("document_id_revision", "document_id", "revision"),
    ),
    "image_assets": (
        _index("image_id", "image_id", unique=True),
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
    "index_releases": (
        _index("index_version", "index_version", unique=True),
    ),
    "retrieval_traces": (
        _index("trace_id", "trace_id", unique=True),
        _index("actor_user_id_created_at", "actor_user_id", "created_at"),
    ),
    "evaluation_datasets": (
        _index("dataset_version", "dataset_version", unique=True),
    ),
    "evaluation_runs": (
        _index("evaluation_run_id", "evaluation_run_id", unique=True),
        _index("created_at", "created_at"),
    ),
    "agent_threads": (
        _index("thread_id", "thread_id", unique=True),
        _index("actor_user_id_updated_at", "actor_scope.user_id", "updated_at"),
    ),
    "checkpoints": (
        _index("thread_id_checkpoint_id", "thread_id", "checkpoint_id"),
    ),
    "checkpoint_writes": (
        _index("thread_id_checkpoint_id", "thread_id", "checkpoint_id"),
    ),
    "long_term_memories": (
        _index("memory_id", "memory_id", unique=True),
        _index("user_id", "user_id"),
    ),
    "audit_events": (
        _index("actor_user_id_created_at", "actor_user_id", "created_at"),
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
    return differences
