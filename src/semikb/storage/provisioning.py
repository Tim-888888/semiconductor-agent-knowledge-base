"""Create the project-owned storage resources after a successful preflight.

Run manually against a least-privilege service account. This module never creates
databases outside the ``semikb`` namespace and never drops data or collections.
"""

from __future__ import annotations

import json

from minio import Minio
from pymilvus import DataType, MilvusClient
from pymongo import ASCENDING, MongoClient

from semikb.config import Settings, get_settings
from semikb.rag_retrieval.milvus_schema import collection_name

MONGO_INDEXES: dict[str, list[tuple[str, int]]] = {
    "document_catalog": [("document_id", ASCENDING), ("revision", ASCENDING)],
    "chunk_catalog": [("chunk_id", ASCENDING)],
    "image_assets": [("image_id", ASCENDING)],
    "ingestion_jobs": [("job_id", ASCENDING), ("idempotency_key", ASCENDING)],
    "ingestion_job_events": [("job_id", ASCENDING), ("created_at", ASCENDING)],
    "index_releases": [("index_version", ASCENDING)],
    "retrieval_traces": [("trace_id", ASCENDING), ("actor_user_id", ASCENDING), ("created_at", ASCENDING)],
    "evaluation_datasets": [("dataset_version", ASCENDING)],
    "evaluation_runs": [("evaluation_run_id", ASCENDING), ("created_at", ASCENDING)],
    "agent_threads": [("thread_id", ASCENDING), ("actor_scope.user_id", ASCENDING), ("updated_at", ASCENDING)],
    "checkpoints": [("thread_id", ASCENDING), ("checkpoint_id", ASCENDING)],
    "checkpoint_writes": [("thread_id", ASCENDING), ("checkpoint_id", ASCENDING)],
    "long_term_memories": [("memory_id", ASCENDING), ("user_id", ASCENDING)],
    "audit_events": [("created_at", ASCENDING), ("actor_user_id", ASCENDING)],
}


def provision(settings: Settings, *, index_version: str = "v1") -> dict[str, object]:
    """Ensure required buckets, catalog indexes, and Milvus index version exist."""

    _require_storage_configuration(settings)
    mongo_created = _provision_mongodb(settings)
    buckets_created = _provision_minio(settings)
    milvus_result = _provision_milvus(settings, index_version)
    return {
        "mongodb_collections": mongo_created,
        "minio_buckets_created": buckets_created,
        "milvus": milvus_result,
        "next_step": "Populate staged data, validate counts, then explicitly switch the active Milvus alias.",
    }


def _require_storage_configuration(settings: Settings) -> None:
    missing = [
        key
        for key, value in {
            "MONGODB_URI": settings.mongodb_uri,
            "MILVUS_URI": settings.milvus_uri,
            "MINIO_ENDPOINT": settings.minio_endpoint,
            "MINIO_ACCESS_KEY": settings.minio_access_key,
            "MINIO_SECRET_KEY": settings.minio_secret_key,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError("Storage provisioning requires: " + ", ".join(missing))


def _provision_mongodb(settings: Settings) -> list[str]:
    client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
    database = client[settings.mongodb_database]
    created: list[str] = []
    for collection_name_value, keys in MONGO_INDEXES.items():
        collection = database[collection_name_value]
        collection.create_index(keys, name="_".join(key for key, _ in keys), unique=collection_name_value in {"chunk_catalog", "image_assets", "ingestion_jobs", "index_releases", "agent_threads"})
        created.append(collection_name_value)
    return created


def _provision_minio(settings: Settings) -> list[str]:
    client = Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
    created: list[str] = []
    for bucket in ("semikb-raw", "semikb-derived"):
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            created.append(bucket)
    return created


def _provision_milvus(settings: Settings, index_version: str) -> dict[str, str]:
    collection = collection_name(index_version)
    client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token or None)
    if not client.has_collection(collection):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=160)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=settings.embedding_dim)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        for field in ("document_id", "revision", "chunk_type", "approval_status", "fab", "product", "process_layer", "tool_id", "chamber", "recipe_id", "recipe_version", "access_scope_key", "index_version"):
            schema.add_field(field, DataType.VARCHAR, max_length=160)
        schema.add_field("effective_at", DataType.INT64)
        schema.add_field("expires_at", DataType.INT64)
        index_params = client.prepare_index_params()
        index_params.add_index("dense_vector", index_type="HNSW", metric_type="IP", params={"M": 16, "efConstruction": 200})
        index_params.add_index("sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP", params={"drop_ratio_build": 0.2})
        client.create_collection(collection_name=collection, schema=schema, index_params=index_params)
        status = "created"
    else:
        status = "already_exists"
    return {"collection": collection, "status": status, "alias": "not_switched"}


def main() -> None:
    print(json.dumps(provision(get_settings()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
