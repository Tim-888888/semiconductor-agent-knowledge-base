"""Create the project-owned storage resources after a successful preflight.

Run manually against a least-privilege service account. This module never creates
databases outside the ``semikb`` namespace and never drops data or collections.
"""

from __future__ import annotations

import json

from pymilvus import DataType, MilvusClient

from semikb.config import Settings, get_settings
from semikb.rag_retrieval.milvus_schema import collection_name
from semikb.storage.clients import StorageClientFactory, missing_storage_settings
from semikb.storage.mongo_schema import MONGO_INDEX_SPECS, compare_index_information


def provision(
    settings: Settings,
    *,
    index_version: str = "v1",
    allow_mongo_index_changes: bool = False,
) -> dict[str, object]:
    """Ensure required buckets, catalog indexes, and Milvus index version exist."""

    _require_storage_configuration(settings)
    mongo_created = _provision_mongodb(
        settings,
        allow_index_changes=allow_mongo_index_changes,
    )
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
        name
        for service in ("mongodb", "milvus", "minio")
        for name in missing_storage_settings(settings, service)
    ]
    if missing:
        raise RuntimeError("Storage provisioning requires: " + ", ".join(missing))


def _provision_mongodb(settings: Settings, *, allow_index_changes: bool) -> list[str]:
    factory = StorageClientFactory(settings)
    with factory.mongodb() as client:
        database = client[settings.mongodb_database]
        differences: list[str] = []
        for collection_name_value, specs in MONGO_INDEX_SPECS.items():
            if collection_name_value not in database.list_collection_names():
                continue
            actual = database[collection_name_value].index_information()
            differences.extend(
                f"{collection_name_value}: {difference}"
                for difference in compare_index_information(specs, actual)
            )
        if differences and not allow_index_changes:
            raise RuntimeError(
                "MongoDB index differences require an approved migration; run the read-only verifier first."
            )

        created: list[str] = []
        for collection_name_value, specs in MONGO_INDEX_SPECS.items():
            collection = database[collection_name_value]
            for spec in specs:
                collection.create_index(list(spec.keys), name=spec.name, unique=spec.unique)
            created.append(collection_name_value)
        return created


def _provision_minio(settings: Settings) -> list[str]:
    client = StorageClientFactory(settings).create_minio()
    created: list[str] = []
    for bucket in ("semikb-raw", "semikb-derived"):
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            created.append(bucket)
    return created


def _provision_milvus(settings: Settings, index_version: str) -> dict[str, str]:
    collection = collection_name(index_version)
    factory = StorageClientFactory(settings)
    with factory.milvus() as client:
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
