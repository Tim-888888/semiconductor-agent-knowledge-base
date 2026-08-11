"""Read-only verification of project-owned datastore resources."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from minio.error import S3Error

from semikb.config import Settings, get_settings
from semikb.rag_retrieval.milvus_schema import MILVUS_STRING_FIELDS, collection_name
from semikb.storage.clients import StorageClientFactory
from semikb.storage.mongo_schema import MONGO_INDEX_SPECS, compare_index_information


@dataclass(frozen=True, slots=True)
class ResourceCheck:
    component: str
    check: str
    passed: bool
    detail: str


def _check(component: str, name: str, passed: bool, detail: str) -> ResourceCheck:
    return ResourceCheck(component=component, check=name, passed=passed, detail=detail)


def _public_bucket_policy(policy: str) -> bool:
    document = json.loads(policy)
    for statement in document.get("Statement", []):
        principal = statement.get("Principal")
        if statement.get("Effect") == "Allow" and principal == "*":
            return True
        if statement.get("Effect") == "Allow" and isinstance(principal, dict):
            if principal.get("AWS") == "*":
                return True
    return False


def _verify_mongodb(factory: StorageClientFactory, database_name: str) -> list[ResourceCheck]:
    checks: list[ResourceCheck] = []
    with factory.mongodb() as client:
        database = client[database_name]
        existing = set(database.list_collection_names())
        missing_collections = sorted(set(MONGO_INDEX_SPECS).difference(existing))
        checks.append(
            _check(
                "mongodb",
                "collections",
                not missing_collections,
                "all expected collections exist"
                if not missing_collections
                else f"missing collections: {', '.join(missing_collections)}",
            )
        )
        for name, specs in MONGO_INDEX_SPECS.items():
            if name not in existing:
                continue
            differences = compare_index_information(specs, database[name].index_information())
            checks.append(
                _check(
                    "mongodb",
                    f"indexes:{name}",
                    not differences,
                    "indexes match desired contract"
                    if not differences
                    else "; ".join(differences),
                )
            )
    return checks


def _verify_minio(factory: StorageClientFactory) -> list[ResourceCheck]:
    client = factory.create_minio()
    checks: list[ResourceCheck] = []
    for bucket in ("semikb-raw", "semikb-derived"):
        exists = client.bucket_exists(bucket)
        checks.append(
            _check(
                "minio",
                f"bucket:{bucket}",
                exists,
                "bucket exists" if exists else "bucket is missing",
            )
        )
        if not exists:
            continue
        try:
            policy = client.get_bucket_policy(bucket)
            is_public = _public_bucket_policy(policy)
            detail = "bucket policy is private" if not is_public else "public allow policy detected"
        except S3Error as exc:
            if exc.code not in {"NoSuchBucketPolicy", "NoSuchPolicy"}:
                raise
            is_public = False
            detail = "no public bucket policy"
        checks.append(_check("minio", f"privacy:{bucket}", not is_public, detail))
    return checks


def _field_type_name(field: dict[str, Any]) -> str:
    value = field.get("type")
    return getattr(value, "name", str(value))


def _verify_milvus(
    factory: StorageClientFactory,
    embedding_dim: int,
    index_version: str,
    require_active_alias: bool,
) -> list[ResourceCheck]:
    expected_collection = collection_name(index_version)
    with factory.milvus() as client:
        exists = client.has_collection(expected_collection)
        checks = [
            _check(
                "milvus",
                "collection",
                exists,
                f"{expected_collection} exists" if exists else f"{expected_collection} is missing",
            )
        ]
        if not exists:
            return checks

        description = client.describe_collection(expected_collection)
        fields = {field["name"]: field for field in description.get("fields", [])}
        expected_fields = {
            "chunk_id",
            "dense_vector",
            "sparse_vector",
            "effective_at",
            "expires_at",
            *MILVUS_STRING_FIELDS,
        }
        missing_fields = sorted(expected_fields.difference(fields))
        dense_dim = int(fields.get("dense_vector", {}).get("params", {}).get("dim", -1))
        checks.append(
            _check(
                "milvus",
                "schema",
                not missing_fields and dense_dim == embedding_dim,
                "field set and dense dimension match"
                if not missing_fields and dense_dim == embedding_dim
                else f"missing fields={missing_fields}; dense_dim={dense_dim}",
            )
        )
        expected_types = {
            "chunk_id": "VARCHAR",
            "dense_vector": "FLOAT_VECTOR",
            "sparse_vector": "SPARSE_FLOAT_VECTOR",
            "effective_at": "INT64",
            "expires_at": "INT64",
        }
        type_differences = [
            name
            for name, expected in expected_types.items()
            if name in fields and _field_type_name(fields[name]) != expected
        ]
        checks.append(
            _check(
                "milvus",
                "field_types",
                not type_differences,
                "field types match" if not type_differences else f"type mismatch: {', '.join(type_differences)}",
            )
        )
        index_names = set(client.list_indexes(expected_collection))
        index_details = {
            name: client.describe_index(expected_collection, name) for name in index_names
        }
        indexes_ok = (
            index_details.get("dense_vector", {}).get("index_type") == "HNSW"
            and index_details.get("sparse_vector", {}).get("index_type")
            == "SPARSE_INVERTED_INDEX"
        )
        checks.append(
            _check(
                "milvus",
                "indexes",
                indexes_ok,
                "dense HNSW and sparse inverted indexes match"
                if indexes_ok
                else "required vector indexes are missing or have the wrong type",
            )
        )
        aliases_value = client.list_aliases(collection_name=expected_collection)
        aliases = (
            aliases_value.get("aliases", [])
            if isinstance(aliases_value, dict)
            else aliases_value
        )
        alias_present = "semikb_chunks_active" in aliases
        checks.append(
            _check(
                "milvus",
                "active_alias",
                alias_present or not require_active_alias,
                "active alias points to collection"
                if alias_present
                else "active alias is not required before the first validated publication",
            )
        )
        return checks


def _verify_redis(factory: StorageClientFactory) -> list[ResourceCheck]:
    with factory.redis() as client:
        ping = bool(client.ping())
        appendonly = client.config_get("appendonly").get("appendonly") == "yes"
        appendfsync = client.config_get("appendfsync").get("appendfsync") == "everysec"
        protected = client.config_get("protected-mode").get("protected-mode") == "yes"
        password_enabled = bool(client.config_get("requirepass").get("requirepass"))
        return [
            _check("redis", "ping", ping, "authenticated ping succeeded" if ping else "ping failed"),
            _check(
                "redis",
                "persistence",
                appendonly and appendfsync,
                "AOF with appendfsync everysec enabled"
                if appendonly and appendfsync
                else "AOF persistence settings differ",
            ),
            _check(
                "redis",
                "security",
                protected and password_enabled,
                "protected mode and password are enabled"
                if protected and password_enabled
                else "protected mode or password is missing",
            ),
        ]


def verify_resources(
    settings: Settings,
    *,
    index_version: str | None = None,
    require_active_alias: bool | None = None,
    factory: StorageClientFactory | None = None,
) -> list[ResourceCheck]:
    """Verify resources without creating, dropping, updating, or inserting anything."""

    client_factory = factory or StorageClientFactory(settings)
    target_index_version = index_version or settings.milvus_index_version
    active_alias_required = (
        settings.milvus_require_active_alias
        if require_active_alias is None
        else require_active_alias
    )
    checks: list[ResourceCheck] = []
    verifiers = (
        lambda: _verify_mongodb(client_factory, settings.mongodb_database),
        lambda: _verify_minio(client_factory),
        lambda: _verify_milvus(
            client_factory,
            settings.embedding_dim,
            target_index_version,
            active_alias_required,
        ),
        lambda: _verify_redis(client_factory),
    )
    for verifier in verifiers:
        try:
            checks.extend(verifier())
        except Exception as exc:  # pragma: no cover - depends on external infrastructure
            checks.append(
                _check(
                    "external",
                    "connection",
                    False,
                    f"{type(exc).__name__}: check configuration and service availability",
                )
            )
    return checks


def main() -> None:
    checks = verify_resources(get_settings())
    print(json.dumps([asdict(item) for item in checks], ensure_ascii=False, indent=2))
    if not all(item.passed for item in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
