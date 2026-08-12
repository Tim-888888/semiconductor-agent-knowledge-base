from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from semikb.config import Settings
from semikb.contracts.models import Chunk, ObjectRef
from semikb.rag_retrieval.milvus_schema import chunk_to_milvus_row
from semikb.storage.clients import StorageClientFactory, StorageConfigurationError
from semikb.storage.external import service_configuration_health
from semikb.storage.milvus_chunks import _alias_names
from semikb.storage.minio_artifacts import MinioArtifactRepository
from semikb.storage.mongo_index_migration import (
    MigrationSafetyError,
    build_migration_plan,
    capture_snapshot,
    migrate_mongo_indexes,
)
from semikb.storage.mongo_schema import MONGO_INDEX_SPECS, compare_index_information
from semikb.storage.preflight import run_preflight
from semikb.storage.t6_mongo_migration import (
    APPROVED_PRE_T6,
    TARGET_COLLECTIONS,
    T6MigrationSafetyError,
)
from semikb.storage.t6_mongo_migration import plan as build_t6_migration_plan
from semikb.storage.verifier import _public_bucket_policy


def blank_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "mongodb_uri": "",
        "milvus_uri": "",
        "minio_endpoint": "",
        "minio_access_key": "",
        "minio_secret_key": "",
        "redis_url": "",
        "mineru_api_base_url": "",
        "mineru_api_key": "",
        "llm_api_base_url": "",
        "llm_api_key": "",
        "llm_model": "",
        "aliyun_web_mcp_url": "",
        "aliyun_web_mcp_api_key": "",
        "aliyun_web_mcp_tool_name": "",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_missing_configuration_names_are_explicit_and_safe() -> None:
    health = service_configuration_health(blank_settings())

    assert all(not item.configured for item in health)
    assert "MONGODB_URI" in next(item.detail for item in health if item.name == "mongodb")
    assert "MINIO_ACCESS_KEY" in next(item.detail for item in health if item.name == "minio")
    assert all("not configured" in item.detail for item in health)


def test_llm_provider_health_uses_primary_and_fallback_configuration() -> None:
    health = service_configuration_health(
        blank_settings(
            llm_primary_provider="closeai",
            llm_fallback_provider="qwen",
            closeai_base_url="https://closeai.invalid/v1",
            closeai_api_key="closeai-secret",
            closeai_model="gpt-5.6-luna",
            qwen_api_base_url="https://qwen.invalid/v1",
            qwen_api_key="qwen-secret",
            qwen_model="qwen-flash",
        )
    )

    primary = next(item for item in health if item.name == "llm_primary")
    fallback = next(item for item in health if item.name == "llm_fallback")
    assert primary.configured is True
    assert fallback.configured is True
    assert "secret" not in primary.detail
    assert "secret" not in fallback.detail


def test_client_factory_rejects_incomplete_configuration_without_values() -> None:
    factory = StorageClientFactory(blank_settings())

    with pytest.raises(StorageConfigurationError, match="MONGODB_URI") as error:
        factory.create_mongodb()

    assert "mongodb://" not in str(error.value)


def test_preflight_does_not_attempt_unconfigured_connections() -> None:
    results = run_preflight(blank_settings())

    assert all(item.reachable is None for item in results)
    assert all(not item.configured for item in results)


def test_mongo_index_contract_keeps_identity_and_idempotency_unique_separately() -> None:
    ingestion_specs = {spec.name: spec for spec in MONGO_INDEX_SPECS["ingestion_jobs"]}

    assert ingestion_specs["job_id"].keys == (("job_id", 1),)
    assert ingestion_specs["job_id"].unique is True
    assert ingestion_specs["idempotency_key"].keys == (("idempotency_key", 1),)
    assert ingestion_specs["idempotency_key"].unique is True


def test_mongo_index_comparison_reports_missing_and_uniqueness_differences() -> None:
    expected = MONGO_INDEX_SPECS["document_catalog"]
    actual = {
        "document_id_revision": {
            "key": [("document_id", 1), ("revision", 1)],
            "unique": False,
        }
    }

    assert compare_index_information(expected, actual) == [
        "index uniqueness differs: document_id_revision"
    ]


def test_mongo_index_comparison_reports_unexpected_indexes() -> None:
    expected = MONGO_INDEX_SPECS["document_catalog"]
    actual = {
        "_id_": {"key": [("_id", 1)]},
        "document_id_revision": {
            "key": [("document_id", 1), ("revision", 1)],
            "unique": True,
        },
        "manual_index": {"key": [("manual", 1)]},
    }

    assert compare_index_information(expected, actual) == [
        "unexpected index: manual_index"
    ]


def test_milvus_row_normalizes_optional_fields_and_no_expiry() -> None:
    chunk = Chunk(
        chunk_id="DOC-R1-001",
        document_id="DOC",
        revision="R1",
        chunk_text="Synthetic evidence",
        page_or_section="正文",
        effective_at=datetime(2026, 1, 1, tzinfo=UTC),
        process_layer=None,
        chamber=None,
        recipe_id=None,
        recipe_version=None,
    )

    row = chunk_to_milvus_row(
        chunk,
        [0.1, 0.2, 0.3, 0.4],
        {1: 0.5},
        embedding_dim=4,
    )

    assert row["process_layer"] == ""
    assert row["recipe_version"] == ""
    assert row["expires_at"] == 0
    assert row["effective_at"] == 1767225600


def test_milvus_row_rejects_wrong_dense_dimension() -> None:
    chunk = Chunk(
        chunk_id="DOC-R1-001",
        document_id="DOC",
        revision="R1",
        chunk_text="Synthetic evidence",
        page_or_section="正文",
    )

    with pytest.raises(ValueError, match="dimension mismatch"):
        chunk_to_milvus_row(chunk, [0.1], {1: 0.5}, embedding_dim=4)


def test_milvus_alias_response_is_normalized_across_client_versions() -> None:
    assert _alias_names({"aliases": ["semikb_chunks_active"]}) == {
        "semikb_chunks_active"
    }
    assert _alias_names(["semikb_chunks_active"]) == {"semikb_chunks_active"}


def test_public_bucket_policy_detection() -> None:
    private_policy = '{"Statement":[{"Effect":"Allow","Principal":{"AWS":"arn:demo"}}]}'
    public_policy = '{"Statement":[{"Effect":"Allow","Principal":"*"}]}'

    assert _public_bucket_policy(private_policy) is False
    assert _public_bucket_policy(public_policy) is True


def test_minio_presigned_url_can_be_rewritten_to_same_origin_proxy() -> None:
    class FakeMinio:
        def presigned_get_object(self, bucket, object_key, *, expires, version_id):
            assert bucket == "semikb-derived"
            assert object_key == "documents/CASE/R1/assets/IMG/original.png"
            return (
                "http://minio:9000/semikb-derived/documents/CASE/R1/assets/IMG/original.png"
                "?X-Amz-Signature=abc123"
            )

    class FakeFactory:
        def create_minio(self):
            return FakeMinio()

    repository = MinioArtifactRepository(FakeFactory(), public_base_url="/objects")
    reference = ObjectRef(
        bucket="semikb-derived",
        object_key="documents/CASE/R1/assets/IMG/original.png",
        content_type="image/png",
        sha256="0" * 64,
    )

    url = repository.presign_get(reference, expires=timedelta(minutes=5))

    assert url.startswith("/objects/semikb-derived/documents/CASE/R1/assets/IMG/original.png?")
    assert "X-Amz-Signature=abc123" in url
    assert "minio:9000" not in url


def migration_snapshot(
    *,
    document_count: int = 0,
    extra_document_index: dict[str, object] | None = None,
) -> dict[str, object]:
    collections: dict[str, object] = {}
    for collection_name, specs in MONGO_INDEX_SPECS.items():
        indexes = [
            {"name": "_id_", "keys": [["_id", 1]], "unique": False},
            *[
                {
                    "name": spec.name,
                    "keys": [list(item) for item in spec.keys],
                    "unique": spec.unique,
                }
                for spec in specs
            ],
        ]
        if collection_name == "document_catalog" and extra_document_index:
            indexes.append(extra_document_index)
        collections[collection_name] = {
            "document_count": document_count if collection_name == "document_catalog" else 0,
            "indexes": indexes,
        }
    return {
        "schema_version": 1,
        "database": "semikb",
        "collections": collections,
    }


def test_mongo_index_migration_is_idempotent_for_current_contract() -> None:
    assert build_migration_plan(migration_snapshot()) == []


def test_mongo_index_migration_replaces_approved_legacy_definition() -> None:
    snapshot = migration_snapshot()
    snapshot["collections"]["document_catalog"]["indexes"] = [
        {"name": "_id_", "keys": [["_id", 1]], "unique": False},
        {
            "name": "document_id_revision",
            "keys": [["document_id", 1], ["revision", 1]],
            "unique": False,
        },
    ]

    plan = build_migration_plan(snapshot)

    assert [(item.operation, item.index_name) for item in plan] == [
        ("drop", "document_id_revision"),
        ("create", "document_id_revision"),
    ]


def test_mongo_index_migration_refuses_non_empty_collection() -> None:
    with pytest.raises(MigrationSafetyError, match="contains 1 documents"):
        build_migration_plan(migration_snapshot(document_count=1))


def test_mongo_index_migration_refuses_unknown_index() -> None:
    unknown = {
        "name": "manual_index",
        "keys": [["manual", 1]],
        "unique": False,
    }

    with pytest.raises(MigrationSafetyError, match="unapproved indexes: manual_index"):
        build_migration_plan(migration_snapshot(extra_document_index=unknown))


class FakeMongoCollection:
    def __init__(
        self,
        state: dict[str, object],
        *,
        fail_once_on_create: str | None = None,
    ) -> None:
        self.document_count = int(state["document_count"])
        self.indexes = {
            item["name"]: {
                "key": [tuple(pair) for pair in item["keys"]],
                "unique": item["unique"],
            }
            for item in state["indexes"]
        }
        self.fail_once_on_create = fail_once_on_create

    def count_documents(self, query: dict[str, object]) -> int:
        assert query == {}
        return self.document_count

    def index_information(self) -> dict[str, dict[str, object]]:
        return deepcopy(self.indexes)

    def drop_index(self, name: str) -> None:
        del self.indexes[name]

    def create_index(
        self,
        keys: list[tuple[str, int]],
        *,
        name: str,
        unique: bool,
    ) -> str:
        if self.fail_once_on_create == name:
            self.fail_once_on_create = None
            raise RuntimeError("injected index creation failure")
        self.indexes[name] = {"key": list(keys), "unique": unique}
        return name


class FakeMongoDatabase:
    name = "semikb"

    def __init__(
        self,
        snapshot: dict[str, Any],
        *,
        fail_collection: str | None = None,
        fail_index: str | None = None,
    ) -> None:
        self.collections = {
            name: FakeMongoCollection(
                state,
                fail_once_on_create=fail_index if name == fail_collection else None,
            )
            for name, state in snapshot["collections"].items()
        }

    def __getitem__(self, name: str) -> FakeMongoCollection:
        return self.collections[name]

    def list_collection_names(self) -> list[str]:
        return list(self.collections)


class FakeMongoClient:
    def __init__(self, database: FakeMongoDatabase) -> None:
        self.database = database

    def __getitem__(self, name: str) -> FakeMongoDatabase:
        assert name == "semikb"
        return self.database


class FakeMongoFactory:
    def __init__(self, database: FakeMongoDatabase) -> None:
        self.client = FakeMongoClient(database)

    @contextmanager
    def mongodb(self) -> Iterator[FakeMongoClient]:
        yield self.client


def legacy_document_index_snapshot() -> dict[str, Any]:
    snapshot = migration_snapshot()
    snapshot["collections"]["document_catalog"]["indexes"] = [
        {"name": "_id_", "keys": [["_id", 1]], "unique": False},
        {
            "name": "document_id_revision",
            "keys": [["document_id", 1], ["revision", 1]],
            "unique": False,
        },
    ]
    return snapshot


def test_mongo_index_migration_applies_and_becomes_idempotent(tmp_path: Path) -> None:
    database = FakeMongoDatabase(legacy_document_index_snapshot())

    result = migrate_mongo_indexes(
        blank_settings(mongodb_uri="mongodb://configured", mongodb_database="semikb"),
        apply=True,
        snapshot_path=tmp_path / "before.json",
        factory=FakeMongoFactory(database),
    )

    assert result["status"] == "migrated"
    assert build_migration_plan(capture_snapshot(database)) == []
    assert (tmp_path / "before.json").is_file()


def test_mongo_index_migration_restores_snapshot_after_failure(tmp_path: Path) -> None:
    database = FakeMongoDatabase(
        legacy_document_index_snapshot(),
        fail_collection="document_catalog",
        fail_index="document_id_revision",
    )

    with pytest.raises(RuntimeError, match="previous indexes restored"):
        migrate_mongo_indexes(
            blank_settings(mongodb_uri="mongodb://configured", mongodb_database="semikb"),
            apply=True,
            snapshot_path=tmp_path / "before.json",
            factory=FakeMongoFactory(database),
        )

    restored = database["document_catalog"].index_information()["document_id_revision"]
    assert restored["unique"] is False


def test_mongo_index_migration_refuses_missing_live_collection() -> None:
    database = FakeMongoDatabase(migration_snapshot())
    del database.collections["audit_events"]

    with pytest.raises(MigrationSafetyError, match="missing collections: audit_events"):
        capture_snapshot(database)


def t6_migration_snapshot(*, desired: bool, non_empty: str | None = None) -> dict[str, Any]:
    collections: dict[str, Any] = {}
    for name in TARGET_COLLECTIONS:
        specs = MONGO_INDEX_SPECS[name] if desired else APPROVED_PRE_T6[name]
        collections[name] = {
            "document_count": 1 if name == non_empty else 0,
            "indexes": [
                {
                    "name": spec.name,
                    "keys": [list(item) for item in spec.keys],
                    "unique": spec.unique,
                }
                for spec in specs
            ],
        }
    return {"schema_version": 1, "database": "semikb", "collections": collections}


def test_t6_migration_replaces_only_approved_empty_collection_indexes() -> None:
    actions = build_t6_migration_plan(t6_migration_snapshot(desired=False))

    assert len(actions) == 8
    assert {action.collection for action in actions} == set(TARGET_COLLECTIONS)
    assert {action.operation for action in actions} == {"drop", "create"}


def test_t6_migration_is_idempotent_for_current_contract() -> None:
    assert build_t6_migration_plan(t6_migration_snapshot(desired=True)) == []


def test_t6_migration_refuses_non_empty_checkpoint_collection() -> None:
    with pytest.raises(T6MigrationSafetyError, match="checkpoints must be empty"):
        build_t6_migration_plan(
            t6_migration_snapshot(desired=False, non_empty="checkpoints")
        )
