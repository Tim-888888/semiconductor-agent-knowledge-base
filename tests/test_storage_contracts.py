from __future__ import annotations

from datetime import UTC, datetime

import pytest

from semikb.config import Settings
from semikb.contracts.models import Chunk
from semikb.rag_retrieval.milvus_schema import chunk_to_milvus_row
from semikb.storage.clients import StorageClientFactory, StorageConfigurationError
from semikb.storage.external import service_configuration_health
from semikb.storage.mongo_schema import MONGO_INDEX_SPECS, compare_index_information
from semikb.storage.preflight import run_preflight
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


def test_public_bucket_policy_detection() -> None:
    private_policy = '{"Statement":[{"Effect":"Allow","Principal":{"AWS":"arn:demo"}}]}'
    public_policy = '{"Statement":[{"Effect":"Allow","Principal":"*"}]}'

    assert _public_bucket_policy(private_policy) is False
    assert _public_bucket_policy(public_policy) is True
