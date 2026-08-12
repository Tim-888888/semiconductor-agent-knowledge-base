from __future__ import annotations

from types import SimpleNamespace

import pytest

from semikb.storage.embedding_index_migration import (
    EmbeddingIndexMigrator,
    EmbeddingMigrationPlan,
    _single_catalog_value,
)


def _plan(*, target_exists: bool) -> EmbeddingMigrationPlan:
    return EmbeddingMigrationPlan(
        source_collection="semikb_chunks_v3",
        source_index_version="v3",
        source_embedding_version="qwen3.7-text-embedding",
        target_collection="semikb_chunks_v4",
        target_index_version="v4",
        target_embedding_version="qwen3.7-text-embedding",
        target_output_type="dense&sparse",
        target_sparse_encoder_version="qwen3.7-text-embedding-sparse-v1",
        embedding_dim=1024,
        published_documents=3,
        published_chunks=8,
        target_exists=target_exists,
        alias_switch_required=True,
    )


class _Vectors:
    def __init__(self) -> None:
        self.upsert_calls = 0
        self.activate_calls = 0

    def upsert_chunks(self, *args, **kwargs) -> None:
        self.upsert_calls += 1

    def activate_alias(self, index_version: str) -> None:
        assert index_version == "v4"
        self.activate_calls += 1


def _bare_migrator(monkeypatch: pytest.MonkeyPatch, plan: EmbeddingMigrationPlan):
    migrator = object.__new__(EmbeddingIndexMigrator)
    migrator.target_index_version = "v4"
    migrator.settings = SimpleNamespace(
        embedding_version="qwen3.7-text-embedding",
        embedding_dim=1024,
        embedding_output_type="dense&sparse",
        sparse_encoder_version="qwen3.7-text-embedding-sparse-v1",
    )
    migrator.vectors = _Vectors()
    monkeypatch.setattr(migrator, "plan", lambda: plan)
    monkeypatch.setattr(migrator, "_load_published_chunks", lambda: [object()] * 8)
    monkeypatch.setattr(migrator, "_verify_catalog_snapshot", lambda *_: None)
    monkeypatch.setattr(migrator, "_verify_rows", lambda *_: None)
    monkeypatch.setattr(migrator, "_verify_dense_probe", lambda *_: None)
    return migrator


def test_catalog_versions_must_match_between_documents_and_chunks() -> None:
    assert _single_catalog_value(["v3"], ["v3"], "index_version") == "v3"
    with pytest.raises(RuntimeError, match="disagree on index_version"):
        _single_catalog_value(["v3"], ["v2"], "index_version")


def test_build_records_candidate_without_switching_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(target_exists=False)
    migrator = _bare_migrator(monkeypatch, plan)
    monkeypatch.setattr(
        "semikb.storage.embedding_index_migration.provision_milvus_index",
        lambda *_: {"collection": "semikb_chunks_v4"},
    )
    monkeypatch.setattr(migrator, "_encode_chunks", lambda chunks: [object()] * 8)
    monkeypatch.setattr(migrator, "_record_candidate", lambda *_: None)
    monkeypatch.setattr(migrator, "_active_collection", lambda: plan.source_collection)

    result = migrator.build()

    assert result["status"] == "candidate_built"
    assert migrator.vectors.upsert_calls == 1
    assert migrator.vectors.activate_calls == 0


def test_publish_failure_restores_alias_and_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(target_exists=True)
    migrator = _bare_migrator(monkeypatch, plan)
    restored: list[str] = []
    monkeypatch.setattr(migrator, "_verify_candidate_release", lambda *_: None)
    monkeypatch.setattr(
        migrator,
        "_update_catalog_versions",
        lambda *_: (_ for _ in ()).throw(RuntimeError("catalog update failed")),
    )
    monkeypatch.setattr(
        migrator,
        "_restore_alias",
        lambda collection: restored.append(f"alias:{collection}"),
    )
    monkeypatch.setattr(
        migrator,
        "_restore_catalog_versions",
        lambda *_: restored.append("catalog"),
    )

    with pytest.raises(RuntimeError, match="catalog update failed"):
        migrator.publish()

    assert migrator.vectors.activate_calls == 1
    assert restored == ["alias:semikb_chunks_v3", "catalog"]
