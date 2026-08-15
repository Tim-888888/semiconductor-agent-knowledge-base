from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest

from semikb.config import Settings
from semikb.storage.mongo_schema import MONGO_INDEX_SPECS
from semikb.storage.t9446a_source_governance_migration import (
    MIGRATION_VERSION,
    SourceGovernanceMigrationSafetyError,
    build_migration_plan,
    capture_snapshot,
    migrate,
    rollback,
)


class FakeCollection:
    def __init__(self, database: FakeDatabase, name: str) -> None:
        self.database = database
        self.name = name
        self.document_count = 0
        self.indexes: dict[str, dict[str, object]] = {
            "_id_": {"key": [("_id", 1)], "unique": False}
        }
        self.fail_once_on_create: str | None = None

    def count_documents(self, query: dict[str, object]) -> int:
        assert query == {}
        return self.document_count

    def index_information(self) -> dict[str, dict[str, object]]:
        return deepcopy(self.indexes)

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

    def drop_index(self, name: str) -> None:
        del self.indexes[name]

    def drop(self) -> None:
        self.database.collections.pop(self.name, None)


class FakeDatabase:
    name = "semikb"

    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}

    def list_collection_names(self) -> list[str]:
        return list(self.collections)

    def __getitem__(self, name: str) -> FakeCollection:
        if name not in self.collections:
            self.collections[name] = FakeCollection(self, name)
        return self.collections[name]


class FakeClient:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database

    def __getitem__(self, name: str) -> FakeDatabase:
        assert name == "semikb"
        return self.database


class FakeFactory:
    def __init__(self, database: FakeDatabase) -> None:
        self.client = FakeClient(database)

    @contextmanager
    def mongodb(self) -> Iterator[FakeClient]:
        yield self.client


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        demo_mode=False,
        mongodb_uri="mongodb://configured",
        mongodb_database="semikb",
    )


def _add_index(collection: FakeCollection, name: str) -> None:
    spec = next(item for item in MONGO_INDEX_SPECS[collection.name] if item.name == name)
    collection.indexes[name] = {"key": list(spec.keys), "unique": spec.unique}


def _pre_migration_database() -> FakeDatabase:
    database = FakeDatabase()
    document_catalog = database["document_catalog"]
    document_catalog.document_count = 7
    _add_index(document_catalog, "document_id_revision")
    audit_events = database["audit_events"]
    audit_events.document_count = 5
    _add_index(audit_events, "event_id")
    _add_index(audit_events, "actor_user_id_created_at")
    return database


def test_migration_plan_is_additive_and_accepts_non_empty_existing_catalogs() -> None:
    actions = build_migration_plan(capture_snapshot(_pre_migration_database()))

    assert {(item.collection, item.operation, item.index_name) for item in actions} == {
        ("document_catalog", "create", "lifecycle_approval_created_at"),
        ("document_catalog", "create", "source_manifest_ref"),
        ("source_manifests", "create", "source_id_manifest_version"),
        ("source_manifests", "create", "status_created_at"),
        ("source_manifests", "create", "source_hash"),
        ("audit_events", "create", "resource_created_at"),
    }


def test_migration_applies_is_idempotent_and_rolls_back_added_indexes(tmp_path: Path) -> None:
    database = _pre_migration_database()
    factory = FakeFactory(database)
    snapshot_path = tmp_path / "before.json"

    result = migrate(
        _settings(),
        apply=True,
        snapshot_path=snapshot_path,
        factory=factory,  # type: ignore[arg-type]
    )

    assert result["status"] == "migrated"
    assert build_migration_plan(capture_snapshot(database)) == []
    assert "source_manifests" in database.list_collection_names()

    rolled_back = rollback(
        _settings(),
        snapshot_path,
        factory=factory,  # type: ignore[arg-type]
    )

    assert rolled_back["migration_version"] == MIGRATION_VERSION
    assert "source_manifests" not in database.list_collection_names()
    assert set(database["document_catalog"].indexes) == {"_id_", "document_id_revision"}
    assert set(database["audit_events"].indexes) == {
        "_id_",
        "event_id",
        "actor_user_id_created_at",
    }


def test_migration_refuses_unknown_or_changed_indexes() -> None:
    unknown = _pre_migration_database()
    unknown["document_catalog"].indexes["manual"] = {
        "key": [("manual", 1)],
        "unique": False,
    }
    with pytest.raises(SourceGovernanceMigrationSafetyError, match="unapproved indexes"):
        build_migration_plan(capture_snapshot(unknown))

    changed = _pre_migration_database()
    changed["document_catalog"].indexes["document_id_revision"]["unique"] = False
    with pytest.raises(SourceGovernanceMigrationSafetyError, match="differs"):
        build_migration_plan(capture_snapshot(changed))


def test_failed_migration_restores_previous_indexes(tmp_path: Path) -> None:
    database = _pre_migration_database()
    database["audit_events"].fail_once_on_create = "resource_created_at"

    with pytest.raises(RuntimeError, match="previous MongoDB state was restored"):
        migrate(
            _settings(),
            apply=True,
            snapshot_path=tmp_path / "before.json",
            factory=FakeFactory(database),  # type: ignore[arg-type]
        )

    assert "source_manifests" not in database.list_collection_names()
    assert set(database["document_catalog"].indexes) == {"_id_", "document_id_revision"}


def test_rollback_refuses_to_drop_registered_source_manifests(tmp_path: Path) -> None:
    database = _pre_migration_database()
    factory = FakeFactory(database)
    snapshot_path = tmp_path / "before.json"
    migrate(
        _settings(),
        apply=True,
        snapshot_path=snapshot_path,
        factory=factory,  # type: ignore[arg-type]
    )
    database["source_manifests"].document_count = 1

    with pytest.raises(SourceGovernanceMigrationSafetyError, match="contains 1 business records"):
        rollback(
            _settings(),
            snapshot_path,
            factory=factory,  # type: ignore[arg-type]
        )
