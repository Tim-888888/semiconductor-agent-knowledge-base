from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest

from semikb.config import Settings
from semikb.storage.t9444_table_assets_migration import (
    COLLECTION_NAME,
    MIGRATION_VERSION,
    TableAssetMigrationSafetyError,
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
        self.fail_next_collection_on_create: str | None = None

    def list_collection_names(self) -> list[str]:
        return list(self.collections)

    def __getitem__(self, name: str) -> FakeCollection:
        if name not in self.collections:
            collection = FakeCollection(self, name)
            collection.fail_once_on_create = self.fail_next_collection_on_create
            self.fail_next_collection_on_create = None
            self.collections[name] = collection
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


def test_missing_table_assets_collection_has_additive_index_plan() -> None:
    snapshot = capture_snapshot(FakeDatabase())

    actions = build_migration_plan(snapshot)

    assert snapshot["collection_existed"] is False
    assert [(action.operation, action.index_name) for action in actions] == [
        ("create", "table_id"),
        ("create", "document_id_revision"),
    ]


def test_table_assets_migration_applies_is_idempotent_and_rolls_back(
    tmp_path: Path,
) -> None:
    database = FakeDatabase()
    factory = FakeFactory(database)
    snapshot_path = tmp_path / "before.json"

    result = migrate(
        _settings(),
        apply=True,
        snapshot_path=snapshot_path,
        factory=factory,
    )

    assert result["status"] == "migrated"
    assert build_migration_plan(capture_snapshot(database)) == []
    assert set(database[COLLECTION_NAME].indexes) == {
        "_id_",
        "table_id",
        "document_id_revision",
    }

    rolled_back = rollback(_settings(), snapshot_path, factory=factory)

    assert rolled_back["migration_version"] == MIGRATION_VERSION
    assert COLLECTION_NAME not in database.list_collection_names()


def test_table_assets_migration_refuses_non_empty_or_unknown_state() -> None:
    non_empty = FakeDatabase()
    non_empty[COLLECTION_NAME].document_count = 1
    with pytest.raises(TableAssetMigrationSafetyError, match="contains 1 documents"):
        build_migration_plan(capture_snapshot(non_empty))

    unknown = FakeDatabase()
    unknown[COLLECTION_NAME].indexes["manual"] = {
        "key": [("manual", 1)],
        "unique": False,
    }
    with pytest.raises(TableAssetMigrationSafetyError, match="unapproved indexes: manual"):
        build_migration_plan(capture_snapshot(unknown))


def test_failed_table_assets_migration_restores_absent_collection(tmp_path: Path) -> None:
    database = FakeDatabase()
    database.fail_next_collection_on_create = "document_id_revision"

    with pytest.raises(RuntimeError, match="previous table_assets state was restored"):
        migrate(
            _settings(),
            apply=True,
            snapshot_path=tmp_path / "before.json",
            factory=FakeFactory(database),
        )

    assert COLLECTION_NAME not in database.list_collection_names()


def test_rollback_refuses_to_drop_business_records(tmp_path: Path) -> None:
    database = FakeDatabase()
    factory = FakeFactory(database)
    snapshot_path = tmp_path / "before.json"
    migrate(
        _settings(),
        apply=True,
        snapshot_path=snapshot_path,
        factory=factory,
    )
    database[COLLECTION_NAME].document_count = 1

    with pytest.raises(TableAssetMigrationSafetyError, match="contains business records"):
        rollback(_settings(), snapshot_path, factory=factory)
