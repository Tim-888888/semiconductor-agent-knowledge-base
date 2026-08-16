from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest

from semikb.config import Settings
from semikb.storage.mongo_schema import MONGO_INDEX_SPECS
from semikb.storage.t9446b_lifecycle_migration import (
    LifecycleMigrationSafetyError,
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

    def create_index(self, keys, *, name: str, unique: bool) -> str:
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


def test_lifecycle_migration_is_additive_idempotent_and_reversible(tmp_path: Path) -> None:
    database = FakeDatabase()
    factory = FakeFactory(database)
    before = capture_snapshot(database)
    actions = build_migration_plan(before)
    assert {item.index_name for item in actions} == {
        spec.name for spec in MONGO_INDEX_SPECS["document_lifecycle_operations"]
    }

    snapshot = tmp_path / "before.json"
    result = migrate(
        _settings(),
        apply=True,
        snapshot_path=snapshot,
        factory=factory,  # type: ignore[arg-type]
    )
    assert result["status"] == "migrated"
    assert build_migration_plan(capture_snapshot(database)) == []

    rolled_back = rollback(
        _settings(),
        snapshot,
        factory=factory,  # type: ignore[arg-type]
    )
    assert rolled_back["rolled_back"] is True
    assert "document_lifecycle_operations" not in database.list_collection_names()


def test_lifecycle_migration_refuses_unknown_index() -> None:
    database = FakeDatabase()
    collection = database["document_lifecycle_operations"]
    collection.indexes["manual"] = {"key": [("manual", 1)], "unique": False}
    with pytest.raises(LifecycleMigrationSafetyError, match="unapproved indexes"):
        build_migration_plan(capture_snapshot(database))


def test_lifecycle_migration_rolls_back_partial_failure(tmp_path: Path) -> None:
    database = FakeDatabase()
    collection = database["document_lifecycle_operations"]
    collection.fail_once_on_create = "document_revision_created_at"
    with pytest.raises(RuntimeError, match="previous MongoDB state was restored"):
        migrate(
            _settings(),
            apply=True,
            snapshot_path=tmp_path / "before.json",
            factory=FakeFactory(database),  # type: ignore[arg-type]
        )
    assert set(collection.indexes) == {"_id_"}


def test_rollback_never_drops_business_operations(tmp_path: Path) -> None:
    database = FakeDatabase()
    factory = FakeFactory(database)
    snapshot = tmp_path / "before.json"
    migrate(
        _settings(),
        apply=True,
        snapshot_path=snapshot,
        factory=factory,  # type: ignore[arg-type]
    )
    database["document_lifecycle_operations"].document_count = 1
    with pytest.raises(LifecycleMigrationSafetyError, match="contains 1 business records"):
        rollback(
            _settings(),
            snapshot,
            factory=factory,  # type: ignore[arg-type]
        )
