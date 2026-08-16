from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest

from semikb.config import Settings
from semikb.storage.mongo_schema import MONGO_INDEX_SPECS
from semikb.storage.t9446c2_corpus_standardization_migration import (
    CorpusMigrationSafetyError,
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
        self.indexes = {"_id_": {"key": [("_id", 1)], "unique": False}}

    def count_documents(self, query):
        assert query == {}
        return self.document_count

    def index_information(self):
        return deepcopy(self.indexes)

    def create_index(self, keys, *, name: str, unique: bool):
        self.indexes[name] = {"key": list(keys), "unique": unique}
        return name

    def drop_index(self, name: str):
        del self.indexes[name]

    def drop(self):
        self.database.collections.pop(self.name, None)


class FakeDatabase:
    name = "semikb"

    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}

    def list_collection_names(self):
        return list(self.collections)

    def __getitem__(self, name: str):
        self.collections.setdefault(name, FakeCollection(self, name))
        return self.collections[name]


class FakeClient:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database

    def __getitem__(self, name: str):
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


def test_corpus_migration_is_additive_idempotent_and_reversible(tmp_path: Path) -> None:
    database = FakeDatabase()
    factory = FakeFactory(database)
    actions = build_migration_plan(capture_snapshot(database))
    assert {item.index_name for item in actions} == {
        item.name for item in MONGO_INDEX_SPECS["corpus_standardization_jobs"]
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
    assert rollback(
        _settings(),
        snapshot,
        factory=factory,  # type: ignore[arg-type]
    )["rolled_back"] is True
    assert "corpus_standardization_jobs" not in database.list_collection_names()


def test_corpus_migration_refuses_unknown_index() -> None:
    database = FakeDatabase()
    database["corpus_standardization_jobs"].indexes["manual"] = {
        "key": [("manual", 1)],
        "unique": False,
    }
    with pytest.raises(CorpusMigrationSafetyError, match="unapproved indexes"):
        build_migration_plan(capture_snapshot(database))
