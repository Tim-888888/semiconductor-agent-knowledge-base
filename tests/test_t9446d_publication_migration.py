from __future__ import annotations

from copy import deepcopy

import pytest

from semikb.storage.mongo_schema import MONGO_INDEX_SPECS
from semikb.storage.t9446d_publication_migration import (
    PublicationMigrationSafetyError,
    build_migration_plan,
    capture_snapshot,
)


class FakeCollection:
    def __init__(self) -> None:
        self.count = 0
        self.indexes = {"_id_": {"key": [("_id", 1)], "unique": False}}

    def count_documents(self, query):
        assert query == {}
        return self.count

    def index_information(self):
        return deepcopy(self.indexes)


class FakeDatabase:
    name = "semikb"

    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}

    def list_collection_names(self):
        return list(self.collections)

    def __getitem__(self, name: str):
        self.collections.setdefault(name, FakeCollection())
        return self.collections[name]


def test_publication_migration_only_plans_declared_additive_indexes() -> None:
    database = FakeDatabase()
    evaluation = database["evaluation_datasets"]
    evaluation.count = 4
    dataset_index = MONGO_INDEX_SPECS["evaluation_datasets"][0]
    evaluation.indexes[dataset_index.name] = {
        "key": list(dataset_index.keys),
        "unique": dataset_index.unique,
    }

    actions = build_migration_plan(capture_snapshot(database))

    assert {action.collection for action in actions} == {
        "corpus_publication_batches",
        "evaluation_datasets",
        "evaluation_release_freezes",
    }
    assert not any(
        action.collection == "evaluation_datasets"
        and action.index_name == "dataset_version"
        for action in actions
    )


def test_publication_migration_refuses_unknown_indexes() -> None:
    database = FakeDatabase()
    database["corpus_publication_batches"].indexes["manual"] = {
        "key": [("manual", 1)],
        "unique": False,
    }
    with pytest.raises(PublicationMigrationSafetyError, match="unapproved indexes"):
        build_migration_plan(capture_snapshot(database))
