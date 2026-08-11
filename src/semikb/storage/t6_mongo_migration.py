"""Safe index migration for LangGraph-owned T6 persistence collections."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from semikb.config import Settings, get_settings
from semikb.storage.clients import StorageClientFactory
from semikb.storage.mongo_schema import MONGO_INDEX_SPECS, MongoIndexSpec

PROJECT_DATABASE = "semikb"
TARGET_COLLECTIONS = (
    "checkpoints",
    "checkpoint_writes",
    "long_term_memories",
    "audit_events",
)


class T6MigrationSafetyError(RuntimeError):
    """The migration boundary is not provably safe."""


@dataclass(frozen=True, slots=True)
class IndexAction:
    collection: str
    operation: str
    index_name: str
    keys: tuple[tuple[str, int], ...]
    unique: bool


def _spec(name: str, *fields: str, unique: bool = False) -> MongoIndexSpec:
    return MongoIndexSpec(name, tuple((field, 1) for field in fields), unique)


APPROVED_PRE_T6: dict[str, tuple[MongoIndexSpec, ...]] = {
    "checkpoints": (_spec("thread_id_checkpoint_id", "thread_id", "checkpoint_id"),),
    "checkpoint_writes": (_spec("thread_id_checkpoint_id", "thread_id", "checkpoint_id"),),
    "long_term_memories": (
        _spec("memory_id", "memory_id", unique=True),
        _spec("user_id", "user_id"),
    ),
    "audit_events": (_spec("actor_user_id_created_at", "actor_user_id", "created_at"),),
}


def _definition(name: str, details: dict[str, Any]) -> MongoIndexSpec:
    return MongoIndexSpec(
        name,
        tuple((str(field), int(direction)) for field, direction in details.get("key", [])),
        bool(details.get("unique", False)),
    )


def snapshot(database: Any) -> dict[str, Any]:
    collections: dict[str, Any] = {}
    for name in TARGET_COLLECTIONS:
        collection = database[name]
        collections[name] = {
            "document_count": collection.count_documents({}),
            "indexes": [
                {
                    "name": definition.name,
                    "keys": [list(key) for key in definition.keys],
                    "unique": definition.unique,
                }
                for index_name, details in collection.index_information().items()
                if (definition := _definition(index_name, details)).name != "_id_"
            ],
        }
    return {
        "schema_version": 1,
        "database": database.name,
        "captured_at": datetime.now(UTC).isoformat(),
        "collections": collections,
    }


def _definitions(items: list[dict[str, Any]]) -> dict[str, MongoIndexSpec]:
    return {
        item["name"]: MongoIndexSpec(
            item["name"],
            tuple((str(field), int(direction)) for field, direction in item["keys"]),
            bool(item.get("unique", False)),
        )
        for item in items
    }


def plan(state: dict[str, Any]) -> list[IndexAction]:
    if state.get("database") != PROJECT_DATABASE:
        raise T6MigrationSafetyError("T6 migration only permits the semikb database.")
    actions: list[IndexAction] = []
    for name in TARGET_COLLECTIONS:
        collection_state = state["collections"][name]
        if int(collection_state["document_count"]) != 0:
            raise T6MigrationSafetyError(f"{name} must be empty before the T6 migration.")
        actual = _definitions(collection_state["indexes"])
        desired = {item.name: item for item in MONGO_INDEX_SPECS[name]}
        approved = {*APPROVED_PRE_T6[name], *MONGO_INDEX_SPECS[name]}
        unknown = [item.name for item in actual.values() if item not in approved]
        if unknown:
            raise T6MigrationSafetyError(f"{name} has unapproved indexes: {', '.join(unknown)}")
        for index_name, definition in sorted(actual.items()):
            if desired.get(index_name) != definition:
                actions.append(IndexAction(name, "drop", index_name, definition.keys, definition.unique))
        for definition in MONGO_INDEX_SPECS[name]:
            if actual.get(definition.name) != definition:
                actions.append(
                    IndexAction(name, "create", definition.name, definition.keys, definition.unique)
                )
    return actions


def migrate(
    settings: Settings,
    *,
    apply: bool = False,
    snapshot_path: Path | None = None,
) -> dict[str, Any]:
    if settings.mongodb_database != PROJECT_DATABASE:
        raise T6MigrationSafetyError("MONGODB_DATABASE must be semikb.")
    with StorageClientFactory(settings).mongodb() as client:
        database = client[settings.mongodb_database]
        before = snapshot(database)
        actions = plan(before)
        path = snapshot_path or Path("data/runtime/migrations") / (
            f"mongo-t6-before-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(before, ensure_ascii=False, indent=2), encoding="utf-8")
        if not apply:
            return {
                "database": PROJECT_DATABASE,
                "applied": False,
                "snapshot_path": str(path.resolve()),
                "actions": [asdict(action) for action in actions],
            }

        touched: set[str] = set()
        try:
            for action in actions:
                collection = database[action.collection]
                touched.add(action.collection)
                if action.operation == "drop":
                    collection.drop_index(action.index_name)
                else:
                    collection.create_index(
                        list(action.keys),
                        name=action.index_name,
                        unique=action.unique,
                    )
            if plan(snapshot(database)):
                raise RuntimeError("T6 index contract still differs after migration.")
        except Exception as migration_error:
            for name in touched:
                collection = database[name]
                for index_name in collection.index_information():
                    if index_name != "_id_":
                        collection.drop_index(index_name)
                for definition in _definitions(before["collections"][name]["indexes"]).values():
                    collection.create_index(
                        list(definition.keys),
                        name=definition.name,
                        unique=definition.unique,
                    )
            raise RuntimeError("T6 migration failed; previous indexes restored.") from migration_error
        return {
            "database": PROJECT_DATABASE,
            "applied": True,
            "snapshot_path": str(path.resolve()),
            "actions": [asdict(action) for action in actions],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or apply T6 LangGraph MongoDB indexes")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--snapshot-path", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            migrate(get_settings(), apply=args.apply, snapshot_path=args.snapshot_path),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
