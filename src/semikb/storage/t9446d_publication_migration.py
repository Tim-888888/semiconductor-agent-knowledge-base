"""Additive MongoDB migration for T9-4.4.6d publication and evaluation governance."""

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
MIGRATION_VERSION = "t9-4.4.6d-publication-evaluation-v1"
TARGET_COLLECTIONS = (
    "corpus_publication_batches",
    "evaluation_datasets",
    "evaluation_release_freezes",
)


class PublicationMigrationSafetyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class IndexAction:
    collection: str
    operation: str
    index_name: str
    keys: tuple[tuple[str, int], ...]
    unique: bool


def _definition(name: str, details: dict[str, Any]) -> MongoIndexSpec:
    return MongoIndexSpec(
        name=name,
        keys=tuple((str(field), int(direction)) for field, direction in details.get("key", [])),
        unique=bool(details.get("unique", False)),
    )


def _indexes(collection: Any) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "name": definition.name,
                "keys": [list(item) for item in definition.keys],
                "unique": definition.unique,
            }
            for name, details in collection.index_information().items()
            if (definition := _definition(name, details))
        ],
        key=lambda item: item["name"],
    )


def capture_snapshot(database: Any) -> dict[str, Any]:
    if database.name != PROJECT_DATABASE:
        raise PublicationMigrationSafetyError(
            f"refusing MongoDB database {database.name!r}; expected {PROJECT_DATABASE!r}"
        )
    existing = set(database.list_collection_names())
    collections = {}
    for name in TARGET_COLLECTIONS:
        existed = name in existing
        collection = database[name] if existed else None
        collections[name] = {
            "collection_existed": existed,
            "document_count": collection.count_documents({}) if collection is not None else 0,
            "indexes": _indexes(collection) if collection is not None else [],
        }
    return {
        "schema_version": 1,
        "database": PROJECT_DATABASE,
        "migration_version": MIGRATION_VERSION,
        "captured_at": datetime.now(UTC).isoformat(),
        "collections": collections,
    }


def _validate(snapshot: dict[str, Any]) -> None:
    if (
        snapshot.get("database") != PROJECT_DATABASE
        or snapshot.get("migration_version") != MIGRATION_VERSION
        or set(snapshot.get("collections", {})) != set(TARGET_COLLECTIONS)
    ):
        raise PublicationMigrationSafetyError("Snapshot does not belong to T9-4.4.6d.")


def _snapshot_definitions(state: dict[str, Any]) -> dict[str, MongoIndexSpec]:
    return {
        item["name"]: MongoIndexSpec(
            name=item["name"],
            keys=tuple((str(field), int(direction)) for field, direction in item["keys"]),
            unique=bool(item.get("unique", False)),
        )
        for item in state.get("indexes", [])
    }


def build_migration_plan(snapshot: dict[str, Any]) -> list[IndexAction]:
    _validate(snapshot)
    actions: list[IndexAction] = []
    for collection_name in TARGET_COLLECTIONS:
        actual = _snapshot_definitions(snapshot["collections"][collection_name])
        actual.pop("_id_", None)
        desired = {
            spec.name: spec for spec in MONGO_INDEX_SPECS[collection_name]
        }
        unknown = sorted(set(actual).difference(desired))
        if unknown:
            raise PublicationMigrationSafetyError(
                f"{collection_name} contains unapproved indexes: {', '.join(unknown)}"
            )
        for name, definition in actual.items():
            if definition != desired[name]:
                raise PublicationMigrationSafetyError(
                    f"{collection_name}.{name} differs from the additive contract."
                )
        actions.extend(
            IndexAction(
                collection=collection_name,
                operation="create",
                index_name=spec.name,
                keys=spec.keys,
                unique=spec.unique,
            )
            for spec in desired.values()
            if spec.name not in actual
        )
    return actions


def _write_snapshot(snapshot: dict[str, Any], path: Path | None) -> Path:
    target = path or Path("data/runtime/migrations") / (
        f"mongo-{MIGRATION_VERSION}-before-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return target.resolve()


def _restore(database: Any, snapshot: dict[str, Any]) -> None:
    _validate(snapshot)
    existing = set(database.list_collection_names())
    for name in TARGET_COLLECTIONS:
        before = snapshot["collections"][name]
        if not before["collection_existed"]:
            if name not in existing:
                continue
            count = database[name].count_documents({})
            if count:
                raise PublicationMigrationSafetyError(
                    f"Rollback refused because {name} contains {count} business records."
                )
            database[name].drop()
            continue
        if name not in existing:
            raise PublicationMigrationSafetyError(
                f"Rollback refused because baseline collection {name} is missing."
            )
        baseline = _snapshot_definitions(before)
        current = {
            index_name: _definition(index_name, details)
            for index_name, details in database[name].index_information().items()
        }
        for index_name, definition in baseline.items():
            if current.get(index_name) != definition:
                raise PublicationMigrationSafetyError(
                    f"Rollback refused because baseline index {name}.{index_name} changed."
                )
        desired = {spec.name: spec for spec in MONGO_INDEX_SPECS[name]}
        added = set(current).difference(baseline, {"_id_"})
        unsafe = sorted(
            index_name
            for index_name in added
            if index_name not in desired or current[index_name] != desired[index_name]
        )
        if unsafe:
            raise PublicationMigrationSafetyError(
                f"Rollback refused because {name} has unrelated indexes: {', '.join(unsafe)}"
            )
        for index_name in sorted(added):
            database[name].drop_index(index_name)


def migrate(
    settings: Settings,
    *,
    apply: bool = False,
    snapshot_path: Path | None = None,
    factory: StorageClientFactory | None = None,
) -> dict[str, Any]:
    if settings.mongodb_database != PROJECT_DATABASE:
        raise PublicationMigrationSafetyError("MONGODB_DATABASE must be semikb.")
    client_factory = factory or StorageClientFactory(settings)
    with client_factory.mongodb() as client:
        database = client[PROJECT_DATABASE]
        snapshot = capture_snapshot(database)
        actions = build_migration_plan(snapshot)
        saved = _write_snapshot(snapshot, snapshot_path)
        if not apply:
            return {
                "database": PROJECT_DATABASE,
                "migration_version": MIGRATION_VERSION,
                "applied": False,
                "status": "dry_run",
                "snapshot_path": str(saved),
                "actions": [asdict(item) for item in actions],
            }
        try:
            for action in actions:
                database[action.collection].create_index(
                    list(action.keys),
                    name=action.index_name,
                    unique=action.unique,
                )
            if build_migration_plan(capture_snapshot(database)):
                raise RuntimeError("T9-4.4.6d indexes still differ after migration.")
        except Exception as migration_error:
            try:
                _restore(database, snapshot)
            except Exception as rollback_error:
                raise RuntimeError("Migration and rollback both failed.") from ExceptionGroup(
                    "migration and rollback errors", [migration_error, rollback_error]
                )
            raise RuntimeError("Migration failed; baseline indexes were restored.") from migration_error
        return {
            "database": PROJECT_DATABASE,
            "migration_version": MIGRATION_VERSION,
            "applied": True,
            "status": "migrated" if actions else "already_current",
            "snapshot_path": str(saved),
            "actions": [asdict(item) for item in actions],
        }


def rollback(
    settings: Settings,
    snapshot_path: Path,
    *,
    factory: StorageClientFactory | None = None,
) -> dict[str, Any]:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    _validate(snapshot)
    client_factory = factory or StorageClientFactory(settings)
    with client_factory.mongodb() as client:
        _restore(client[PROJECT_DATABASE], snapshot)
    return {
        "database": PROJECT_DATABASE,
        "migration_version": MIGRATION_VERSION,
        "rolled_back": True,
        "snapshot_path": str(snapshot_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="T9-4.4.6d MongoDB migration")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rollback", type=Path)
    parser.add_argument("--snapshot-path", type=Path)
    args = parser.parse_args()
    result = (
        rollback(get_settings(), args.rollback)
        if args.rollback
        else migrate(get_settings(), apply=args.apply, snapshot_path=args.snapshot_path)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
