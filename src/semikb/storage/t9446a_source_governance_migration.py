"""Additive MongoDB indexes for T9-4.4.6a source and lifecycle governance."""

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
MIGRATION_VERSION = "t9-4.4.6a-source-governance-v1"
TARGET_COLLECTIONS = ("document_catalog", "source_manifests", "audit_events")


class SourceGovernanceMigrationSafetyError(RuntimeError):
    """The live database is outside this migration's additive safety boundary."""


@dataclass(frozen=True, slots=True)
class IndexAction:
    collection: str
    operation: str
    index_name: str
    keys: tuple[tuple[str, int], ...]
    unique: bool


def _actual_definition(index_name: str, details: dict[str, Any]) -> MongoIndexSpec:
    return MongoIndexSpec(
        name=index_name,
        keys=tuple((str(field), int(direction)) for field, direction in details.get("key", [])),
        unique=bool(details.get("unique", False)),
    )


def _serialized_indexes(collection: Any) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "name": definition.name,
                "keys": [list(item) for item in definition.keys],
                "unique": definition.unique,
            }
            for name, details in collection.index_information().items()
            if (definition := _actual_definition(name, details))
        ),
        key=lambda item: item["name"],
    )


def capture_snapshot(database: Any) -> dict[str, Any]:
    if database.name != PROJECT_DATABASE:
        raise SourceGovernanceMigrationSafetyError(
            f"refusing MongoDB database {database.name!r}; expected {PROJECT_DATABASE!r}"
        )
    existing = set(database.list_collection_names())
    collections: dict[str, Any] = {}
    for collection_name in TARGET_COLLECTIONS:
        existed = collection_name in existing
        if not existed:
            collections[collection_name] = {
                "collection_existed": False,
                "document_count": 0,
                "indexes": [],
            }
            continue
        collection = database[collection_name]
        collections[collection_name] = {
            "collection_existed": True,
            "document_count": collection.count_documents({}),
            "indexes": _serialized_indexes(collection),
        }
    return {
        "schema_version": 1,
        "database": PROJECT_DATABASE,
        "migration_version": MIGRATION_VERSION,
        "captured_at": datetime.now(UTC).isoformat(),
        "collections": collections,
    }


def _snapshot_definitions(collection_snapshot: dict[str, Any]) -> dict[str, MongoIndexSpec]:
    return {
        item["name"]: MongoIndexSpec(
            name=item["name"],
            keys=tuple((str(field), int(direction)) for field, direction in item["keys"]),
            unique=bool(item.get("unique", False)),
        )
        for item in collection_snapshot.get("indexes", [])
    }


def _validate_snapshot(snapshot: dict[str, Any]) -> None:
    if (
        snapshot.get("database") != PROJECT_DATABASE
        or snapshot.get("migration_version") != MIGRATION_VERSION
        or set(snapshot.get("collections", {})) != set(TARGET_COLLECTIONS)
    ):
        raise SourceGovernanceMigrationSafetyError(
            "Snapshot does not belong to T9-4.4.6a source governance."
        )


def build_migration_plan(snapshot: dict[str, Any]) -> list[IndexAction]:
    _validate_snapshot(snapshot)
    actions: list[IndexAction] = []
    for collection_name in TARGET_COLLECTIONS:
        actual = _snapshot_definitions(snapshot["collections"][collection_name])
        actual.pop("_id_", None)
        desired = {spec.name: spec for spec in MONGO_INDEX_SPECS[collection_name]}
        unknown = sorted(set(actual).difference(desired))
        if unknown:
            raise SourceGovernanceMigrationSafetyError(
                f"{collection_name} contains unapproved indexes: {', '.join(unknown)}"
            )
        for index_name, definition in actual.items():
            if definition != desired[index_name]:
                raise SourceGovernanceMigrationSafetyError(
                    f"{collection_name}.{index_name} differs from the desired contract; "
                    "this additive migration will not replace it"
                )
        for spec in desired.values():
            if spec.name not in actual:
                actions.append(
                    IndexAction(
                        collection=collection_name,
                        operation="create",
                        index_name=spec.name,
                        keys=spec.keys,
                        unique=spec.unique,
                    )
                )
    return actions


def _write_snapshot(snapshot: dict[str, Any], path: Path | None) -> Path:
    target = path or Path("data/runtime/migrations") / (
        f"mongo-{MIGRATION_VERSION}-before-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return target.resolve()


def _preflight_rollback(database: Any, snapshot: dict[str, Any]) -> None:
    existing = set(database.list_collection_names())
    for collection_name in TARGET_COLLECTIONS:
        before = snapshot["collections"][collection_name]
        if not before["collection_existed"] and collection_name in existing:
            count = database[collection_name].count_documents({})
            if count:
                raise SourceGovernanceMigrationSafetyError(
                    f"Rollback refused because {collection_name} contains {count} business records."
                )
        if before["collection_existed"] and collection_name not in existing:
            raise SourceGovernanceMigrationSafetyError(
                f"Rollback refused because pre-existing collection {collection_name} is missing."
            )
        if before["collection_existed"]:
            current = {
                name: _actual_definition(name, details)
                for name, details in database[collection_name].index_information().items()
            }
            for name, definition in _snapshot_definitions(before).items():
                if current.get(name) != definition:
                    raise SourceGovernanceMigrationSafetyError(
                        f"Rollback refused because baseline index {collection_name}.{name} changed."
                    )


def _restore_snapshot(database: Any, snapshot: dict[str, Any]) -> None:
    _validate_snapshot(snapshot)
    _preflight_rollback(database, snapshot)
    for collection_name in TARGET_COLLECTIONS:
        before = snapshot["collections"][collection_name]
        if not before["collection_existed"]:
            if collection_name in set(database.list_collection_names()):
                database[collection_name].drop()
            continue
        baseline_names = set(_snapshot_definitions(before))
        collection = database[collection_name]
        current = {
            name: _actual_definition(name, details)
            for name, details in collection.index_information().items()
        }
        added_names = set(current).difference(baseline_names, {"_id_"})
        desired = {spec.name: spec for spec in MONGO_INDEX_SPECS[collection_name]}
        unsafe = sorted(
            name for name in added_names if name not in desired or current[name] != desired[name]
        )
        if unsafe:
            raise SourceGovernanceMigrationSafetyError(
                f"Rollback refused because {collection_name} has unrelated new indexes: "
                + ", ".join(unsafe)
            )
        for index_name in sorted(added_names):
            collection.drop_index(index_name)


def migrate(
    settings: Settings,
    *,
    apply: bool = False,
    snapshot_path: Path | None = None,
    factory: StorageClientFactory | None = None,
) -> dict[str, Any]:
    if settings.mongodb_database != PROJECT_DATABASE:
        raise SourceGovernanceMigrationSafetyError("MONGODB_DATABASE must be semikb.")
    client_factory = factory or StorageClientFactory(settings)
    with client_factory.mongodb() as client:
        database = client[PROJECT_DATABASE]
        snapshot = capture_snapshot(database)
        actions = build_migration_plan(snapshot)
        saved_snapshot = _write_snapshot(snapshot, snapshot_path)
        if not apply:
            return {
                "database": PROJECT_DATABASE,
                "migration_version": MIGRATION_VERSION,
                "applied": False,
                "status": "dry_run",
                "snapshot_path": str(saved_snapshot),
                "actions": [asdict(action) for action in actions],
            }
        try:
            for action in actions:
                database[action.collection].create_index(
                    list(action.keys),
                    name=action.index_name,
                    unique=action.unique,
                )
            if build_migration_plan(capture_snapshot(database)):
                raise RuntimeError("source governance index contract still differs after migration")
        except Exception as migration_error:
            try:
                _restore_snapshot(database, snapshot)
            except Exception as rollback_error:
                raise RuntimeError(
                    "T9-4.4.6a migration and automatic rollback both failed"
                ) from ExceptionGroup(
                    "migration and rollback errors",
                    [migration_error, rollback_error],
                )
            raise RuntimeError(
                "T9-4.4.6a migration failed; previous MongoDB state was restored"
            ) from migration_error
        return {
            "database": PROJECT_DATABASE,
            "migration_version": MIGRATION_VERSION,
            "applied": True,
            "status": "migrated" if actions else "already_current",
            "snapshot_path": str(saved_snapshot),
            "actions": [asdict(action) for action in actions],
        }


def rollback(
    settings: Settings,
    snapshot_path: Path,
    *,
    factory: StorageClientFactory | None = None,
) -> dict[str, Any]:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    _validate_snapshot(snapshot)
    if settings.mongodb_database != PROJECT_DATABASE:
        raise SourceGovernanceMigrationSafetyError("MONGODB_DATABASE must be semikb.")
    client_factory = factory or StorageClientFactory(settings)
    with client_factory.mongodb() as client:
        _restore_snapshot(client[PROJECT_DATABASE], snapshot)
    return {
        "database": PROJECT_DATABASE,
        "migration_version": MIGRATION_VERSION,
        "rolled_back": True,
        "snapshot_path": str(snapshot_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or apply the T9-4.4.6a source governance MongoDB migration"
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rollback", type=Path)
    parser.add_argument("--snapshot-path", type=Path)
    args = parser.parse_args()
    if args.rollback:
        result = rollback(get_settings(), args.rollback)
    else:
        result = migrate(
            get_settings(),
            apply=args.apply,
            snapshot_path=args.snapshot_path,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
