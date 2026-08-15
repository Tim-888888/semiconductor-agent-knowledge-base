"""Additive, reversible MongoDB migration for the T9-4.4.4 table asset catalog."""

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
COLLECTION_NAME = "table_assets"
MIGRATION_VERSION = "t9-4.4.4-table-assets-v1"


class TableAssetMigrationSafetyError(RuntimeError):
    """The live collection is outside the narrow additive migration boundary."""


@dataclass(frozen=True, slots=True)
class IndexAction:
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


def capture_snapshot(database: Any) -> dict[str, Any]:
    if database.name != PROJECT_DATABASE:
        raise TableAssetMigrationSafetyError(
            f"refusing MongoDB database {database.name!r}; expected {PROJECT_DATABASE!r}"
        )
    existed = COLLECTION_NAME in set(database.list_collection_names())
    if not existed:
        return {
            "schema_version": 1,
            "database": PROJECT_DATABASE,
            "migration_version": MIGRATION_VERSION,
            "captured_at": datetime.now(UTC).isoformat(),
            "collection_existed": False,
            "document_count": 0,
            "indexes": [],
        }
    collection = database[COLLECTION_NAME]
    indexes = [
        {
            "name": definition.name,
            "keys": [list(item) for item in definition.keys],
            "unique": definition.unique,
        }
        for name, details in collection.index_information().items()
        if (definition := _actual_definition(name, details))
    ]
    return {
        "schema_version": 1,
        "database": PROJECT_DATABASE,
        "migration_version": MIGRATION_VERSION,
        "captured_at": datetime.now(UTC).isoformat(),
        "collection_existed": True,
        "document_count": collection.count_documents({}),
        "indexes": sorted(indexes, key=lambda item: item["name"]),
    }


def _snapshot_definitions(snapshot: dict[str, Any]) -> dict[str, MongoIndexSpec]:
    return {
        item["name"]: MongoIndexSpec(
            name=item["name"],
            keys=tuple((str(field), int(direction)) for field, direction in item["keys"]),
            unique=bool(item.get("unique", False)),
        )
        for item in snapshot.get("indexes", [])
    }


def build_migration_plan(snapshot: dict[str, Any]) -> list[IndexAction]:
    if (
        snapshot.get("database") != PROJECT_DATABASE
        or snapshot.get("migration_version") != MIGRATION_VERSION
    ):
        raise TableAssetMigrationSafetyError("Snapshot does not belong to T9-4.4.4.")
    document_count = int(snapshot.get("document_count", -1))
    if document_count != 0:
        raise TableAssetMigrationSafetyError(
            f"{COLLECTION_NAME} contains {document_count} documents; migration requires 0"
        )

    actual = _snapshot_definitions(snapshot)
    actual.pop("_id_", None)
    desired = {spec.name: spec for spec in MONGO_INDEX_SPECS[COLLECTION_NAME]}
    unknown = sorted(set(actual).difference(desired))
    if unknown:
        raise TableAssetMigrationSafetyError(
            f"{COLLECTION_NAME} contains unapproved indexes: {', '.join(unknown)}"
        )

    actions: list[IndexAction] = []
    for name, definition in sorted(actual.items()):
        if definition != desired[name]:
            actions.append(IndexAction("drop", name, definition.keys, definition.unique))
    for spec in desired.values():
        if actual.get(spec.name) != spec:
            actions.append(IndexAction("create", spec.name, spec.keys, spec.unique))
    return actions


def _write_snapshot(snapshot: dict[str, Any], path: Path | None) -> Path:
    target = path or Path("data/runtime/migrations") / (
        f"mongo-{MIGRATION_VERSION}-before-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return target.resolve()


def _restore_snapshot(database: Any, snapshot: dict[str, Any]) -> None:
    collection = database[COLLECTION_NAME]
    if collection.count_documents({}) != 0:
        raise TableAssetMigrationSafetyError(
            "Rollback refused because table_assets contains business records."
        )
    if not snapshot["collection_existed"]:
        collection.drop()
        return
    for index_name in collection.index_information():
        if index_name != "_id_":
            collection.drop_index(index_name)
    for definition in _snapshot_definitions(snapshot).values():
        if definition.name != "_id_":
            collection.create_index(
                list(definition.keys),
                name=definition.name,
                unique=definition.unique,
            )


def migrate(
    settings: Settings,
    *,
    apply: bool = False,
    snapshot_path: Path | None = None,
    factory: StorageClientFactory | None = None,
) -> dict[str, Any]:
    if settings.mongodb_database != PROJECT_DATABASE:
        raise TableAssetMigrationSafetyError("MONGODB_DATABASE must be semikb.")
    client_factory = factory or StorageClientFactory(settings)
    with client_factory.mongodb() as client:
        database = client[settings.mongodb_database]
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
            collection = database[COLLECTION_NAME]
            for action in actions:
                if action.operation == "drop":
                    collection.drop_index(action.index_name)
                else:
                    collection.create_index(
                        list(action.keys),
                        name=action.index_name,
                        unique=action.unique,
                    )
            if build_migration_plan(capture_snapshot(database)):
                raise RuntimeError("table_assets index contract still differs after migration")
        except Exception as migration_error:
            try:
                _restore_snapshot(database, snapshot)
            except Exception as rollback_error:
                raise RuntimeError(
                    "T9-4.4.4 migration and automatic rollback both failed"
                ) from ExceptionGroup(
                    "migration and rollback errors",
                    [migration_error, rollback_error],
                )
            raise RuntimeError(
                "T9-4.4.4 migration failed; previous table_assets state was restored"
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
    if (
        settings.mongodb_database != PROJECT_DATABASE
        or snapshot.get("database") != PROJECT_DATABASE
        or snapshot.get("migration_version") != MIGRATION_VERSION
    ):
        raise TableAssetMigrationSafetyError("Snapshot does not belong to T9-4.4.4.")
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
    parser = argparse.ArgumentParser(description="Plan or apply the T9-4.4.4 MongoDB migration")
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
