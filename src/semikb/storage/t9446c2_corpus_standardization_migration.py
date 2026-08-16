"""Additive MongoDB migration for T9-4.4.6c-2 corpus standardization jobs."""

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
MIGRATION_VERSION = "t9-4.4.6c2-corpus-standardization-v1"
TARGET_COLLECTION = "corpus_standardization_jobs"


class CorpusMigrationSafetyError(RuntimeError):
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
        raise CorpusMigrationSafetyError(
            f"refusing MongoDB database {database.name!r}; expected {PROJECT_DATABASE!r}"
        )
    existed = TARGET_COLLECTION in set(database.list_collection_names())
    collection = database[TARGET_COLLECTION] if existed else None
    return {
        "schema_version": 1,
        "database": PROJECT_DATABASE,
        "migration_version": MIGRATION_VERSION,
        "captured_at": datetime.now(UTC).isoformat(),
        "collection": {
            "name": TARGET_COLLECTION,
            "collection_existed": existed,
            "document_count": collection.count_documents({}) if collection is not None else 0,
            "indexes": _indexes(collection) if collection is not None else [],
        },
    }


def _validate(snapshot: dict[str, Any]) -> None:
    if (
        snapshot.get("database") != PROJECT_DATABASE
        or snapshot.get("migration_version") != MIGRATION_VERSION
        or snapshot.get("collection", {}).get("name") != TARGET_COLLECTION
    ):
        raise CorpusMigrationSafetyError("Snapshot does not belong to T9-4.4.6c-2.")


def _snapshot_definitions(snapshot: dict[str, Any]) -> dict[str, MongoIndexSpec]:
    return {
        item["name"]: MongoIndexSpec(
            name=item["name"],
            keys=tuple((str(field), int(direction)) for field, direction in item["keys"]),
            unique=bool(item.get("unique", False)),
        )
        for item in snapshot["collection"].get("indexes", [])
    }


def build_migration_plan(snapshot: dict[str, Any]) -> list[IndexAction]:
    _validate(snapshot)
    actual = _snapshot_definitions(snapshot)
    actual.pop("_id_", None)
    desired = {spec.name: spec for spec in MONGO_INDEX_SPECS[TARGET_COLLECTION]}
    unknown = sorted(set(actual).difference(desired))
    if unknown:
        raise CorpusMigrationSafetyError(
            f"{TARGET_COLLECTION} contains unapproved indexes: {', '.join(unknown)}"
        )
    for name, definition in actual.items():
        if definition != desired[name]:
            raise CorpusMigrationSafetyError(
                f"{TARGET_COLLECTION}.{name} differs from the additive contract."
            )
    return [
        IndexAction(
            collection=TARGET_COLLECTION,
            operation="create",
            index_name=spec.name,
            keys=spec.keys,
            unique=spec.unique,
        )
        for spec in desired.values()
        if spec.name not in actual
    ]


def _write_snapshot(snapshot: dict[str, Any], path: Path | None) -> Path:
    target = path or Path("data/runtime/migrations") / (
        f"mongo-{MIGRATION_VERSION}-before-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return target.resolve()


def _restore(database: Any, snapshot: dict[str, Any]) -> None:
    _validate(snapshot)
    before = snapshot["collection"]
    existing = set(database.list_collection_names())
    if not before["collection_existed"]:
        if TARGET_COLLECTION not in existing:
            return
        count = database[TARGET_COLLECTION].count_documents({})
        if count:
            raise CorpusMigrationSafetyError(
                f"Rollback refused because {TARGET_COLLECTION} contains {count} business records."
            )
        database[TARGET_COLLECTION].drop()
        return
    if TARGET_COLLECTION not in existing:
        raise CorpusMigrationSafetyError("Rollback refused because the baseline collection is missing.")
    baseline = _snapshot_definitions(snapshot)
    collection = database[TARGET_COLLECTION]
    current = {
        name: _definition(name, details)
        for name, details in collection.index_information().items()
    }
    for name, definition in baseline.items():
        if current.get(name) != definition:
            raise CorpusMigrationSafetyError(
                f"Rollback refused because baseline index {name} changed."
            )
    desired = {spec.name: spec for spec in MONGO_INDEX_SPECS[TARGET_COLLECTION]}
    added = set(current).difference(baseline, {"_id_"})
    unsafe = sorted(
        name for name in added if name not in desired or current[name] != desired[name]
    )
    if unsafe:
        raise CorpusMigrationSafetyError(
            "Rollback refused because unrelated indexes exist: " + ", ".join(unsafe)
        )
    for name in sorted(added):
        collection.drop_index(name)


def migrate(
    settings: Settings,
    *,
    apply: bool = False,
    snapshot_path: Path | None = None,
    factory: StorageClientFactory | None = None,
) -> dict[str, Any]:
    if settings.mongodb_database != PROJECT_DATABASE:
        raise CorpusMigrationSafetyError("MONGODB_DATABASE must be semikb.")
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
                raise RuntimeError("Corpus standardization indexes still differ after migration.")
        except Exception as migration_error:
            try:
                _restore(database, snapshot)
            except Exception as rollback_error:
                raise RuntimeError("Migration and rollback both failed.") from ExceptionGroup(
                    "migration and rollback errors", [migration_error, rollback_error]
                )
            raise RuntimeError("Migration failed; previous MongoDB state was restored.") from migration_error
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
    parser = argparse.ArgumentParser(description="T9-4.4.6c-2 MongoDB migration")
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
