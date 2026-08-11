"""Approved T2-G4 migration for project-owned MongoDB indexes.

The command is a dry run unless ``--apply`` is supplied. It refuses databases
outside ``semikb``, non-empty collections, and indexes that are not part of the
known pre-migration or desired contracts.
"""

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
SNAPSHOT_SCHEMA_VERSION = 1


class MigrationSafetyError(RuntimeError):
    """Raised before writes when the live state is outside the approved boundary."""


def _legacy(name: str, *fields: str, unique: bool = False) -> MongoIndexSpec:
    return MongoIndexSpec(
        name=name,
        keys=tuple((field, 1) for field in fields),
        unique=unique,
    )


# Exact indexes created by the initial T2 provisioning run. Nothing else may be
# dropped automatically, even when it exists in the semikb database.
APPROVED_LEGACY_INDEX_SPECS: dict[str, tuple[MongoIndexSpec, ...]] = {
    "document_catalog": (
        _legacy("document_id_revision", "document_id", "revision"),
    ),
    "ingestion_jobs": (
        _legacy(
            "job_id_idempotency_key",
            "job_id",
            "idempotency_key",
            unique=True,
        ),
    ),
    "retrieval_traces": (
        _legacy(
            "trace_id_actor_user_id_created_at",
            "trace_id",
            "actor_user_id",
            "created_at",
        ),
    ),
    "evaluation_datasets": (
        _legacy("dataset_version", "dataset_version"),
    ),
    "evaluation_runs": (
        _legacy(
            "evaluation_run_id_created_at",
            "evaluation_run_id",
            "created_at",
        ),
    ),
    "agent_threads": (
        _legacy(
            "thread_id_actor_scope.user_id_updated_at",
            "thread_id",
            "actor_scope.user_id",
            "updated_at",
            unique=True,
        ),
    ),
    "long_term_memories": (
        _legacy("memory_id_user_id", "memory_id", "user_id"),
    ),
    "audit_events": (
        _legacy("created_at_actor_user_id", "created_at", "actor_user_id"),
    ),
}


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


def capture_snapshot(database: Any) -> dict[str, Any]:
    """Capture counts and index metadata without reading document contents."""

    existing_collections = set(database.list_collection_names())
    missing_collections = sorted(set(MONGO_INDEX_SPECS).difference(existing_collections))
    if missing_collections:
        raise MigrationSafetyError(
            f"database is missing collections: {', '.join(missing_collections)}"
        )

    collections: dict[str, Any] = {}
    for collection_name, _ in MONGO_INDEX_SPECS.items():
        collection = database[collection_name]
        indexes = []
        for index_name, details in collection.index_information().items():
            definition = _actual_definition(index_name, details)
            indexes.append(
                {
                    "name": definition.name,
                    "keys": [list(item) for item in definition.keys],
                    "unique": definition.unique,
                }
            )
        collections[collection_name] = {
            "document_count": collection.count_documents({}),
            "indexes": sorted(indexes, key=lambda item: item["name"]),
        }
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "database": database.name,
        "captured_at": datetime.now(UTC).isoformat(),
        "collections": collections,
    }


def _snapshot_definitions(collection_state: dict[str, Any]) -> dict[str, MongoIndexSpec]:
    return {
        item["name"]: MongoIndexSpec(
            name=item["name"],
            keys=tuple((str(field), int(direction)) for field, direction in item["keys"]),
            unique=bool(item.get("unique", False)),
        )
        for item in collection_state["indexes"]
    }


def build_migration_plan(snapshot: dict[str, Any]) -> list[IndexAction]:
    """Build a deterministic plan and reject state outside the approved contract."""

    if snapshot.get("database") != PROJECT_DATABASE:
        raise MigrationSafetyError(
            f"refusing MongoDB database {snapshot.get('database')!r}; expected {PROJECT_DATABASE!r}"
        )

    collection_states = snapshot.get("collections", {})
    missing = sorted(set(MONGO_INDEX_SPECS).difference(collection_states))
    if missing:
        raise MigrationSafetyError(f"snapshot is missing collections: {', '.join(missing)}")

    actions: list[IndexAction] = []
    for collection_name, desired_specs in MONGO_INDEX_SPECS.items():
        state = collection_states[collection_name]
        document_count = int(state.get("document_count", -1))
        if document_count != 0:
            raise MigrationSafetyError(
                f"{collection_name} contains {document_count} documents; automatic migration requires 0"
            )

        actual = _snapshot_definitions(state)
        actual.pop("_id_", None)
        desired = {spec.name: spec for spec in desired_specs}
        legacy_specs = APPROVED_LEGACY_INDEX_SPECS.get(collection_name, ())
        approved_definitions = {*desired_specs, *legacy_specs}

        unknown = sorted(
            definition.name
            for definition in actual.values()
            if definition not in approved_definitions
        )
        if unknown:
            raise MigrationSafetyError(
                f"{collection_name} contains unapproved indexes: {', '.join(unknown)}"
            )

        for index_name, definition in sorted(actual.items()):
            if desired.get(index_name) != definition:
                actions.append(
                    IndexAction(
                        collection=collection_name,
                        operation="drop",
                        index_name=index_name,
                        keys=definition.keys,
                        unique=definition.unique,
                    )
                )
        for spec in desired_specs:
            if actual.get(spec.name) != spec:
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


def _write_snapshot(snapshot: dict[str, Any], snapshot_path: Path | None) -> Path:
    path = snapshot_path or Path("data/runtime/migrations") / (
        f"mongo-indexes-before-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return path.resolve()


def _restore_snapshot(database: Any, snapshot: dict[str, Any], collections: set[str]) -> None:
    for collection_name in sorted(collections):
        collection = database[collection_name]
        for index_name in collection.index_information():
            if index_name != "_id_":
                collection.drop_index(index_name)
        definitions = _snapshot_definitions(snapshot["collections"][collection_name])
        for definition in definitions.values():
            if definition.name == "_id_":
                continue
            collection.create_index(
                list(definition.keys),
                name=definition.name,
                unique=definition.unique,
            )


def migrate_mongo_indexes(
    settings: Settings,
    *,
    apply: bool = False,
    snapshot_path: Path | None = None,
    factory: StorageClientFactory | None = None,
) -> dict[str, Any]:
    """Plan or apply the approved T2-G4 index-only migration."""

    if settings.mongodb_database != PROJECT_DATABASE:
        raise MigrationSafetyError(
            f"MONGODB_DATABASE must be {PROJECT_DATABASE!r} for T2-G4"
        )

    client_factory = factory or StorageClientFactory(settings)
    with client_factory.mongodb() as client:
        database = client[settings.mongodb_database]
        snapshot = capture_snapshot(database)
        actions = build_migration_plan(snapshot)
        saved_snapshot = _write_snapshot(snapshot, snapshot_path)

        if not apply:
            return {
                "database": PROJECT_DATABASE,
                "applied": False,
                "status": "dry_run",
                "snapshot_path": str(saved_snapshot),
                "actions": [asdict(action) for action in actions],
            }

        touched_collections: set[str] = set()
        try:
            for action in actions:
                collection = database[action.collection]
                touched_collections.add(action.collection)
                if action.operation == "drop":
                    collection.drop_index(action.index_name)
                else:
                    collection.create_index(
                        list(action.keys),
                        name=action.index_name,
                        unique=action.unique,
                    )
            remaining = build_migration_plan(capture_snapshot(database))
            if remaining:
                raise RuntimeError("post-migration MongoDB index contract still differs")
        except Exception as migration_error:
            try:
                _restore_snapshot(database, snapshot, touched_collections)
            except Exception as rollback_error:
                raise RuntimeError(
                    "MongoDB index migration and automatic rollback both failed"
                ) from ExceptionGroup(
                    "migration and rollback errors",
                    [migration_error, rollback_error],
                )
            raise RuntimeError("MongoDB index migration failed; previous indexes restored") from migration_error

        return {
            "database": PROJECT_DATABASE,
            "applied": True,
            "status": "migrated" if actions else "already_current",
            "snapshot_path": str(saved_snapshot),
            "actions": [asdict(action) for action in actions],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or apply the approved T2-G4 migration")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="apply index changes; without this flag the command is read-only",
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        help="optional path for the pre-migration JSON snapshot",
    )
    args = parser.parse_args()
    result = migrate_mongo_indexes(
        get_settings(),
        apply=args.apply,
        snapshot_path=args.snapshot_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
