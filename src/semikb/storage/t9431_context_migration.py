"""Reversible T9-4.3.1 backfill for conversation ordering and context fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from bson import json_util

from semikb.config import Settings
from semikb.contracts.models import ActiveConversationContext
from semikb.storage.clients import StorageClientFactory

PROJECT_DATABASE = "semikb"
MIGRATION_VERSION = "t9-4.3.1-context-v1"
THREAD_FIELDS = (
    "summary_upto_message_id",
    "context_version",
    "active_context",
    "next_turn_seq",
    "last_turn_seq",
)


class ContextMigrationSafetyError(RuntimeError):
    """The requested migration or rollback is outside its narrow safety boundary."""


@dataclass(frozen=True, slots=True)
class ContextMigrationPlan:
    database: str
    migration_version: str
    scanned_threads: int
    changed_threads: int
    assigned_message_sequences: int


def _thread_snapshot(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "thread_id": document["thread_id"],
        "fields": {
            field: {
                "exists": field in document,
                "value": document.get(field),
            }
            for field in THREAD_FIELDS
        },
        "message_turn_seq": [
            {
                "message_id": item.get("message_id"),
                "exists": "turn_seq" in item,
                "value": item.get("turn_seq"),
            }
            for item in document.get("messages", [])
        ],
    }


def _backfilled_document(document: dict[str, Any]) -> tuple[dict[str, Any], int]:
    updated = dict(document)
    messages = [dict(item) for item in document.get("messages", [])]
    sequenced = [item for item in messages if isinstance(item.get("turn_seq"), int)]
    if sequenced:
        existing_values = [int(item["turn_seq"]) for item in sequenced]
        if len(existing_values) != len(set(existing_values)) or any(
            value < 1 for value in existing_values
        ):
            raise ContextMigrationSafetyError(
                f"thread has invalid existing message ordering: {document.get('thread_id')}"
            )
        for index, item in enumerate(messages, start=1):
            current = item.get("turn_seq")
            if current is not None and current != index:
                raise ContextMigrationSafetyError(
                    f"thread sequence conflicts with message order: {document.get('thread_id')}"
                )
            item["turn_seq"] = index
        values = list(range(1, len(messages) + 1))
        assigned = len(messages) - len(sequenced)
    else:
        for index, item in enumerate(messages, start=1):
            item["turn_seq"] = index
        values = list(range(1, len(messages) + 1))
        assigned = len(messages)
    used = set(values)
    last_seq = max(used, default=0)
    updated["messages"] = messages
    updated.setdefault("summary_upto_message_id", None)
    updated.setdefault("context_version", 1)
    updated.setdefault("active_context", ActiveConversationContext().model_dump(mode="python"))
    updated["last_turn_seq"] = max(int(updated.get("last_turn_seq") or 0), last_seq)
    updated["next_turn_seq"] = max(
        int(updated.get("next_turn_seq") or 1),
        updated["last_turn_seq"] + 1,
    )
    return updated, assigned


def plan(database: Any) -> tuple[ContextMigrationPlan, list[dict[str, Any]]]:
    if database.name != PROJECT_DATABASE:
        raise ContextMigrationSafetyError("T9-4.3.1 migration only permits the semikb database.")
    live_cutoff = datetime.now(UTC) - timedelta(minutes=30)
    active_request = database["agent_message_requests"].find_one(
        {
            "status": {"$in": ["accepted", "running"]},
            "updated_at": {"$gte": live_cutoff},
        },
        {"_id": 0, "request_id": 1},
    )
    leased_thread = database["agent_threads"].find_one(
        {"active_request_id": {"$nin": [None, ""]}},
        {"_id": 0, "thread_id": 1},
    )
    if active_request or leased_thread:
        raise ContextMigrationSafetyError(
            "Conversation requests are active; finish or cancel them before migration."
        )
    snapshots: list[dict[str, Any]] = []
    assigned = 0
    scanned = 0
    for document in database["agent_threads"].find({}, {"_id": 0}):
        scanned += 1
        updated, item_assigned = _backfilled_document(document)
        if updated != document:
            snapshots.append(_thread_snapshot(document))
            assigned += item_assigned
    return (
        ContextMigrationPlan(
            database=database.name,
            migration_version=MIGRATION_VERSION,
            scanned_threads=scanned,
            changed_threads=len(snapshots),
            assigned_message_sequences=assigned,
        ),
        snapshots,
    )


def migrate(
    settings: Settings,
    *,
    apply: bool = False,
    snapshot_path: Path | None = None,
) -> dict[str, Any]:
    if settings.mongodb_database != PROJECT_DATABASE:
        raise ContextMigrationSafetyError("MONGODB_DATABASE must be semikb.")
    with StorageClientFactory(settings).mongodb() as client:
        database = client[settings.mongodb_database]
        migration_plan, snapshots = plan(database)
        path = snapshot_path or Path("data/runtime/migrations") / (
            f"mongo-{MIGRATION_VERSION}-before-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        payload = {
            "schema_version": 1,
            "database": database.name,
            "migration_version": MIGRATION_VERSION,
            "captured_at": datetime.now(UTC).isoformat(),
            "threads": snapshots,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_util.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if apply:
            collection = database["agent_threads"]
            try:
                for item in snapshots:
                    current = collection.find_one({"thread_id": item["thread_id"]})
                    if current is None:
                        raise ContextMigrationSafetyError(
                            f"thread disappeared during migration: {item['thread_id']}"
                        )
                    updated, _ = _backfilled_document(current)
                    result = collection.replace_one(
                        {"_id": current["_id"]},
                        updated,
                        upsert=False,
                    )
                    if result.matched_count != 1:
                        raise RuntimeError(
                            f"thread migration was not acknowledged: {item['thread_id']}"
                        )
                remaining, _ = plan(database)
                if remaining.changed_threads:
                    raise RuntimeError("T9-4.3.1 migration is not idempotent after apply.")
            except Exception as migration_error:
                _restore_collection(collection, snapshots)
                raise RuntimeError(
                    "T9-4.3.1 migration failed; migrated fields were restored."
                ) from migration_error
        return {
            **asdict(migration_plan),
            "applied": apply,
            "snapshot_path": str(path.resolve()),
        }


def rollback(settings: Settings, snapshot_path: Path) -> dict[str, Any]:
    payload = json_util.loads(snapshot_path.read_text(encoding="utf-8"))
    if (
        settings.mongodb_database != PROJECT_DATABASE
        or payload.get("database") != PROJECT_DATABASE
        or payload.get("migration_version") != MIGRATION_VERSION
    ):
        raise ContextMigrationSafetyError("Snapshot does not belong to this semikb migration.")
    with StorageClientFactory(settings).mongodb() as client:
        collection = client[settings.mongodb_database]["agent_threads"]
        restored = _restore_collection(collection, payload.get("threads", []))
        return {
            "database": PROJECT_DATABASE,
            "migration_version": MIGRATION_VERSION,
            "rolled_back": True,
            "restored_threads": restored,
            "snapshot_path": str(snapshot_path.resolve()),
        }


def _restore_collection(collection: Any, snapshots: list[dict[str, Any]]) -> int:
    restored = 0
    for item in snapshots:
        document = collection.find_one({"thread_id": item["thread_id"]})
        if document is None:
            raise ContextMigrationSafetyError(
                f"rollback target no longer exists: {item['thread_id']}"
            )
        for field, state in item["fields"].items():
            if state["exists"]:
                document[field] = state["value"]
            else:
                document.pop(field, None)
        original_sequences = {
            entry["message_id"]: entry for entry in item["message_turn_seq"]
        }
        if len(document.get("messages", [])) != len(original_sequences):
            raise ContextMigrationSafetyError(
                f"message set changed after migration: {item['thread_id']}"
            )
        for message in document.get("messages", []):
            state = original_sequences.get(message.get("message_id"))
            if state is None:
                raise ContextMigrationSafetyError(
                    f"message set changed after migration: {item['thread_id']}"
                )
            if state["exists"]:
                message["turn_seq"] = state["value"]
            else:
                message.pop("turn_seq", None)
        result = collection.replace_one({"_id": document["_id"]}, document, upsert=False)
        if result.matched_count != 1:
            raise RuntimeError(f"thread rollback was not acknowledged: {item['thread_id']}")
        restored += 1
    return restored
