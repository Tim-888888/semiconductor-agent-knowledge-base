"""Business persistence for Agent threads and append-only audit events."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from pymongo import MongoClient

from semikb.config import Settings
from semikb.contracts.models import AuditEvent, ThreadRecord


class ConversationRepository(Protocol):
    def create_thread(self, thread: ThreadRecord) -> ThreadRecord: ...

    def get_thread(self, thread_id: str) -> ThreadRecord | None: ...

    def list_threads(self, user_id: str) -> list[ThreadRecord]: ...

    def save_thread(self, thread: ThreadRecord) -> ThreadRecord: ...

    def append_audit(self, event: AuditEvent) -> AuditEvent: ...


class MongoConversationRepository:
    """Keep business metadata separate from LangGraph-owned checkpoint documents."""

    def __init__(self, settings: Settings, *, client: MongoClient | None = None) -> None:
        if not settings.mongodb_uri:
            raise ValueError("MONGODB_URI is required for production conversations.")
        self.client = client or MongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
        self.database = self.client[settings.mongodb_database]
        self.threads = self.database["agent_threads"]
        self.audit_events = self.database["audit_events"]

    def create_thread(self, thread: ThreadRecord) -> ThreadRecord:
        self.threads.insert_one(thread.model_dump(mode="python"))
        return thread

    def get_thread(self, thread_id: str) -> ThreadRecord | None:
        document = self.threads.find_one({"thread_id": thread_id}, {"_id": 0})
        return ThreadRecord.model_validate(document) if document else None

    def list_threads(self, user_id: str) -> list[ThreadRecord]:
        cursor = self.threads.find(
            {"actor_scope.user_id": user_id},
            {"_id": 0},
        ).sort("updated_at", -1)
        return [ThreadRecord.model_validate(document) for document in cursor]

    def save_thread(self, thread: ThreadRecord) -> ThreadRecord:
        thread.updated_at = datetime.now(UTC)
        result = self.threads.replace_one(
            {"thread_id": thread.thread_id},
            thread.model_dump(mode="python"),
            upsert=False,
        )
        if result.matched_count != 1:
            raise KeyError(thread.thread_id)
        return thread

    def append_audit(self, event: AuditEvent) -> AuditEvent:
        self.audit_events.update_one(
            {"event_id": event.event_id},
            {"$setOnInsert": event.model_dump(mode="python")},
            upsert=True,
        )
        return event
