"""Business persistence for Agent threads and append-only audit events."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from pymongo import MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError

from semikb.config import Settings
from semikb.contracts.models import AuditEvent, ChatMessage, ThreadRecord
from semikb.contracts.streaming import (
    AgentMessageRequestRecord,
    AgentMessageRequestStatus,
    AgentStreamErrorCode,
)


class MessageRequestConflictError(ValueError):
    """The same idempotency key was reused for different content."""


class MessageRequestInProgressError(RuntimeError):
    """The same idempotent request is already accepted or running."""


class ConversationRepository(Protocol):
    def create_thread(self, thread: ThreadRecord) -> ThreadRecord: ...

    def get_thread(self, thread_id: str) -> ThreadRecord | None: ...

    def list_threads(self, user_id: str) -> list[ThreadRecord]: ...

    def save_thread(self, thread: ThreadRecord) -> ThreadRecord: ...

    def prepare_message_request(
        self,
        record: AgentMessageRequestRecord,
    ) -> tuple[AgentMessageRequestRecord, bool]: ...

    def get_message_request(
        self,
        thread_id: str,
        actor_user_id: str,
        request_id: str,
    ) -> AgentMessageRequestRecord | None: ...

    def append_message_once(self, thread_id: str, message: ChatMessage) -> ThreadRecord: ...

    def finalize_stream_response(
        self,
        thread_id: str,
        message: ChatMessage,
        *,
        status: str,
        summary: str,
        pending_fields: list[str],
        clarification_round: int,
    ) -> ThreadRecord: ...

    def mark_message_request_running(
        self,
        record: AgentMessageRequestRecord,
    ) -> AgentMessageRequestRecord: ...

    def mark_message_request_terminal(
        self,
        record: AgentMessageRequestRecord,
        status: AgentMessageRequestStatus,
        *,
        result_payload: dict[str, object] | None = None,
        assistant_message_id: str | None = None,
        trace_id: str | None = None,
        error_code: AgentStreamErrorCode | None = None,
    ) -> AgentMessageRequestRecord: ...

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
        self.message_requests = self.database["agent_message_requests"]
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

    def prepare_message_request(
        self,
        record: AgentMessageRequestRecord,
    ) -> tuple[AgentMessageRequestRecord, bool]:
        key = self._request_key(record)
        try:
            self.message_requests.insert_one(record.model_dump(mode="python"))
            return record, False
        except DuplicateKeyError:
            existing = self.get_message_request(
                record.thread_id,
                record.actor_user_id,
                record.request_id,
            )
        if existing is None:
            raise RuntimeError("message request disappeared after duplicate-key conflict")
        if existing.content_sha256 != record.content_sha256:
            raise MessageRequestConflictError(record.request_id)
        if existing.status is AgentMessageRequestStatus.COMPLETED:
            return existing, True
        if existing.status in {
            AgentMessageRequestStatus.ACCEPTED,
            AgentMessageRequestStatus.RUNNING,
        }:
            raise MessageRequestInProgressError(record.request_id)

        now = datetime.now(UTC)
        document = self.message_requests.find_one_and_update(
            {
                **key,
                "content_sha256": record.content_sha256,
                "status": {
                    "$in": [
                        AgentMessageRequestStatus.FAILED,
                        AgentMessageRequestStatus.CANCELLED,
                    ]
                },
            },
            {
                "$set": {
                    "status": AgentMessageRequestStatus.ACCEPTED,
                    "run_id": record.run_id,
                    "updated_at": now,
                    "finished_at": None,
                    "assistant_message_id": None,
                    "trace_id": None,
                    "result_payload": {},
                    "error_code": None,
                },
                "$inc": {"attempt": 1},
            },
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise MessageRequestInProgressError(record.request_id)
        return AgentMessageRequestRecord.model_validate(document), False

    def get_message_request(
        self,
        thread_id: str,
        actor_user_id: str,
        request_id: str,
    ) -> AgentMessageRequestRecord | None:
        document = self.message_requests.find_one(
            {
                "thread_id": thread_id,
                "actor_user_id": actor_user_id,
                "request_id": request_id,
            },
            {"_id": 0},
        )
        return AgentMessageRequestRecord.model_validate(document) if document else None

    def append_message_once(self, thread_id: str, message: ChatMessage) -> ThreadRecord:
        now = datetime.now(UTC)
        request_filter: dict[str, object] = {"thread_id": thread_id}
        if message.request_id:
            request_filter["messages"] = {
                "$not": {
                    "$elemMatch": {
                        "request_id": message.request_id,
                        "role": message.role,
                    }
                }
            }
        result = self.threads.update_one(
            request_filter,
            {"$push": {"messages": message.model_dump(mode="python")}, "$set": {"updated_at": now}},
        )
        if result.matched_count == 0:
            thread = self.get_thread(thread_id)
            if thread is None:
                raise KeyError(thread_id)
            if not message.request_id or not any(
                item.request_id == message.request_id and item.role == message.role
                for item in thread.messages
            ):
                raise RuntimeError("thread message append was not acknowledged")
            return thread
        thread = self.get_thread(thread_id)
        if thread is None:
            raise KeyError(thread_id)
        return thread

    def finalize_stream_response(
        self,
        thread_id: str,
        message: ChatMessage,
        *,
        status: str,
        summary: str,
        pending_fields: list[str],
        clarification_round: int,
    ) -> ThreadRecord:
        now = datetime.now(UTC)
        result = self.threads.update_one(
            {
                "thread_id": thread_id,
                "messages": {
                    "$not": {
                        "$elemMatch": {
                            "request_id": message.request_id,
                            "role": "assistant",
                        }
                    }
                },
            },
            {
                "$push": {"messages": message.model_dump(mode="python")},
                "$set": {
                    "status": status,
                    "summary": summary,
                    "pending_fields": pending_fields,
                    "clarification_round": clarification_round,
                    "updated_at": now,
                },
            },
        )
        if result.matched_count == 0:
            thread = self.get_thread(thread_id)
            if thread is None:
                raise KeyError(thread_id)
            if not any(
                item.request_id == message.request_id and item.role == "assistant"
                for item in thread.messages
            ):
                raise RuntimeError("assistant response persistence was not acknowledged")
            return thread
        thread = self.get_thread(thread_id)
        if thread is None:
            raise KeyError(thread_id)
        return thread

    def mark_message_request_running(
        self,
        record: AgentMessageRequestRecord,
    ) -> AgentMessageRequestRecord:
        document = self.message_requests.find_one_and_update(
            {**self._request_key(record), "status": AgentMessageRequestStatus.ACCEPTED},
            {"$set": {"status": AgentMessageRequestStatus.RUNNING, "updated_at": datetime.now(UTC)}},
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            raise MessageRequestInProgressError(record.request_id)
        return AgentMessageRequestRecord.model_validate(document)

    def mark_message_request_terminal(
        self,
        record: AgentMessageRequestRecord,
        status: AgentMessageRequestStatus,
        *,
        result_payload: dict[str, object] | None = None,
        assistant_message_id: str | None = None,
        trace_id: str | None = None,
        error_code: AgentStreamErrorCode | None = None,
    ) -> AgentMessageRequestRecord:
        if status not in {
            AgentMessageRequestStatus.COMPLETED,
            AgentMessageRequestStatus.FAILED,
            AgentMessageRequestStatus.CANCELLED,
        }:
            raise ValueError("terminal request status required")
        now = datetime.now(UTC)
        document = self.message_requests.find_one_and_update(
            {
                **self._request_key(record),
                "status": {
                    "$in": [
                        AgentMessageRequestStatus.ACCEPTED,
                        AgentMessageRequestStatus.RUNNING,
                    ]
                },
            },
            {
                "$set": {
                    "status": status,
                    "result_payload": result_payload or {},
                    "assistant_message_id": assistant_message_id,
                    "trace_id": trace_id,
                    "error_code": error_code,
                    "updated_at": now,
                    "finished_at": now,
                }
            },
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            existing = self.get_message_request(
                record.thread_id,
                record.actor_user_id,
                record.request_id,
            )
            if existing is not None and existing.status is status:
                return existing
            raise RuntimeError("message request terminal transition was not acknowledged")
        return AgentMessageRequestRecord.model_validate(document)

    @staticmethod
    def _request_key(record: AgentMessageRequestRecord) -> dict[str, str]:
        return {
            "thread_id": record.thread_id,
            "actor_user_id": record.actor_user_id,
            "request_id": record.request_id,
        }

    def append_audit(self, event: AuditEvent) -> AuditEvent:
        self.audit_events.update_one(
            {"event_id": event.event_id},
            {"$setOnInsert": event.model_dump(mode="python")},
            upsert=True,
        )
        return event
