"""Business persistence for Agent threads and append-only audit events."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol

from pymongo import MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError

from semikb.config import Settings
from semikb.contracts.models import (
    ActiveConversationContext,
    AffectSignals,
    AuditEvent,
    ChatMessage,
    ThreadRecord,
)
from semikb.contracts.streaming import (
    AgentMessageRequestRecord,
    AgentMessageRequestStatus,
    AgentStreamErrorCode,
    UnderstandingAudit,
)


class MessageRequestConflictError(ValueError):
    """The same idempotency key was reused for different content."""


class MessageRequestInProgressError(RuntimeError):
    """The same idempotent request is already accepted or running."""


class ThreadBusyError(MessageRequestInProgressError):
    """Another request currently owns the thread write lease."""


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
        summary_upto_message_id: str | None,
        active_context: ActiveConversationContext,
        context_version: int,
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
        self.thread_lease_seconds = settings.agent_thread_lease_seconds

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
        """Compatibility-only whole-document save; reject writes over an active request."""

        if thread.active_request_id is not None:
            raise ThreadBusyError(thread.thread_id)
        thread.updated_at = datetime.now(UTC)
        result = self.threads.replace_one(
            {
                "thread_id": thread.thread_id,
                "$or": [
                    {"active_request_id": {"$exists": False}},
                    {"active_request_id": None},
                ],
            },
            thread.model_dump(mode="python"),
            upsert=False,
        )
        if result.matched_count != 1:
            if self.threads.count_documents({"thread_id": thread.thread_id}, limit=1):
                raise ThreadBusyError(thread.thread_id)
            raise KeyError(thread.thread_id)
        return thread

    def prepare_message_request(
        self,
        record: AgentMessageRequestRecord,
    ) -> tuple[AgentMessageRequestRecord, bool]:
        existing = self.get_message_request(
            record.thread_id,
            record.actor_user_id,
            record.request_id,
        )
        if existing is not None:
            return self._prepare_existing_request(existing, record)

        turn_seq = self._claim_thread(record, allocate_user_turn=True)
        record.user_turn_seq = turn_seq
        try:
            self.message_requests.insert_one(record.model_dump(mode="python"))
            return record, False
        except DuplicateKeyError:
            self._release_thread(record)
            existing = self.get_message_request(
                record.thread_id,
                record.actor_user_id,
                record.request_id,
            )
        except Exception:
            self._release_thread(record)
            raise
        if existing is None:
            raise RuntimeError("message request disappeared after duplicate-key conflict")
        return self._prepare_existing_request(existing, record)

    def _prepare_existing_request(
        self,
        existing: AgentMessageRequestRecord,
        record: AgentMessageRequestRecord,
    ) -> tuple[AgentMessageRequestRecord, bool]:
        key = self._request_key(existing)
        if existing.content_sha256 != record.content_sha256:
            raise MessageRequestConflictError(record.request_id)
        if existing.status is AgentMessageRequestStatus.COMPLETED:
            return existing, True
        retry_statuses = [
            AgentMessageRequestStatus.FAILED,
            AgentMessageRequestStatus.CANCELLED,
        ]
        request_filter: dict[str, object] = {}
        if existing.status in {
            AgentMessageRequestStatus.ACCEPTED,
            AgentMessageRequestStatus.RUNNING,
        }:
            stale_cutoff = datetime.now(UTC) - timedelta(seconds=self.thread_lease_seconds)
            retry_statuses = [
                AgentMessageRequestStatus.ACCEPTED,
                AgentMessageRequestStatus.RUNNING,
            ]
            request_filter = {"updated_at": {"$lt": stale_cutoff}}

        self._claim_thread(existing, allocate_user_turn=False)
        now = datetime.now(UTC)
        document = self.message_requests.find_one_and_update(
            {
                **key,
                "content_sha256": record.content_sha256,
                "status": {"$in": retry_statuses},
                **request_filter,
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
                    "interaction_mode": None,
                    "route_decision": None,
                    "route_confidence": None,
                    "task_items": [],
                    "task_decisions": [],
                    "context_message_ids": [],
                    "standalone_query": "",
                    "retrieval_skipped_reason": None,
                    "slot_operations": [],
                    "inherited_slots": {},
                    "invalidated_context_refs": [],
                    "cancel_scope": None,
                    "affect": AffectSignals().model_dump(mode="python"),
                    "understanding_audit": UnderstandingAudit().model_dump(mode="python"),
                    "direct_reply_audit": None,
                    "error_code": None,
                },
                "$inc": {"attempt": 1},
            },
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            self._release_thread(existing)
            raise MessageRequestInProgressError(record.request_id)
        return AgentMessageRequestRecord.model_validate(document), False

    def _claim_thread(
        self,
        record: AgentMessageRequestRecord,
        *,
        allocate_user_turn: bool,
    ) -> int | None:
        now = datetime.now(UTC)
        stale_cutoff = now - timedelta(seconds=self.thread_lease_seconds)
        base_seq = {"$ifNull": ["$next_turn_seq", {"$add": [{"$size": {"$ifNull": ["$messages", []]}}, 1]}]}
        assignments: dict[str, object] = {
            "active_request_id": record.request_id,
            "active_request_started_at": now,
            "updated_at": now,
        }
        if allocate_user_turn:
            assignments.update(
                {
                    "last_turn_seq": base_seq,
                    "next_turn_seq": {"$add": [base_seq, 1]},
                }
            )
        document = self.threads.find_one_and_update(
            {
                "thread_id": record.thread_id,
                "actor_scope.user_id": record.actor_user_id,
                "$or": [
                    {"active_request_id": {"$exists": False}},
                    {"active_request_id": None},
                    {
                        "active_request_id": record.request_id,
                        "active_request_started_at": {"$lt": stale_cutoff},
                    },
                ],
            },
            [{"$set": assignments}],
            projection={"_id": 0, "last_turn_seq": 1},
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            if self.threads.count_documents({"thread_id": record.thread_id}, limit=1) == 0:
                raise KeyError(record.thread_id)
            raise ThreadBusyError(record.thread_id)
        return int(document["last_turn_seq"]) if allocate_user_turn else None

    def _release_thread(self, record: AgentMessageRequestRecord) -> None:
        self.threads.update_one(
            {"thread_id": record.thread_id, "active_request_id": record.request_id},
            {
                "$set": {"updated_at": datetime.now(UTC)},
                "$unset": {"active_request_id": "", "active_request_started_at": ""},
            },
        )

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
        if message.request_id:
            request_filter["active_request_id"] = message.request_id
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
        summary_upto_message_id: str | None,
        active_context: ActiveConversationContext,
        context_version: int,
        pending_fields: list[str],
        clarification_round: int,
    ) -> ThreadRecord:
        now = datetime.now(UTC)
        next_seq = {
            "$ifNull": [
                "$next_turn_seq",
                {"$add": [{"$size": {"$ifNull": ["$messages", []]}}, 1]},
            ]
        }
        message_document = message.model_dump(mode="python", exclude={"turn_seq"})
        document = self.threads.find_one_and_update(
            {
                "thread_id": thread_id,
                "active_request_id": message.request_id,
                "messages": {
                    "$not": {
                        "$elemMatch": {
                            "request_id": message.request_id,
                            "role": "assistant",
                        }
                    }
                },
            },
            [
                {
                    "$set": {
                        "messages": {
                            "$concatArrays": [
                                {"$ifNull": ["$messages", []]},
                                [{"$mergeObjects": [message_document, {"turn_seq": next_seq}]}],
                            ]
                        },
                        "status": status,
                        "summary": summary,
                        "summary_upto_message_id": summary_upto_message_id,
                        "active_context": active_context.model_dump(mode="python"),
                        "context_version": context_version,
                        "last_turn_seq": next_seq,
                        "next_turn_seq": {"$add": [next_seq, 1]},
                        "pending_fields": pending_fields,
                        "clarification_round": clarification_round,
                        "updated_at": now,
                    }
                }
            ],
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
        if document is None:
            thread = self.get_thread(thread_id)
            if thread is None:
                raise KeyError(thread_id)
            if not any(
                item.request_id == message.request_id and item.role == "assistant"
                for item in thread.messages
            ):
                raise RuntimeError("assistant response persistence was not acknowledged")
            return thread
        thread = ThreadRecord.model_validate(document)
        message.turn_seq = thread.last_turn_seq
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
                    "assistant_turn_seq": record.assistant_turn_seq,
                    "trace_id": trace_id,
                    "interaction_mode": record.interaction_mode,
                    "route_decision": record.route_decision,
                    "route_confidence": record.route_confidence,
                    "task_items": [item.model_dump(mode="python") for item in record.task_items],
                    "task_decisions": [
                        item.model_dump(mode="python") for item in record.task_decisions
                    ],
                    "context_message_ids": record.context_message_ids,
                    "standalone_query": record.standalone_query,
                    "retrieval_skipped_reason": record.retrieval_skipped_reason,
                    "slot_operations": [
                        item.model_dump(mode="python") for item in record.slot_operations
                    ],
                    "inherited_slots": record.inherited_slots,
                    "invalidated_context_refs": record.invalidated_context_refs,
                    "cancel_scope": record.cancel_scope,
                    "affect": record.affect.model_dump(mode="python"),
                    "understanding_audit": record.understanding_audit.model_dump(mode="python"),
                    "direct_reply_audit": (
                        record.direct_reply_audit.model_dump(mode="python")
                        if record.direct_reply_audit is not None
                        else None
                    ),
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
                self._release_thread(existing)
                return existing
            raise RuntimeError("message request terminal transition was not acknowledged")
        terminal = AgentMessageRequestRecord.model_validate(document)
        self._release_thread(terminal)
        return terminal

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
