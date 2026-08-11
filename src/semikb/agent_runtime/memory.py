"""Explicitly approved cross-thread memory backed by a LangGraph Store."""

from __future__ import annotations

from datetime import UTC, datetime

from langgraph.store.base import BaseStore

from semikb.contracts.models import (
    ActorScope,
    ApprovalStatus,
    CreateMemoryRequest,
    MemoryRecord,
)


class MemoryService:
    """Long-term memory is user-approved context, never an automatic fact source."""

    def __init__(self, store: BaseStore) -> None:
        self.store = store

    @staticmethod
    def namespace(user_id: str) -> tuple[str, str]:
        return (user_id, "approved_memories")

    def create(self, request: CreateMemoryRequest, actor_scope: ActorScope) -> MemoryRecord:
        if request.memory_type != "preference" and not {
            "admin",
            "knowledge_admin",
        }.intersection(actor_scope.roles):
            raise PermissionError("Only knowledge administrators can approve governed memories.")
        record = MemoryRecord(
            user_id=actor_scope.user_id,
            memory_type=request.memory_type,
            content=request.content,
            scope=request.scope,
            source_refs=request.source_refs,
            confidence=request.confidence,
            approval_status=ApprovalStatus.APPROVED,
            expires_at=request.expires_at,
            created_by=actor_scope.user_id,
        )
        self.store.put(
            self.namespace(actor_scope.user_id),
            record.memory_id,
            record.model_dump(mode="json"),
        )
        return record

    def list(self, actor_scope: ActorScope) -> list[MemoryRecord]:
        current = datetime.now(UTC)
        records: list[MemoryRecord] = []
        for item in self.store.search(self.namespace(actor_scope.user_id), limit=100):
            record = MemoryRecord.model_validate(item.value)
            if record.approval_status is not ApprovalStatus.APPROVED:
                continue
            if record.expires_at and record.expires_at <= current:
                continue
            records.append(record)
        return sorted(records, key=lambda record: record.created_at, reverse=True)

    def delete(self, memory_id: str, actor_scope: ActorScope) -> None:
        namespace = self.namespace(actor_scope.user_id)
        if self.store.get(namespace, memory_id) is None:
            raise KeyError(memory_id)
        self.store.delete(namespace, memory_id)

    def approved_preferences(self, user_id: str) -> list[str]:
        scope = ActorScope(user_id=user_id)
        return [
            record.content
            for record in self.list(scope)
            if record.memory_type == "preference"
        ]
