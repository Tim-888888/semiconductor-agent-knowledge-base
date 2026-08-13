"""Bounded, source-aware conversation context assembly."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from semikb.config import Settings
from semikb.contracts.models import (
    ActiveConversationContext,
    AssembledConversationContext,
    ContextEvidenceRef,
    ContextSlot,
    ConversationContextMessage,
    SlotOperation,
    SlotOperationKind,
    ThreadRecord,
    utc_now,
)

SLOT_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "chamber": ("tool_id",),
    "recipe_id": ("tool_id", "chamber"),
    "recipe_version": ("recipe_id", "tool_id", "chamber"),
}
CONTEXT_SLOT_NAMES = (
    "product",
    "process_layer",
    "tool_id",
    "chamber",
    "recipe_id",
    "recipe_version",
    "time_range",
    "lot_id",
    "case_id",
)


@dataclass(frozen=True, slots=True)
class ContextCompaction:
    summary: str
    summary_upto_message_id: str | None


class ContextAssembler:
    """Build graph context without treating summaries as exact conversation history."""

    def __init__(self, settings: Settings) -> None:
        self.recent_turns = settings.agent_context_recent_turns
        self.summary_max_chars = settings.agent_context_summary_max_chars

    def assemble(
        self,
        thread: ThreadRecord,
        *,
        current_message_id: str | None = None,
        approved_preferences: list[str] | None = None,
    ) -> AssembledConversationContext:
        normalized = self.normalized_messages(thread)
        history = [item for item in normalized if item.message_id != current_message_id]
        keep_count = self.recent_turns * 2
        recent = history[-keep_count:]
        older = history[:-keep_count]
        compaction = self._incremental_compaction(thread, older)
        return AssembledConversationContext(
            thread_id=thread.thread_id,
            context_version=thread.context_version,
            summary=compaction.summary,
            summary_upto_message_id=compaction.summary_upto_message_id,
            recent_messages=[
                ConversationContextMessage(
                    message_id=item.message_id,
                    turn_seq=int(item.turn_seq),
                    role=item.role,
                    content=item.content,
                    created_at=item.created_at,
                )
                for item in recent
            ],
            active_context=thread.active_context,
            approved_preferences=list(approved_preferences or []),
            current_message_id=current_message_id,
        )

    def compact_thread(self, thread: ThreadRecord) -> ContextCompaction:
        messages = self.normalized_messages(thread)
        keep_count = self.recent_turns * 2
        return self._incremental_compaction(thread, messages[:-keep_count])

    @staticmethod
    def normalized_messages(thread: ThreadRecord):
        messages = [message.model_copy(deep=True) for message in thread.messages]
        used: set[int] = set()
        next_fallback = 1
        for message in messages:
            if message.turn_seq is not None and message.turn_seq not in used:
                used.add(message.turn_seq)
                next_fallback = max(next_fallback, message.turn_seq + 1)
                continue
            while next_fallback in used:
                next_fallback += 1
            message.turn_seq = next_fallback
            used.add(next_fallback)
            next_fallback += 1
        return sorted(messages, key=lambda item: (int(item.turn_seq), item.created_at, item.message_id))

    def update_active_context(
        self,
        thread: ThreadRecord,
        result: dict[str, Any],
        *,
        source_message_id: str,
    ) -> ActiveConversationContext:
        context = thread.active_context.model_copy(deep=True)
        now = utc_now()
        constraints = result.get("constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}

        raw_operations = result.get("slot_operations", [])
        operations = []
        if isinstance(raw_operations, list):
            for item in raw_operations:
                try:
                    operations.append(SlotOperation.model_validate(item))
                except ValueError:
                    continue
        if str(result.get("route_decision", "")) == "refuse":
            constraints = {}
            operations = []

        changed: set[str] = {
            item.slot_name
            for item in operations
            if item.operation in {SlotOperationKind.CORRECT, SlotOperationKind.CLEAR}
        }
        for name in CONTEXT_SLOT_NAMES:
            raw = constraints.get(name)
            if raw in (None, ""):
                continue
            value = str(raw)
            existing = context.slots.get(name)
            if existing is not None and existing.valid and existing.value != value:
                changed.add(name)

        if changed:
            self._invalidate_dependents(context, changed, source_message_id, now)

        for operation in operations:
            if operation.operation is not SlotOperationKind.CLEAR:
                continue
            existing = context.slots.get(operation.slot_name)
            if existing is None:
                continue
            existing.valid = False
            existing.invalidated_by_message_id = source_message_id
            existing.invalidation_reason = "user_cleared"
            existing.updated_at = now

        for name in CONTEXT_SLOT_NAMES:
            raw = constraints.get(name)
            if raw in (None, ""):
                continue
            value = str(raw)
            inherited = next(
                (
                    item
                    for item in operations
                    if item.slot_name == name and item.operation is SlotOperationKind.INHERIT
                ),
                None,
            )
            existing = context.slots.get(name)
            if inherited is not None and existing is not None and existing.valid:
                continue
            context.slots[name] = ContextSlot(
                value=value,
                source_message_id=(
                    inherited.source_message_id
                    if inherited is not None and inherited.source_message_id
                    else self._find_source_message(thread, value, source_message_id)
                ),
                source_kind="inherited" if inherited is not None else "explicit",
                depends_on=list(SLOT_DEPENDENCIES.get(name, ())),
                updated_at=now,
            )

        ledger = result.get("evidence_ledger", [])
        if isinstance(ledger, list) and ledger:
            context.evidence_refs = [
                ContextEvidenceRef(
                    evidence_id=str(item["evidence_id"]),
                    source_type=str(item.get("source_type", "unknown")),
                    source_message_id=source_message_id,
                    trace_id=str(result.get("trace_id")) if result.get("trace_id") else None,
                    updated_at=now,
                )
                for item in ledger
                if isinstance(item, dict) and item.get("evidence_id")
            ]
        if result.get("trace_id"):
            context.trace_id = str(result["trace_id"])
        if result.get("intent"):
            context.topic = str(result["intent"])
        context.updated_at = now
        return context

    @staticmethod
    def _invalidate_dependents(
        context: ActiveConversationContext,
        changed: set[str],
        source_message_id: str,
        now,
    ) -> None:
        for slot in context.slots.values():
            if slot.valid and changed.intersection(slot.depends_on):
                slot.valid = False
                slot.invalidated_by_message_id = source_message_id
                slot.invalidation_reason = "dependency_changed"
                slot.updated_at = now
        for ref in context.evidence_refs:
            if ref.valid:
                ref.valid = False
                ref.invalidated_by_message_id = source_message_id
                ref.updated_at = now

    @staticmethod
    def _find_source_message(thread: ThreadRecord, value: str, fallback: str) -> str:
        normalized = re.sub(r"\s+", "", value).lower()
        for message in reversed(ContextAssembler.normalized_messages(thread)):
            if message.role != "user":
                continue
            content = re.sub(r"\s+", "", message.content).lower()
            if normalized and normalized in content:
                return message.message_id
        return fallback

    def _incremental_compaction(self, thread: ThreadRecord, messages) -> ContextCompaction:
        if not messages:
            return ContextCompaction("", None)
        base_summary = ""
        pending = messages
        if thread.summary and thread.summary_upto_message_id:
            for index, message in enumerate(messages):
                if message.message_id == thread.summary_upto_message_id:
                    base_summary = thread.summary
                    pending = messages[index + 1 :]
                    break
        lines: list[str] = []
        for message in pending:
            content = " ".join(message.content.split())
            if len(content) > 240:
                content = content[:237] + "..."
            role = "USER" if message.role == "user" else "ASSISTANT"
            lines.append(f"[{message.turn_seq}][{role}] {content}")
        summary = "\n".join(item for item in (base_summary, *lines) if item)
        if len(summary) > self.summary_max_chars:
            head_size = self.summary_max_chars // 3
            tail_size = self.summary_max_chars - head_size - 36
            summary = f"{summary[:head_size]}\n...[older turns omitted]...\n{summary[-tail_size:]}"
        return ContextCompaction(summary, messages[-1].message_id)
