from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from semikb.agent_runtime.context import ContextAssembler
from semikb.agent_runtime.graph import ConversationGraph
from semikb.config import Settings
from semikb.contracts.models import (
    ActiveConversationContext,
    ActorScope,
    ChatMessage,
    ContextEvidenceRef,
    ContextSlot,
    ThreadRecord,
)
from semikb.storage.conversations import ThreadBusyError


def _long_thread(rounds: int = 16) -> ThreadRecord:
    created = datetime(2026, 8, 13, tzinfo=UTC)
    messages: list[ChatMessage] = []
    for index in range(rounds):
        messages.extend(
            (
                ChatMessage(
                    message_id=f"msg_user_{index:02d}",
                    turn_seq=index * 2 + 1,
                    role="user",
                    content=f"第 {index + 1} 轮用户问题",
                    created_at=created + timedelta(minutes=index * 2),
                ),
                ChatMessage(
                    message_id=f"msg_assistant_{index:02d}",
                    turn_seq=index * 2 + 2,
                    role="assistant",
                    content=f"第 {index + 1} 轮助手回答",
                    created_at=created + timedelta(minutes=index * 2 + 1),
                ),
            )
        )
    return ThreadRecord(
        thread_id="thread_long_context",
        actor_scope=ActorScope(user_id="context_owner"),
        messages=messages,
        next_turn_seq=len(messages) + 1,
        last_turn_seq=len(messages),
    )


def test_legacy_thread_document_is_backward_compatible_and_normalized() -> None:
    thread = ThreadRecord.model_validate(
        {
            "thread_id": "thread_legacy_context",
            "messages": [
                {"message_id": "legacy_user", "role": "user", "content": "旧问题"},
                {"message_id": "legacy_answer", "role": "assistant", "content": "旧回答"},
            ],
        }
    )

    messages = ContextAssembler(Settings(demo_mode=True)).normalized_messages(thread)

    assert [message.turn_seq for message in messages] == [1, 2]
    assert thread.context_version == 1
    assert thread.active_context.slots == {}


def test_context_assembler_keeps_recent_turns_exact_and_summarizes_only_older_history() -> None:
    thread = _long_thread()
    assembler = ContextAssembler(
        Settings(demo_mode=True, agent_context_recent_turns=12)
    )

    context = assembler.assemble(thread, approved_preferences=["回答使用中文"])

    assert len(context.recent_messages) == 24
    assert context.recent_messages[0].message_id == "msg_user_04"
    assert context.recent_messages[-1].message_id == "msg_assistant_15"
    assert context.summary_upto_message_id == "msg_assistant_03"
    assert "第 1 轮用户问题" in context.summary
    assert "第 5 轮用户问题" not in context.summary
    assert context.approved_preferences == ["回答使用中文"]


def test_context_compaction_advances_from_the_persisted_summary_boundary() -> None:
    thread = _long_thread(rounds=14)
    assembler = ContextAssembler(Settings(demo_mode=True, agent_context_recent_turns=12))
    first = assembler.compact_thread(thread)
    thread.summary = first.summary
    thread.summary_upto_message_id = first.summary_upto_message_id
    thread.messages.extend(
        (
            ChatMessage(message_id="msg_user_14", turn_seq=29, role="user", content="第 15 轮用户问题"),
            ChatMessage(
                message_id="msg_assistant_14",
                turn_seq=30,
                role="assistant",
                content="第 15 轮助手回答",
            ),
        )
    )

    second = assembler.compact_thread(thread)

    assert second.summary.startswith(first.summary)
    assert second.summary.count("第 1 轮用户问题") == 1
    assert "第 3 轮助手回答" in second.summary
    assert second.summary_upto_message_id == "msg_assistant_02"


def test_context_assembler_excludes_current_message_from_prior_history() -> None:
    thread = _long_thread(rounds=2)
    assembler = ContextAssembler(Settings(demo_mode=True))

    context = assembler.assemble(
        thread,
        current_message_id="msg_assistant_01",
    )

    assert [item.message_id for item in context.recent_messages] == [
        "msg_user_00",
        "msg_assistant_00",
        "msg_user_01",
    ]


def test_slot_source_and_dependency_invalidation_preserve_unrelated_context() -> None:
    thread = ThreadRecord(
        thread_id="thread_slot_context",
        messages=[
            ChatMessage(message_id="msg_old", turn_seq=1, role="user", content="检查 ETCH-03 Chamber B"),
            ChatMessage(message_id="msg_new", turn_seq=2, role="user", content="不是 ETCH-03，是 ETCH-04"),
        ],
        active_context=ActiveConversationContext(
            slots={
                "product": ContextSlot(value="P-ALPHA", source_message_id="msg_old"),
                "tool_id": ContextSlot(value="ETCH-03", source_message_id="msg_old"),
                "chamber": ContextSlot(
                    value="B",
                    source_message_id="msg_old",
                    depends_on=["tool_id"],
                ),
                "recipe_version": ContextSlot(
                    value="V2.3",
                    source_message_id="msg_old",
                    depends_on=["tool_id", "chamber"],
                ),
            },
            evidence_refs=[
                ContextEvidenceRef(
                    evidence_id="chunk:SOP-001",
                    source_type="internal_controlled",
                    source_message_id="msg_old",
                )
            ],
        ),
    )
    assembler = ContextAssembler(Settings(demo_mode=True))

    context = assembler.update_active_context(
        thread,
        {"constraints": {"tool_id": "ETCH-04"}},
        source_message_id="msg_new",
    )

    assert context.slots["tool_id"].value == "ETCH-04"
    assert context.slots["tool_id"].source_message_id == "msg_new"
    assert context.slots["product"].valid is True
    assert context.slots["chamber"].valid is False
    assert context.slots["recipe_version"].valid is False
    assert context.evidence_refs[0].valid is False


def test_graph_accepts_only_sourced_valid_context_slots() -> None:
    context = {
        "active_context": {
            "slots": {
                "tool_id": {
                    "value": "ETCH-03",
                    "valid": True,
                    "source_message_id": "msg_source",
                }
            }
        }
    }

    assert ConversationGraph._constraint_is_grounded("tool_id", "ETCH-03", "它呢", context)
    context["active_context"]["slots"]["tool_id"]["valid"] = False
    assert not ConversationGraph._constraint_is_grounded("tool_id", "ETCH-03", "它呢", context)


@pytest.mark.asyncio
async def test_different_request_ids_cannot_run_concurrently_in_one_thread(seeded_services) -> None:
    _, _, _, conversation, _ = seeded_services
    scope = ActorScope(user_id="thread_order_owner")
    thread = conversation.create_thread("thread ordering", scope)

    first = await conversation.prepare_stream_message(
        thread.thread_id,
        "ETCH-03 当前 SOP 是什么？",
        "req_thread_order_001",
        scope,
    )
    with pytest.raises(ThreadBusyError):
        await conversation.prepare_stream_message(
            thread.thread_id,
            "再检查 Chamber B",
            "req_thread_order_002",
            scope,
        )

    await conversation.cancel_stream_message(thread.thread_id, first.record.request_id, scope)
    second = await conversation.prepare_stream_message(
        thread.thread_id,
        "再检查 Chamber B",
        "req_thread_order_002",
        scope,
    )
    assert second.record.user_turn_seq == 2


@pytest.mark.asyncio
async def test_completed_messages_have_stable_monotonic_sequences_and_active_slots(
    seeded_services,
) -> None:
    store, _, _, conversation, _ = seeded_services
    scope = ActorScope(user_id="sequence_owner")
    thread = conversation.create_thread("sequence", scope)

    await conversation.send_message(thread.thread_id, "ETCH-03 Chamber B 当前 SOP 是什么？", scope)
    persisted = store.get_thread(thread.thread_id)

    assert persisted is not None
    assert [message.turn_seq for message in persisted.messages] == [1, 2]
    assert persisted.next_turn_seq == 3
    assert persisted.last_turn_seq == 2
    assert persisted.active_request_id is None
    assert persisted.active_context.slots["tool_id"].value == "ETCH-03"
    assert persisted.active_context.slots["chamber"].value == "B"


def test_context_isolation_uses_only_the_selected_thread() -> None:
    assembler = ContextAssembler(Settings(demo_mode=True))
    owner_thread = ThreadRecord(
        thread_id="thread_owner",
        actor_scope=ActorScope(user_id="owner"),
        messages=[ChatMessage(turn_seq=1, role="user", content="OWNER-SECRET")],
    )
    other_thread = ThreadRecord(
        thread_id="thread_other",
        actor_scope=ActorScope(user_id="other"),
        messages=[ChatMessage(turn_seq=1, role="user", content="OTHER-SECRET")],
    )

    owner_context = assembler.assemble(owner_thread)

    assert owner_context.thread_id == "thread_owner"
    assert "OWNER-SECRET" in owner_context.recent_messages[0].content
    assert all("OTHER-SECRET" not in item.content for item in owner_context.recent_messages)
    assert assembler.assemble(other_thread).thread_id == "thread_other"
