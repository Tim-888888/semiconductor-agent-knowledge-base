from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime

import pytest

from semikb.agent_runtime.llm_gateway import LLMCompletion
from semikb.agent_runtime.routing import RoutePolicy
from semikb.agent_runtime.service import ConversationService
from semikb.agent_runtime.tools import ManufacturingToolbox
from semikb.agent_runtime.understanding import ConversationUnderstandingService
from semikb.config import Settings
from semikb.contracts.models import (
    ActiveConversationContext,
    ActorScope,
    AgentRoute,
    ChatMessage,
    ContextEvidenceRef,
    ContextSlot,
    ConversationUnderstanding,
    IntentTarget,
    IntentTaskAction,
    IntentTaskItem,
    InteractionMode,
    MessageRenderMode,
    PrimaryIntent,
    TaskExecutionDecision,
    TaskExecutionStatus,
)
from semikb.demo_factory import demo_actor_scope


class CountingRetrieval:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.search_calls = 0
        self.reuse_calls = 0

    def search(self, *args, **kwargs):
        self.search_calls += 1
        return self.delegate.search(*args, **kwargs)

    def reuse_trace_evidence(self, *args, **kwargs):
        self.reuse_calls += 1
        return self.delegate.reuse_trace_evidence(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.delegate, name)


class CountingToolbox(ManufacturingToolbox):
    def __init__(self) -> None:
        self.query_calls = 0

    def query_for_case(self, query, constraints):
        self.query_calls += 1
        return super().query_for_case(query, constraints)


class CountingWeb:
    def __init__(self) -> None:
        self.search_calls = 0

    async def search(self, query):
        self.search_calls += 1
        return []


class ForbiddenLLM:
    async def complete(self, *args, **kwargs):
        raise AssertionError("L0 must not call the LLM")


class FixedUnderstandingLLM:
    async def complete(self, *args, **kwargs):
        payload = {
            "interaction_mode": "task",
            "primary_intent": "investigation",
            "task_items": [
                {
                    "task_id": "task_1",
                    "primary_intent": "data_query",
                    "target_type": "wafer_map",
                    "action": "recall",
                    "depends_on": [],
                    "execution_policy": "execute",
                }
            ],
            "affect": {
                "sentiment": "neutral",
                "urgency": "normal",
                "complaint_signal": False,
            },
            "slot_operations": [],
            "explicit_slots": [
                {"slot_name": "product", "value": "P-ALPHA"},
                {"slot_name": "tool_id", "value": "ETCH-03"},
                {"slot_name": "chamber", "value": "ETCH-03"},
                {"slot_name": "time_range", "value": "最近24小时"},
            ],
            "inherited_slot_names": [],
            "missing_slots": ["chamber"],
            "context_message_ids": [],
            "standalone_query": "分析 P-ALPHA ETCH-03 最近24小时晶圆图异常",
            "cancel_scope": None,
            "suggested_route": "tool_only",
            "confidence": 0.95,
        }
        return LLMCompletion(
            content=json.dumps(payload, ensure_ascii=False),
            provider="test",
            requested_model="test",
            reported_model="test",
            fallback_used=False,
            attempted_providers=("test",),
            usage={},
        )


class FalseCancelUnderstandingLLM(FixedUnderstandingLLM):
    async def complete(self, *args, **kwargs):
        completion = await super().complete(*args, **kwargs)
        payload = json.loads(completion.content)
        payload["cancel_scope"] = "current_task"
        payload["clarification_relation"] = "cancel_current"
        return LLMCompletion(
            content=json.dumps(payload, ensure_ascii=False),
            provider=completion.provider,
            requested_model=completion.requested_model,
            reported_model=completion.reported_model,
            fallback_used=completion.fallback_used,
            attempted_providers=completion.attempted_providers,
            usage=completion.usage,
        )


class UnsafeMixedLLM:
    async def complete(self, *args, **kwargs):
        payload = {
            "interaction_mode": "task",
            "primary_intent": "action_request",
            "task_items": [
                {
                    "task_id": f"task_{index}",
                    "primary_intent": "data_query",
                    "target_type": "fdc",
                    "action": "lookup",
                    "depends_on": [],
                    "execution_policy": "execute",
                }
                for index in range(1, 4)
            ],
            "affect": {
                "sentiment": "neutral",
                "urgency": "normal",
                "complaint_signal": False,
            },
            "slot_operations": [],
            "explicit_slots": [],
            "inherited_slot_names": [],
            "missing_slots": [],
            "context_message_ids": [],
            "standalone_query": "",
            "cancel_scope": None,
            "suggested_route": "tool_only",
            "confidence": 0.95,
        }
        return LLMCompletion(
            content=json.dumps(payload, ensure_ascii=False),
            provider="test",
            requested_model="test",
            reported_model="test",
            fallback_used=False,
            attempted_providers=("test",),
            usage={},
        )


class MisleadingHistoryLLM:
    async def complete(self, *args, **kwargs):
        payload = {
            "interaction_mode": "task",
            "primary_intent": "knowledge_query",
            "task_items": [
                {
                    "task_id": "task_1",
                    "primary_intent": "knowledge_query",
                    "target_type": "previous_user_message",
                    "action": "recall",
                    "depends_on": [],
                    "execution_policy": "execute",
                }
            ],
            "affect": {
                "sentiment": "neutral",
                "urgency": "normal",
                "complaint_signal": False,
            },
            "slot_operations": [],
            "explicit_slots": [],
            "inherited_slot_names": [],
            "missing_slots": [],
            "context_message_ids": ["msg_meta"],
            "standalone_query": "复述最近一条用户输入",
            "cancel_scope": None,
            "suggested_route": "internal_rag",
            "confidence": 0.96,
        }
        return LLMCompletion(
            content=json.dumps(payload, ensure_ascii=False),
            provider="test",
            requested_model="test",
            reported_model="test",
            fallback_used=False,
            attempted_providers=("test",),
            usage={},
        )


class MisclassifiedOutOfScopeLLM:
    async def complete(self, *args, **kwargs):
        payload = {
            "interaction_mode": "conversation",
            "primary_intent": "conversation",
            "task_items": [
                {
                    "task_id": "task_1",
                    "primary_intent": "conversation",
                    "target_type": "general",
                    "action": "explain",
                    "depends_on": [],
                    "execution_policy": "execute",
                }
            ],
            "affect": {
                "sentiment": "neutral",
                "urgency": "normal",
                "complaint_signal": False,
            },
            "slot_operations": [],
            "explicit_slots": [],
            "inherited_slot_names": [],
            "missing_slots": [],
            "context_message_ids": [],
            "standalone_query": "帮我写一首关于晚风的诗。",
            "cancel_scope": None,
            "suggested_route": "chat_direct",
            "confidence": 0.94,
        }
        return LLMCompletion(
            content=json.dumps(payload, ensure_ascii=False),
            provider="test",
            requested_model="test",
            reported_model="test",
            fallback_used=False,
            attempted_providers=("test",),
            usage={},
        )


class GovernedCorpusLookupLLM:
    async def complete(self, messages, **kwargs):
        request = json.loads(messages[-1]["content"])["current_request"]
        payload = {
            "interaction_mode": "task",
            "primary_intent": "knowledge_query",
            "task_items": [
                {
                    "task_id": "task_1",
                    "primary_intent": "knowledge_query",
                    "target_type": "general",
                    "action": "lookup",
                    "depends_on": [],
                    "execution_policy": "execute",
                }
            ],
            "affect": {
                "sentiment": "neutral",
                "urgency": "normal",
                "complaint_signal": False,
            },
            "slot_operations": [],
            "explicit_slots": [],
            "inherited_slot_names": [],
            "missing_slots": [],
            "context_message_ids": [],
            "standalone_query": request,
            "cancel_scope": None,
            "suggested_route": "internal_rag",
            "confidence": 0.94,
        }
        return LLMCompletion(
            content=json.dumps(payload, ensure_ascii=False),
            provider="test",
            requested_model="test",
            reported_model="test",
            fallback_used=False,
            attempted_providers=("test",),
            usage={},
        )


def _service(seeded_services):
    store, _, retrieval, _, _ = seeded_services
    counting_retrieval = CountingRetrieval(retrieval)
    toolbox = CountingToolbox()
    web = CountingWeb()
    service = ConversationService(
        store,
        counting_retrieval,
        Settings(_env_file=None, demo_mode=True),
        toolbox=toolbox,
        web_search=web,
    )
    return store, service, counting_retrieval, toolbox, web


@pytest.mark.asyncio
async def test_history_recall_uses_exact_message_and_skips_all_downstream(seeded_services) -> None:
    _, service, retrieval, toolbox, web = _service(seeded_services)
    scope = demo_actor_scope()
    thread = service.create_thread("history", scope)
    first_question = "ETCH-03 Chamber B 清腔后首片异常，当前 SOP 怎么要求？"

    first = await service.send_message(thread.thread_id, first_question, scope)
    search_calls = retrieval.search_calls
    tool_calls = toolbox.query_calls
    result = await service.send_message(thread.thread_id, "我刚才问什么了？", scope)

    assert first["route_decision"] is AgentRoute.INTERNAL_RAG
    assert result["route_decision"] is AgentRoute.CHAT_DIRECT
    assert result["interaction_mode"] is InteractionMode.CONVERSATION
    assert first_question in result["response"]
    assert result["citations"] == []
    assert result["retrieval_skipped_reason"] == "answer_available_without_external_retrieval"
    assert retrieval.search_calls == search_calls
    assert toolbox.query_calls == tool_calls
    assert web.search_calls == 0


@pytest.mark.asyncio
async def test_repeated_history_recall_skips_meta_question_and_returns_business_question(
    seeded_services,
) -> None:
    _, service, retrieval, toolbox, web = _service(seeded_services)
    scope = ActorScope()
    thread = service.create_thread("repeated-history", scope)
    first_question = "ETCH-03 Chamber B 清腔后首片异常，当前 SOP 怎么要求？"

    await service.send_message(thread.thread_id, first_question, scope)
    search_calls = retrieval.search_calls
    tool_calls = toolbox.query_calls
    first_recall = await service.send_message(thread.thread_id, "我刚刚说什么?", scope)
    second_recall = await service.send_message(thread.thread_id, "我刚刚说了什么?", scope)

    for result in (first_recall, second_recall):
        assert result["route_decision"] is AgentRoute.CHAT_DIRECT
        assert result["interaction_mode"] is InteractionMode.CONVERSATION
        assert first_question in result["response"]
        assert "我刚刚说什么" not in result["response"]
        assert result["citations"] == []
        assert result["retrieval_skipped_reason"] == "answer_available_without_external_retrieval"
    assert retrieval.search_calls == search_calls
    assert toolbox.query_calls == tool_calls
    assert web.search_calls == 0


@pytest.mark.asyncio
async def test_feedback_is_not_hijacked_into_retrieval(seeded_services) -> None:
    store, service, retrieval, toolbox, web = _service(seeded_services)
    scope = ActorScope()
    thread = service.create_thread("feedback", scope)

    result = await service.send_message(thread.thread_id, "回答太复杂了", scope)

    assert result["interaction_mode"] is InteractionMode.FEEDBACK
    assert result["route_decision"] is AgentRoute.CHAT_DIRECT
    assert result["thread"]["messages"][-1]["presentation"]["mode"] is MessageRenderMode.BUBBLE
    assert retrieval.search_calls == 0
    assert toolbox.query_calls == 0
    assert web.search_calls == 0
    request_id = result["thread"]["messages"][-1]["request_id"]
    record = store.get_message_request(thread.thread_id, scope.user_id, request_id)
    assert record is not None
    assert record.direct_reply_audit is not None
    assert record.direct_reply_audit.reply_kind == "feedback"


@pytest.mark.asyncio
async def test_help_is_natural_chat_without_downstream_calls(seeded_services) -> None:
    store, service, retrieval, toolbox, web = _service(seeded_services)
    scope = ActorScope()
    thread = service.create_thread("help", scope)

    result = await service.send_message(thread.thread_id, "/help", scope)

    assert result["route_decision"] is AgentRoute.CHAT_DIRECT
    assert result["thread"]["messages"][-1]["presentation"]["mode"] is MessageRenderMode.BUBBLE
    assert retrieval.search_calls == 0
    assert toolbox.query_calls == 0
    assert web.search_calls == 0
    request_id = result["thread"]["messages"][-1]["request_id"]
    record = store.get_message_request(thread.thread_id, scope.user_id, request_id)
    assert record is not None
    assert record.direct_reply_audit is not None
    assert record.direct_reply_audit.reply_kind == "general_chat"


@pytest.mark.asyncio
async def test_tool_only_does_not_call_embedding_retrieval(seeded_services) -> None:
    _, service, retrieval, toolbox, web = _service(seeded_services)
    thread = service.create_thread("fdc", ActorScope())

    result = await service.send_message(
        thread.thread_id,
        "查 P-ALPHA ETCH-03 Chamber B 最近24小时 FDC 报警",
    )

    assert result["route_decision"] is AgentRoute.TOOL_ONLY
    assert result["tool_facts"]
    assert retrieval.search_calls == 0
    assert toolbox.query_calls == 1
    assert web.search_calls == 0


@pytest.mark.asyncio
async def test_internal_rag_does_not_call_tool_or_web(seeded_services) -> None:
    _, service, retrieval, toolbox, web = _service(seeded_services)
    thread = service.create_thread("sop", ActorScope())

    result = await service.send_message(thread.thread_id, "当前 ETCH-03 SOP 怎么要求？")

    assert result["route_decision"] is AgentRoute.INTERNAL_RAG
    assert retrieval.search_calls == 1
    assert toolbox.query_calls == 0
    assert web.search_calls == 0


@pytest.mark.asyncio
async def test_mixed_task_keeps_three_items_and_refuses_only_recipe_write(seeded_services) -> None:
    store, service, retrieval, toolbox, _ = _service(seeded_services)
    thread = service.create_thread("mixed", ActorScope())

    result = await service.send_message(
        thread.thread_id,
        "查 P-ALPHA ETCH-03 最近24小时 FDC 报警、修改 Recipe、生成报告",
    )
    request_id = result["thread"]["messages"][-1]["request_id"]
    ledger = store.get_message_request(thread.thread_id, thread.actor_scope.user_id, request_id)

    assert result["route_decision"] is AgentRoute.TOOL_ONLY
    assert len(result["task_items"]) == 3
    assert ledger is not None
    assert [item.execution_policy for item in ledger.task_items] == [
        TaskExecutionDecision.EXECUTE,
        TaskExecutionDecision.REFUSE,
        TaskExecutionDecision.DEFER,
    ]
    assert [item["status"] for item in result["task_results"]] == [
        TaskExecutionStatus.COMPLETED,
        TaskExecutionStatus.REFUSED,
        TaskExecutionStatus.DEFERRED,
    ]
    assert [item["task_id"] for item in result["task_results"]] == [
        "task_1",
        "task_2",
        "task_3",
    ]
    assert [item.model_dump(mode="python") for item in ledger.task_results] == result[
        "task_results"
    ]
    assert result["thread"]["messages"][-1]["presentation"]["task_results"] == result[
        "task_results"
    ]
    assert retrieval.search_calls == 0
    assert toolbox.query_calls == 1


@pytest.mark.asyncio
async def test_history_plus_new_rag_task_executes_both_without_silent_omission(
    seeded_services,
) -> None:
    _, service, retrieval, toolbox, web = _service(seeded_services)
    scope = demo_actor_scope()
    thread = service.create_thread("history-plus-rag", scope)
    previous = "ETCH-03 Chamber B 清腔后首片异常，当前 SOP 怎么要求？"

    await service.send_message(thread.thread_id, previous, scope)
    search_calls = retrieval.search_calls
    result = await service.send_message(
        thread.thread_id,
        "我刚才问了什么，再查一下当前 ETCH-03 SOP",
        scope,
    )

    assert result["interaction_mode"] is InteractionMode.MIXED
    assert result["route_decision"] is AgentRoute.INTERNAL_RAG
    assert previous in result["response"]
    assert [item["status"] for item in result["task_results"]] == [
        TaskExecutionStatus.COMPLETED,
        TaskExecutionStatus.COMPLETED,
    ]
    assert [item["route"] for item in result["task_results"]] == [
        AgentRoute.CHAT_DIRECT,
        AgentRoute.INTERNAL_RAG,
    ]
    assert result["task_results"][0]["evidence_ids"] == []
    assert result["task_results"][0]["tool_fact_ids"] == []
    assert retrieval.search_calls == search_calls + 1
    assert toolbox.query_calls == 0
    assert web.search_calls == 0


@pytest.mark.asyncio
async def test_rag_and_tool_tasks_each_finish_with_matching_evidence(seeded_services) -> None:
    _, service, retrieval, toolbox, web = _service(seeded_services)
    thread = service.create_thread("rag-tool", demo_actor_scope())

    result = await service.send_message(
        thread.thread_id,
        "查 P-ALPHA ETCH-03 最近24小时 FDC 报警，再对照 SOP 给排查建议",
    )

    assert result["route_decision"] is AgentRoute.RAG_AND_TOOL
    assert result["task_results"]
    assert all(
        item["status"] is TaskExecutionStatus.COMPLETED
        for item in result["task_results"]
    )
    assert any(item["evidence_ids"] for item in result["task_results"])
    assert any(item["tool_fact_ids"] for item in result["task_results"])
    assert retrieval.search_calls == 1
    assert toolbox.query_calls == 1
    assert web.search_calls == 0


@pytest.mark.asyncio
async def test_slot_correction_invalidates_dependents_but_keeps_product(seeded_services) -> None:
    store, service, retrieval, toolbox, web = _service(seeded_services)
    scope = ActorScope(tool_ids=["ETCH-03", "ETCH-04"])
    thread = service.create_thread("correction", scope)
    now = datetime.now(UTC)
    source = ChatMessage(role="user", content="P-ALPHA ETCH-03 Chamber B", turn_seq=1)
    thread.messages.append(source)
    thread.last_turn_seq = 1
    thread.next_turn_seq = 2
    thread.active_context = ActiveConversationContext(
        slots={
            "product": ContextSlot(value="P-ALPHA", source_message_id=source.message_id),
            "tool_id": ContextSlot(value="ETCH-03", source_message_id=source.message_id),
            "chamber": ContextSlot(
                value="B",
                source_message_id=source.message_id,
                depends_on=["tool_id"],
            ),
            "recipe_id": ContextSlot(
                value="ETCH-ALPHA",
                source_message_id=source.message_id,
                depends_on=["tool_id", "chamber"],
            ),
        },
        evidence_refs=[
            ContextEvidenceRef(
                evidence_id="chunk:test",
                source_type="internal_controlled",
                source_message_id=source.message_id,
                updated_at=now,
            )
        ],
    )
    store.save_thread(thread)

    result = await service.send_message(thread.thread_id, "不是 ETCH-03，是 ETCH-04", scope)
    updated = service.get_thread(thread.thread_id, scope)

    assert result["route_decision"] is AgentRoute.CHAT_DIRECT
    assert updated is not None
    assert updated.active_context.slots["product"].valid is True
    assert updated.active_context.slots["tool_id"].value == "ETCH-04"
    assert updated.active_context.slots["chamber"].valid is False
    assert updated.active_context.slots["recipe_id"].valid is False
    assert updated.active_context.evidence_refs[0].valid is False
    assert result["thread"]["messages"][-1]["presentation"]["mode"] is MessageRenderMode.BUBBLE
    assert retrieval.search_calls == 0
    assert toolbox.query_calls == 0
    assert web.search_calls == 0
    request_id = result["thread"]["messages"][-1]["request_id"]
    record = store.get_message_request(thread.thread_id, scope.user_id, request_id)
    assert record is not None
    assert record.direct_reply_audit is not None
    assert record.direct_reply_audit.reply_kind == "control_ack"


@pytest.mark.asyncio
async def test_chamber_correction_uses_affirmed_value_and_invalidates_dependents(
    seeded_services,
) -> None:
    store, service, retrieval, toolbox, web = _service(seeded_services)
    scope = ActorScope(tool_ids=["ETCH-03"])
    thread = service.create_thread("chamber-correction", scope)
    now = datetime.now(UTC)
    source = ChatMessage(role="user", content="P-ALPHA ETCH-03 Chamber B", turn_seq=1)
    thread.messages.append(source)
    thread.last_turn_seq = 1
    thread.next_turn_seq = 2
    thread.active_context = ActiveConversationContext(
        slots={
            "product": ContextSlot(value="P-ALPHA", source_message_id=source.message_id),
            "tool_id": ContextSlot(value="ETCH-03", source_message_id=source.message_id),
            "chamber": ContextSlot(
                value="B",
                source_message_id=source.message_id,
                depends_on=["tool_id"],
            ),
            "recipe_id": ContextSlot(
                value="ETCH-ALPHA",
                source_message_id=source.message_id,
                depends_on=["tool_id", "chamber"],
            ),
        },
        evidence_refs=[
            ContextEvidenceRef(
                evidence_id="chunk:chamber-test",
                source_type="internal_controlled",
                source_message_id=source.message_id,
                updated_at=now,
            )
        ],
    )
    store.save_thread(thread)

    result = await service.send_message(
        thread.thread_id,
        "不是 Chamber B，是 Chamber A",
        scope,
    )
    updated = service.get_thread(thread.thread_id, scope)

    assert result["route_decision"] is AgentRoute.CHAT_DIRECT
    assert result["interaction_mode"] is InteractionMode.CONTROL
    assert updated is not None
    assert updated.active_context.slots["product"].valid is True
    assert updated.active_context.slots["tool_id"].valid is True
    assert updated.active_context.slots["chamber"].value == "A"
    assert updated.active_context.slots["chamber"].valid is True
    assert updated.active_context.slots["recipe_id"].valid is False
    assert updated.active_context.evidence_refs[0].valid is False
    assert "chamber=A" in result["response"]
    assert "chamber=B" not in result["response"]
    assert result["thread"]["messages"][-1]["presentation"]["mode"] is MessageRenderMode.BUBBLE
    assert retrieval.search_calls == 0
    assert toolbox.query_calls == 0
    assert web.search_calls == 0
    request_id = result["thread"]["messages"][-1]["request_id"]
    record = store.get_message_request(thread.thread_id, scope.user_id, request_id)
    assert record is not None
    assert record.slot_operations[0].slot_name == "chamber"
    assert record.slot_operations[0].value == "A"
    assert record.direct_reply_audit is not None
    assert record.direct_reply_audit.reply_kind == "control_ack"


@pytest.mark.asyncio
async def test_cancel_current_task_skips_all_downstream(seeded_services) -> None:
    store, service, retrieval, toolbox, web = _service(seeded_services)
    scope = ActorScope()
    thread = service.create_thread("cancel", scope)

    result = await service.send_message(thread.thread_id, "别查了", scope)

    assert result["route_decision"] is AgentRoute.CHAT_DIRECT
    assert result["retrieval_skipped_reason"] == "answer_available_without_external_retrieval"
    assert result["thread"]["messages"][-1]["presentation"]["mode"] is MessageRenderMode.BUBBLE
    assert retrieval.search_calls == 0
    assert toolbox.query_calls == 0
    assert web.search_calls == 0
    request_id = result["thread"]["messages"][-1]["request_id"]
    record = store.get_message_request(thread.thread_id, scope.user_id, request_id)
    assert record is not None
    assert record.direct_reply_audit is not None
    assert record.direct_reply_audit.reply_kind == "control_ack"


@pytest.mark.asyncio
async def test_out_of_scope_tool_is_refused_before_downstream(seeded_services) -> None:
    _, service, retrieval, toolbox, web = _service(seeded_services)
    scope = ActorScope(tool_ids=["ETCH-03"])
    thread = service.create_thread("scope", scope)

    result = await service.send_message(
        thread.thread_id,
        "查 P-ALPHA ETCH-04 Chamber B 最近24小时 FDC 报警",
        scope,
    )

    assert result["route_decision"] is AgentRoute.REFUSE
    assert result["citations"] == []
    assert retrieval.search_calls == 0
    assert toolbox.query_calls == 0
    assert web.search_calls == 0


@pytest.mark.asyncio
async def test_empty_history_target_clarifies_without_downstream_calls(seeded_services) -> None:
    store, service, retrieval, toolbox, web = _service(seeded_services)
    scope = ActorScope()
    thread = service.create_thread("empty-history", scope)

    result = await service.send_message(thread.thread_id, "我刚才说什么？", scope)

    assert result["route_decision"] is AgentRoute.CLARIFY
    assert result["missing_fields"] == ["history_reference"]
    assert result["clarification_required"] is True
    assert retrieval.search_calls == 0
    assert toolbox.query_calls == 0
    assert web.search_calls == 0
    request_id = result["thread"]["messages"][-1]["request_id"]
    record = store.get_message_request(thread.thread_id, scope.user_id, request_id)
    assert record is not None
    assert record.direct_reply_audit is not None
    assert record.direct_reply_audit.reply_kind == "clarification"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "utterance",
    [
        "替我完成一个无关的外部任务",
        "帮我写一首关于晚风的诗。",
    ],
)
async def test_generic_out_of_scope_request_is_plain_refusal_without_downstream(
    seeded_services,
    utterance: str,
) -> None:
    store, service, retrieval, toolbox, web = _service(seeded_services)
    scope = ActorScope()
    thread = service.create_thread("generic-boundary", scope)

    result = await service.send_message(thread.thread_id, utterance, scope)

    assert result["route_decision"] is AgentRoute.REFUSE
    assert result["status"] == "refused"
    assert result["answer"] is None
    assert "能力范围" in result["response"]
    assert retrieval.search_calls == 0
    assert toolbox.query_calls == 0
    assert web.search_calls == 0
    request_id = result["thread"]["messages"][-1]["request_id"]
    record = store.get_message_request(thread.thread_id, scope.user_id, request_id)
    assert record is not None
    assert record.direct_reply_audit is not None
    assert record.direct_reply_audit.reply_kind == "refusal"


@pytest.mark.asyncio
async def test_llm_cannot_relabel_generic_out_of_scope_task_as_conversation() -> None:
    settings = Settings(_env_file=None, demo_mode=False)
    understanding_service = ConversationUnderstandingService(
        settings,
        MisclassifiedOutOfScopeLLM(),
    )

    result = await understanding_service.understand(
        "帮我写一首关于晚风的诗。",
        {},
    )

    assert result.understanding.primary_intent is PrimaryIntent.ACTION_REQUEST
    assert result.understanding.task_items[0].target_type is IntentTarget.GENERAL
    assert result.understanding.task_items[0].execution_policy is TaskExecutionDecision.REFUSE
    assert result.understanding.suggested_route is AgentRoute.REFUSE


@pytest.mark.asyncio
async def test_knowledge_base_word_alone_does_not_bypass_capability_boundary() -> None:
    settings = Settings(_env_file=None, demo_mode=False)
    understanding_service = ConversationUnderstandingService(
        settings,
        MisclassifiedOutOfScopeLLM(),
    )

    result = await understanding_service.understand(
        "让知识库帮我写一首关于晚风的诗。",
        {},
    )

    assert result.understanding.primary_intent is PrimaryIntent.ACTION_REQUEST
    assert result.understanding.task_items[0].execution_policy is TaskExecutionDecision.REFUSE
    assert result.understanding.suggested_route is AgentRoute.REFUSE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "utterance",
    [
        "请查询知识库里批准入库的某公开数据集说明。",
        "概括已入库论文中介绍的实验方法。",
        "从内部资料中查找某工艺数据卡的字段定义。",
    ],
)
async def test_generic_governed_corpus_queries_route_to_internal_rag(
    utterance: str,
) -> None:
    settings = Settings(_env_file=None, demo_mode=False)
    service = ConversationUnderstandingService(settings, GovernedCorpusLookupLLM())

    result = await service.understand(utterance, {})
    plan = RoutePolicy().decide(result.understanding, ActorScope(), {}, utterance)

    assert result.understanding.primary_intent is PrimaryIntent.KNOWLEDGE_QUERY
    assert result.understanding.task_items[0].target_type is IntentTarget.GENERAL
    assert result.understanding.task_items[0].action is IntentTaskAction.LOOKUP
    assert result.understanding.suggested_route is AgentRoute.INTERNAL_RAG
    assert plan.route is AgentRoute.INTERNAL_RAG


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "utterance",
    [
        "我刚才问什么了",
        "我刚刚说什么?",
        "我刚刚说了什么?",
        "我刚在说什么?",
        "上一个问题是什么？",
    ],
)
async def test_l0_history_rule_never_calls_llm(utterance: str) -> None:
    service = ConversationUnderstandingService(
        Settings(_env_file=None, demo_mode=False),
        ForbiddenLLM(),
    )
    context = {
        "recent_messages": [
            {"message_id": "msg_1", "role": "user", "content": "当前 SOP 是什么？"},
            {"message_id": "msg_2", "role": "assistant", "content": "回答"},
        ]
    }

    result = await service.understand(utterance, context)

    assert result.understanding.classifier_source == "l0"
    assert result.understanding.suggested_route is AgentRoute.CHAT_DIRECT
    assert result.metadata["understanding_calls"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "utterance",
    [
        "不是 Chamber B，是 Chamber A",
        "不是腔体B，改成腔体A",
        "不是B腔，换成A腔",
    ],
)
async def test_l0_chamber_correction_uses_affirmed_value_without_llm(utterance: str) -> None:
    service = ConversationUnderstandingService(
        Settings(_env_file=None, demo_mode=False),
        ForbiddenLLM(),
    )
    context = {
        "active_context": {
            "slots": {
                "chamber": {
                    "value": "B",
                    "valid": True,
                    "source_message_id": "msg_source",
                }
            }
        }
    }

    result = await service.understand(utterance, context)
    understanding = result.understanding

    assert understanding.classifier_source == "l0"
    assert understanding.interaction_mode is InteractionMode.CONTROL
    assert understanding.suggested_route is AgentRoute.CHAT_DIRECT
    assert understanding.explicit_slots == {"chamber": "A"}
    assert len(understanding.slot_operations) == 1
    assert understanding.slot_operations[0].operation.value == "correct"
    assert understanding.slot_operations[0].slot_name == "chamber"
    assert understanding.slot_operations[0].value == "A"
    assert result.metadata["understanding_calls"] == 0


@pytest.mark.parametrize(
    "utterance,slot_name,expected",
    [
        ("不是 ETCH-03，是 ETCH-04", "tool_id", "ETCH-04"),
        ("不是 P-ALPHA，改成 P-BETA", "product", "P-BETA"),
        ("不是 V2.3，换成 V2.4", "recipe_version", "V2.4"),
        ("不是最近24小时，是最近12小时", "time_range", "最近12小时"),
        ("不是 LOT-A01，是 LOT-B02", "lot_id", "LOT-B02"),
        ("不是 CASE-OLD，是 CASE-NEW", "case_id", "CASE-NEW"),
    ],
)
def test_explicit_slot_extraction_prefers_affirmed_correction_value(
    utterance: str,
    slot_name: str,
    expected: str,
) -> None:
    slots = ConversationUnderstandingService.extract_explicit_slots(utterance)

    assert slots[slot_name] == expected


@pytest.mark.parametrize(
    "utterance,expected",
    [
        ("查询 ETCH-03 Chamber B 当前 SOP", "B"),
        ("查询 ETCH-03 Chamber A1 当前 SOP", "A1"),
        ("查询 ETCH-03 腔体 2 当前 SOP", "2"),
    ],
)
def test_explicit_slot_extraction_accepts_short_chamber_identifiers(
    utterance: str,
    expected: str,
) -> None:
    slots = ConversationUnderstandingService.extract_explicit_slots(utterance)

    assert slots["chamber"] == expected


@pytest.mark.parametrize(
    "utterance",
    [
        "查询受控知识库中 chamber recovery 12 Pa 的要求",
        "查询 chamber pressure alarm 的处置说明",
        "总结 chamber cleaning 后的检查步骤",
    ],
)
def test_explicit_slot_extraction_ignores_chamber_business_phrases(utterance: str) -> None:
    slots = ConversationUnderstandingService.extract_explicit_slots(utterance)

    assert "chamber" not in slots


@pytest.mark.asyncio
async def test_semantic_history_task_overrides_model_route_and_meta_context() -> None:
    request = "复述最近一条用户输入"
    context = {
        "recent_messages": [
            {"message_id": "msg_business", "role": "user", "content": "当前 SOP 怎么要求？"},
            {"message_id": "msg_answer", "role": "assistant", "content": "回答"},
            {"message_id": "msg_meta", "role": "user", "content": "我刚刚说什么?"},
            {"message_id": "msg_meta_answer", "role": "assistant", "content": "上一条用户问题是……"},
        ]
    }
    service = ConversationUnderstandingService(
        Settings(_env_file=None, demo_mode=False),
        MisleadingHistoryLLM(),
    )

    result = await service.understand(request, context)
    plan = RoutePolicy().decide(result.understanding, ActorScope(), context, request)

    assert result.understanding.primary_intent.value == "conversation"
    assert result.understanding.interaction_mode is InteractionMode.CONVERSATION
    assert result.understanding.context_message_ids == ["msg_business"]
    assert result.understanding.suggested_route is AgentRoute.CHAT_DIRECT
    assert plan.route is AgentRoute.CHAT_DIRECT


@pytest.mark.asyncio
async def test_server_policy_keeps_investigation_route_and_rejects_wrong_slot_type() -> None:
    request = "分析 P-ALPHA ETCH-03 最近24小时晶圆图异常"
    service = ConversationUnderstandingService(
        Settings(_env_file=None, demo_mode=False),
        FixedUnderstandingLLM(),
    )

    result = await service.understand(request, {})
    plan = RoutePolicy().decide(result.understanding, ActorScope(), {}, request)

    assert plan.route is AgentRoute.RAG_AND_TOOL
    assert plan.missing_slots == []
    assert result.understanding.explicit_slots["tool_id"] == "ETCH-03"
    assert "chamber" not in result.understanding.explicit_slots


@pytest.mark.asyncio
async def test_llm_cannot_turn_process_term_into_cancel_control() -> None:
    request = "ETCH-03 Chamber B 清腔后首片异常，当前 SOP 怎么要求？"
    service = ConversationUnderstandingService(
        Settings(_env_file=None, demo_mode=False),
        FalseCancelUnderstandingLLM(),
    )

    result = await service.understand(request, {}, clarification_pending=True)

    assert result.understanding.cancel_scope is None
    assert result.understanding.clarification_relation.value == "ambiguous"


@pytest.mark.asyncio
async def test_protected_mixed_tasks_override_unsafe_model_labels() -> None:
    request = "查 P-ALPHA ETCH-03 最近24小时 FDC 报警、修改 Recipe、生成报告"
    service = ConversationUnderstandingService(
        Settings(_env_file=None, demo_mode=False),
        UnsafeMixedLLM(),
    )

    result = await service.understand(request, {})
    plan = RoutePolicy().decide(result.understanding, ActorScope(), {}, request)

    assert len(result.understanding.task_items) == 3
    assert Counter(item.decision for item in plan.task_decisions) == Counter(
        {
            TaskExecutionDecision.EXECUTE: 1,
            TaskExecutionDecision.REFUSE: 1,
            TaskExecutionDecision.DEFER: 1,
        }
    )


def test_unsupported_tool_and_web_combination_defers_incompatible_task() -> None:
    understanding = ConversationUnderstanding(
        classifier_source="deterministic_fallback",
        interaction_mode=InteractionMode.MIXED,
        primary_intent=PrimaryIntent.DATA_QUERY,
        task_items=[
            IntentTaskItem(
                task_id="task_1",
                primary_intent=PrimaryIntent.DATA_QUERY,
                target_type=IntentTarget.FDC,
                action=IntentTaskAction.LOOKUP,
            ),
            IntentTaskItem(
                task_id="task_2",
                primary_intent=PrimaryIntent.KNOWLEDGE_QUERY,
                target_type=IntentTarget.GENERAL,
                action=IntentTaskAction.LOOKUP,
            ),
        ],
        explicit_slots={
            "product": "P-ALPHA",
            "tool_id": "ETCH-03",
            "time_range": "最近24小时",
        },
        standalone_query="查 FDC 并搜索外部公开资料",
        suggested_route=AgentRoute.TOOL_ONLY,
        confidence=0.95,
    )

    plan = RoutePolicy().decide(
        understanding,
        ActorScope(),
        {},
        "查 P-ALPHA ETCH-03 最近24小时 FDC 并搜索外部公开资料",
    )

    assert plan.route is AgentRoute.TOOL_ONLY
    assert [item.decision for item in plan.task_decisions] == [
        TaskExecutionDecision.EXECUTE,
        TaskExecutionDecision.DEFER,
    ]
    assert plan.task_decisions[1].reason_code == "unsupported_route_combination_deferred"


def test_explicit_no_web_instruction_forces_internal_rag() -> None:
    understanding = ConversationUnderstanding(
        classifier_source="llm",
        interaction_mode=InteractionMode.TASK,
        primary_intent=PrimaryIntent.KNOWLEDGE_QUERY,
        task_items=[
            IntentTaskItem(
                task_id="task_1",
                primary_intent=PrimaryIntent.KNOWLEDGE_QUERY,
                target_type=IntentTarget.GENERAL,
                action=IntentTaskAction.LOOKUP,
            )
        ],
        suggested_route=AgentRoute.RAG_AND_WEB,
        confidence=0.95,
    )

    plan = RoutePolicy().decide(
        understanding,
        ActorScope(),
        {},
        "查询内部知识库中的数据集说明，不要使用 Web。",
    )

    assert plan.route is AgentRoute.INTERNAL_RAG
