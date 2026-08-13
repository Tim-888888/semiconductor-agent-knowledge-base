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
    InteractionMode,
    TaskExecutionDecision,
)


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
    scope = ActorScope()
    thread = service.create_thread("history", scope)
    first_question = "ETCH-03 Chamber B 清腔后首片异常，当前 SOP 怎么要求？"

    first = await service.send_message(thread.thread_id, first_question, scope)
    search_calls = retrieval.search_calls
    tool_calls = toolbox.query_calls
    result = await service.send_message(thread.thread_id, "我刚才问什么了？", scope)

    assert first["route_decision"] is AgentRoute.INTERNAL_RAG
    assert result["route_decision"] is AgentRoute.HISTORY_DIRECT
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
        assert result["route_decision"] is AgentRoute.HISTORY_DIRECT
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
    _, service, retrieval, toolbox, web = _service(seeded_services)
    thread = service.create_thread("feedback", ActorScope())

    result = await service.send_message(thread.thread_id, "回答太复杂了")

    assert result["interaction_mode"] is InteractionMode.FEEDBACK
    assert result["route_decision"] is AgentRoute.CHAT_DIRECT
    assert retrieval.search_calls == 0
    assert toolbox.query_calls == 0
    assert web.search_calls == 0


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
    assert retrieval.search_calls == 0
    assert toolbox.query_calls == 1


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
    assert retrieval.search_calls == 0
    assert toolbox.query_calls == 0
    assert web.search_calls == 0


@pytest.mark.asyncio
async def test_cancel_current_task_skips_all_downstream(seeded_services) -> None:
    _, service, retrieval, toolbox, web = _service(seeded_services)
    thread = service.create_thread("cancel", ActorScope())

    result = await service.send_message(thread.thread_id, "别查了")

    assert result["route_decision"] is AgentRoute.CHAT_DIRECT
    assert result["retrieval_skipped_reason"] == "answer_available_without_external_retrieval"
    assert retrieval.search_calls == 0
    assert toolbox.query_calls == 0
    assert web.search_calls == 0


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
    assert result.understanding.suggested_route is AgentRoute.HISTORY_DIRECT
    assert result.metadata["understanding_calls"] == 0


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
    assert result.understanding.suggested_route is AgentRoute.HISTORY_DIRECT
    assert plan.route is AgentRoute.HISTORY_DIRECT


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
