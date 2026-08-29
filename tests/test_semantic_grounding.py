from __future__ import annotations

import json

import pytest

from semikb.agent_runtime.evidence_sufficiency import EvidenceSufficiencyService
from semikb.agent_runtime.llm_gateway import LLMCompletion, OpenAICompatibleLLMGateway
from semikb.agent_runtime.routing import RoutePolicy
from semikb.agent_runtime.understanding import ConversationUnderstandingService
from semikb.config import Settings
from semikb.contracts.models import (
    ActorScope,
    AgentRoute,
    ApprovalStatus,
    Chunk,
    ConversationUnderstanding,
    EvidenceSufficiencyStatus,
    ExpectedOutput,
    IntentTarget,
    KnowledgeScope,
    PrimaryIntent,
    RetrievalCandidate,
    RetrievalTrace,
    TaskShape,
)

EMPTY_CONTEXT = {
    "current_message_id": "msg_current",
    "recent_messages": [],
    "active_context": {"slots": {}, "evidence_refs": []},
}


class BadThreeTaskPlanner:
    async def complete(self, messages, **kwargs):
        request = json.loads(messages[-1]["content"])["current_request"]
        payload = {
            "interaction_mode": "task",
            "primary_intent": "data_query",
            "task_items": [
                {
                    "task_id": f"task_{index}",
                    "primary_intent": "data_query",
                    "target_type": target,
                    "action": "lookup",
                    "task_shape": "entity_lookup",
                    "group_by": [],
                    "supporting_spans": [{"quote": request, "message_id": None}],
                    "depends_on": [],
                    "execution_policy": "execute",
                }
                for index, target in enumerate(("spc", "fdc", "lot"), start=1)
            ],
            "semantic_frame": {
                "temporal_scope": "current",
                "expected_output": "records",
                "knowledge_scope": "not_applicable",
            },
            "affect": {
                "sentiment": "neutral",
                "urgency": "normal",
                "complaint_signal": False,
            },
            "slot_operations": [],
            "explicit_slots": [],
            "inherited_slot_names": [],
            "missing_slots": ["product", "time_range", "tool_id"],
            "context_message_ids": [],
            "standalone_query": "semikb-intent-catalog-v5",
            "cancel_scope": None,
            "clarification_relation": None,
            "suggested_route": "tool_only",
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


class ForbiddenLLM:
    async def complete(self, *args, **kwargs):
        raise AssertionError("deterministic evidence cases must not invoke the LLM judge")


async def _understand(request: str):
    settings = Settings(_env_file=None, demo_mode=True)
    gateway = OpenAICompatibleLLMGateway(settings)
    service = ConversationUnderstandingService(settings, gateway)
    return (await service.understand(request, EMPTY_CONTEXT)).understanding


@pytest.mark.asyncio
async def test_general_process_question_is_knowledge_without_manufacturing_slots() -> None:
    request = "半导体检测流程一般包括什么？"
    understanding = await _understand(request)
    plan = RoutePolicy().decide(understanding, ActorScope(), EMPTY_CONTEXT, request)

    assert understanding.primary_intent is PrimaryIntent.KNOWLEDGE_QUERY
    assert understanding.semantic_frame.expected_output is ExpectedOutput.ENUMERATION
    assert understanding.semantic_frame.knowledge_scope is KnowledgeScope.PUBLIC_GENERAL
    assert len(understanding.task_items) == 1
    assert understanding.task_items[0].task_shape is TaskShape.CONCEPT_EXPLANATION
    assert plan.route is AgentRoute.INTERNAL_RAG
    assert plan.missing_slots == []


@pytest.mark.asyncio
async def test_recent_product_yield_ranking_treats_product_as_output_dimension() -> None:
    request = "最近有哪些产品良率低？"
    understanding = await _understand(request)
    plan = RoutePolicy().decide(understanding, ActorScope(), EMPTY_CONTEXT, request)

    assert understanding.primary_intent is PrimaryIntent.DATA_QUERY
    assert understanding.task_items[0].task_shape is TaskShape.AGGREGATE_RANKING
    assert [item.value for item in understanding.task_items[0].group_by] == ["product"]
    assert plan.route is AgentRoute.CLARIFY
    assert plan.missing_slots == ["time_range"]


@pytest.mark.asyncio
async def test_bounded_product_yield_ranking_needs_no_single_product_or_tool() -> None:
    request = "最近24小时有哪些产品良率低？"
    understanding = await _understand(request)
    plan = RoutePolicy().decide(understanding, ActorScope(), EMPTY_CONTEXT, request)

    assert plan.route is AgentRoute.TOOL_ONLY
    assert plan.missing_slots == []
    assert understanding.explicit_slots == {"time_range": "最近24小时"}


@pytest.mark.asyncio
async def test_general_yield_causes_are_knowledge_but_entity_cause_is_investigation() -> None:
    general = await _understand("良率低一般有哪些原因？")
    concrete = await _understand("P-ALPHA 最近24小时为什么良率低？")

    assert general.primary_intent is PrimaryIntent.KNOWLEDGE_QUERY
    assert general.task_items == [
        general.task_items[0].model_copy(
            update={
                "primary_intent": PrimaryIntent.KNOWLEDGE_QUERY,
                "target_type": IntentTarget.GENERAL,
                "task_shape": TaskShape.CONCEPT_EXPLANATION,
            }
        )
    ]
    assert concrete.primary_intent is PrimaryIntent.INVESTIGATION
    assert concrete.task_items[-1].task_shape is TaskShape.CAUSAL_INVESTIGATION


@pytest.mark.asyncio
async def test_server_grounding_collapses_bad_three_task_plan_and_rejects_query_pollution() -> None:
    request = "半导体检测流程一般包括什么？"
    settings = Settings(_env_file=None, demo_mode=False)
    service = ConversationUnderstandingService(settings, BadThreeTaskPlanner())
    result = await service.understand(request, EMPTY_CONTEXT)
    plan = RoutePolicy().decide(result.understanding, ActorScope(), EMPTY_CONTEXT, request)

    assert result.understanding.primary_intent is PrimaryIntent.KNOWLEDGE_QUERY
    assert len(result.understanding.task_items) == 1
    assert result.understanding.task_items[0].target_type is IntentTarget.GENERAL
    assert result.understanding.standalone_query == request
    assert plan.route is AgentRoute.INTERNAL_RAG
    assert plan.missing_slots == []


def _chunk(text: str, chunk_id: str = "C1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="DOC-1",
        revision="R1",
        chunk_text=text,
        title_path=["公开知识"],
        page_or_section="section-1",
        approval_status=ApprovalStatus.APPROVED,
    )


@pytest.mark.asyncio
async def test_public_empty_evidence_allows_web_but_internal_sop_does_not() -> None:
    settings = Settings(_env_file=None, demo_mode=True)
    service = EvidenceSufficiencyService(settings, ForbiddenLLM())
    public = ConversationUnderstanding(
        classifier_source="deterministic_fallback",
        interaction_mode="task",
        primary_intent=PrimaryIntent.KNOWLEDGE_QUERY,
        semantic_frame={"knowledge_scope": "public_general"},
        suggested_route="internal_rag",
        confidence=0.9,
    )
    internal = ConversationUnderstanding(
        classifier_source="deterministic_fallback",
        interaction_mode="task",
        primary_intent=PrimaryIntent.KNOWLEDGE_QUERY,
        task_items=[
            {
                "task_id": "task_1",
                "primary_intent": "knowledge_query",
                "target_type": "sop",
                "action": "lookup",
            }
        ],
        semantic_frame={"knowledge_scope": "internal_controlled"},
        suggested_route="internal_rag",
        confidence=0.9,
    )

    public_result = await service.assess(
        query="公开检测原理",
        understanding=public,
        evidence=[],
        trace=None,
        initial_route=AgentRoute.INTERNAL_RAG,
    )
    internal_result = await service.assess(
        query="当前 SOP",
        understanding=internal,
        evidence=[],
        trace=None,
        initial_route=AgentRoute.INTERNAL_RAG,
    )

    assert public_result.status is EvidenceSufficiencyStatus.INSUFFICIENT
    assert public_result.web_fallback_allowed is True
    assert internal_result.status is EvidenceSufficiencyStatus.INSUFFICIENT
    assert internal_result.web_fallback_allowed is False


@pytest.mark.asyncio
async def test_high_coverage_internal_result_is_sufficient_without_web() -> None:
    settings = Settings(_env_file=None, demo_mode=True)
    service = EvidenceSufficiencyService(settings, ForbiddenLLM())
    understanding = ConversationUnderstanding(
        classifier_source="deterministic_fallback",
        interaction_mode="task",
        primary_intent=PrimaryIntent.KNOWLEDGE_QUERY,
        semantic_frame={
            "knowledge_scope": "public_general",
            "expected_output": "explanation",
        },
        suggested_route="internal_rag",
        confidence=0.9,
    )
    chunk = _chunk("晶圆缺陷检测流程包含图像采集、缺陷分类与复核。")
    candidate = RetrievalCandidate(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        revision=chunk.revision,
        title="公开知识",
        page_or_section=chunk.page_or_section,
        routes=["dense", "sparse"],
        dense_score=0.8,
        sparse_score=0.7,
        rrf_score=0.03,
        rerank_score=0.81,
        selected=True,
    )
    trace = RetrievalTrace(
        actor_user_id="demo_engineer",
        original_query="晶圆缺陷检测流程包含什么",
        candidates=[candidate],
        final_evidence_ids=[chunk.chunk_id],
    )

    result = await service.assess(
        query="晶圆缺陷检测流程包含什么",
        understanding=understanding,
        evidence=[chunk],
        trace=trace,
        initial_route=AgentRoute.INTERNAL_RAG,
    )

    assert result.status is EvidenceSufficiencyStatus.SUFFICIENT
    assert result.web_fallback_allowed is False
