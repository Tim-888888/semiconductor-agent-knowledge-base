from __future__ import annotations

import pytest

from semikb.agent_runtime.clarification import adapt_legacy_frame
from semikb.agent_runtime.graph import CLARIFICATION_PROMPTS
from semikb.agent_runtime.service import ConversationService
from semikb.config import Settings
from semikb.contracts.models import (
    ActorScope,
    AgentRoute,
    ClarificationFrame,
    ClarificationFrameStatus,
)
from semikb.contracts.streaming import AgentMessageRequestRecord


async def _start_investigation_clarification(service):
    thread = service.create_thread("clarification transition", ActorScope())
    first = await service.send_message(thread.thread_id, "最近良率下降，原因是什么？")
    assert first["clarification_required"] is True
    assert first["clarification_round"] == 1
    task_text = " ".join(item["message"] for item in first["task_results"])
    assert "product" not in task_text
    assert "time_range" not in task_text
    assert "tool_or_chamber" not in task_text
    return thread, first


@pytest.mark.asyncio
async def test_valid_clarification_answer_continues_original_task(seeded_services) -> None:
    _, _, _, service, _ = seeded_services
    thread, _ = await _start_investigation_clarification(service)

    result = await service.send_message(
        thread.thread_id,
        "P-ALPHA 最近24小时 ETCH-03 Chamber B 出现 pressure alarm。",
    )

    assert result["clarification_required"] is False
    assert result["route_decision"] is AgentRoute.RAG_AND_TOOL
    assert result["trace_id"]


@pytest.mark.asyncio
async def test_explicit_cancel_exits_pending_clarification_without_downstream(
    seeded_services,
) -> None:
    store, _, _, service, _ = seeded_services
    thread, _ = await _start_investigation_clarification(service)
    traces_before = len(store.traces)

    result = await service.send_message(thread.thread_id, "别查了")

    assert result["clarification_required"] is False
    assert result["route_decision"] is AgentRoute.CHAT_DIRECT
    assert result["thread"]["status"] == "active"
    assert result["thread"]["clarification_round"] == 0
    assert result["thread"]["pending_fields"] == []
    assert len(store.traces) == traces_before
    request_id = result["thread"]["messages"][-1]["request_id"]
    record = store.get_message_request(thread.thread_id, thread.actor_scope.user_id, request_id)
    assert record is not None
    assert record.clarification_transition_audit is not None
    assert record.clarification_transition_audit.relation == "cancel_current"


@pytest.mark.asyncio
async def test_complete_new_question_supersedes_pending_clarification(
    seeded_services,
) -> None:
    store, _, _, service, _ = seeded_services
    thread, _ = await _start_investigation_clarification(service)

    result = await service.send_message(
        thread.thread_id,
        "ETCH-03 Chamber B 清腔后首片异常，当前 SOP 怎么要求？",
    )

    assert result["clarification_required"] is False
    assert result["route_decision"] is AgentRoute.INTERNAL_RAG
    assert result["trace_id"]
    request_id = result["thread"]["messages"][-1]["request_id"]
    record = store.get_message_request(thread.thread_id, thread.actor_scope.user_id, request_id)
    assert record is not None
    assert record.clarification_transition_audit is not None
    assert record.clarification_transition_audit.relation == "replace_with_new_request"
    assert record.clarification_transition_audit.pending_before
    assert record.clarification_transition_audit.pending_after == []


@pytest.mark.asyncio
async def test_side_conversation_does_not_consume_business_clarification_round(
    seeded_services,
) -> None:
    store, _, _, service, _ = seeded_services
    thread, first = await _start_investigation_clarification(service)
    traces_before = len(store.traces)

    result = await service.send_message(thread.thread_id, "你能做什么？")

    assert result["route_decision"] is AgentRoute.CHAT_DIRECT
    assert result["clarification_required"] is True
    assert result["clarification_round"] == first["clarification_round"]
    assert result["missing_fields"] == first["missing_fields"]
    assert result["task_results"] == []
    assert "半导体" in result["response"]
    assert len(store.traces) == traces_before
    request_id = result["thread"]["messages"][-1]["request_id"]
    record = store.get_message_request(thread.thread_id, thread.actor_scope.user_id, request_id)
    assert record is not None
    assert record.clarification_transition_audit is not None
    assert record.clarification_transition_audit.relation == "side_conversation"


@pytest.mark.asyncio
async def test_unknown_answer_keeps_bounded_no_progress_behavior(seeded_services) -> None:
    store, _, _, service, _ = seeded_services
    thread, first = await _start_investigation_clarification(service)
    traces_before = len(store.traces)

    second = await service.send_message(thread.thread_id, "暂时不知道")
    third = await service.send_message(thread.thread_id, "还是没有更多信息")

    assert second["clarification_required"] is True
    assert second["clarification_round"] == first["clarification_round"] + 1
    assert third["status"] == "insufficient_information"
    assert "product" not in third["response"]
    assert "time_range" not in third["response"]
    assert "tool_or_chamber" not in third["response"]
    assert len(store.traces) == traces_before


@pytest.mark.asyncio
async def test_partial_answer_resolves_only_matching_pending_items(seeded_services) -> None:
    _, _, _, service, _ = seeded_services
    thread, _ = await _start_investigation_clarification(service)
    config = {"configurable": {"thread_id": thread.thread_id}}
    first_frame = ClarificationFrame.model_validate(
        service.graph.compiled.get_state(config).values["clarification_frame"]
    )

    second = await service.send_message(thread.thread_id, "P-ALPHA")
    second_frame = ClarificationFrame.model_validate(
        service.graph.compiled.get_state(config).values["clarification_frame"]
    )

    assert second["clarification_required"] is True
    assert second["clarification_round"] == 2
    assert second["missing_fields"] == ["time_range", "tool_or_chamber"]
    assert [item.key for item in second_frame.resolved_items] == ["product"]
    assert second_frame.signature != first_frame.signature


@pytest.mark.asyncio
async def test_two_round_answers_accumulate_verified_slots(seeded_services) -> None:
    _, _, _, service, _ = seeded_services
    thread, _ = await _start_investigation_clarification(service)

    partial = await service.send_message(thread.thread_id, "P-ALPHA")
    completed = await service.send_message(
        thread.thread_id,
        "最近24小时，ETCH-03 Chamber B。",
    )

    assert partial["missing_fields"] == ["time_range", "tool_or_chamber"]
    assert completed["clarification_required"] is False
    assert completed["route_decision"] is AgentRoute.RAG_AND_TOOL
    assert completed["trace_id"]


@pytest.mark.asyncio
async def test_no_progress_changes_signature_before_bounded_stop(seeded_services) -> None:
    _, _, _, service, _ = seeded_services
    thread, _ = await _start_investigation_clarification(service)
    config = {"configurable": {"thread_id": thread.thread_id}}
    first_frame = ClarificationFrame.model_validate(
        service.graph.compiled.get_state(config).values["clarification_frame"]
    )

    second = await service.send_message(thread.thread_id, "暂时不知道")
    second_frame = ClarificationFrame.model_validate(
        service.graph.compiled.get_state(config).values["clarification_frame"]
    )

    assert second["clarification_required"] is True
    assert second_frame.no_progress_count == 1
    assert second_frame.signature != first_frame.signature
    assert "暂时无法确认" in second["response"]


@pytest.mark.asyncio
async def test_side_conversation_then_new_task_can_still_supersede_frame(
    seeded_services,
) -> None:
    store, _, _, service, _ = seeded_services
    thread, _ = await _start_investigation_clarification(service)

    side = await service.send_message(thread.thread_id, "你能做什么？")
    replacement = await service.send_message(
        thread.thread_id,
        "CMP-01 抛光后表面刮伤时，当前 SOP 怎么处理？",
    )

    assert side["clarification_required"] is True
    assert side["clarification_round"] == 1
    assert replacement["clarification_required"] is False
    assert replacement["route_decision"] is AgentRoute.INTERNAL_RAG
    request_id = replacement["thread"]["messages"][-1]["request_id"]
    record = store.get_message_request(thread.thread_id, thread.actor_scope.user_id, request_id)
    assert record is not None
    assert record.clarification_transition_audit is not None
    assert record.clarification_transition_audit.next_status is ClarificationFrameStatus.SUPERSEDED


@pytest.mark.asyncio
async def test_explicit_resume_after_side_conversation_does_not_consume_round(
    seeded_services,
) -> None:
    _, _, _, service, _ = seeded_services
    thread, first = await _start_investigation_clarification(service)

    await service.send_message(thread.thread_id, "你能做什么？")
    resumed = await service.send_message(thread.thread_id, "继续刚才")
    completed = await service.send_message(
        thread.thread_id,
        "P-ALPHA 最近24小时 ETCH-03 Chamber B 出现 pressure alarm。",
    )

    assert resumed["clarification_required"] is True
    assert resumed["clarification_round"] == first["clarification_round"]
    assert resumed["missing_fields"] == first["missing_fields"]
    assert "继续当前问题" in resumed["response"]
    assert completed["clarification_required"] is False
    assert completed["route_decision"] is AgentRoute.RAG_AND_TOOL


@pytest.mark.asyncio
async def test_switch_without_a_new_question_asks_relation_without_consuming_round(
    seeded_services,
) -> None:
    _, _, _, service, _ = seeded_services
    thread, _ = await _start_investigation_clarification(service)

    result = await service.send_message(thread.thread_id, "我想换个问题")

    assert result["clarification_required"] is True
    assert result["clarification_round"] == 1
    assert "继续" in result["response"]
    assert "新问题" in result["response"]


@pytest.mark.asyncio
async def test_dangerous_new_task_is_not_hidden_inside_pending_clarification(
    seeded_services,
) -> None:
    store, _, _, service, _ = seeded_services
    thread, _ = await _start_investigation_clarification(service)

    result = await service.send_message(
        thread.thread_id,
        "把 Recipe V2.3 下发到 ETCH-03。",
    )

    assert result["clarification_required"] is False
    assert result["route_decision"] is AgentRoute.REFUSE
    assert result["status"] == "refused"
    assert result["trace_id"] is None
    request_id = result["thread"]["messages"][-1]["request_id"]
    record = store.get_message_request(thread.thread_id, thread.actor_scope.user_id, request_id)
    assert record is not None
    assert record.clarification_transition_audit is not None
    assert record.clarification_transition_audit.relation == "replace_with_new_request"


def test_legacy_request_record_without_transition_audit_remains_readable() -> None:
    record = AgentMessageRequestRecord.model_validate(
        {
            "request_id": "request-legacy-001",
            "thread_id": "thread_legacy",
            "actor_user_id": "engineer",
            "content_sha256": "0" * 64,
            "user_message_id": "msg_legacy",
            "run_id": "run_legacy",
        }
    )

    assert record.clarification_transition_audit is None


def test_legacy_checkpoint_without_frame_is_adapted_to_versioned_frame() -> None:
    state = {
        "request": "最近良率下降，原因是什么？",
        "clarification_frame": {},
        "missing_required_fields": ["product", "time_range", "tool_or_chamber"],
        "understanding": {
            "classifier_source": "deterministic_fallback",
            "interaction_mode": "task",
            "primary_intent": "investigation",
            "task_items": [
                {
                    "task_id": "task_1",
                    "primary_intent": "investigation",
                    "target_type": "case",
                    "action": "diagnose",
                    "execution_policy": "clarify",
                }
            ],
            "explicit_slots": {},
            "inherited_slots": {},
            "missing_slots": ["product", "time_range", "tool_or_chamber"],
            "standalone_query": "最近良率下降，原因是什么？",
            "suggested_route": "rag_and_tool",
            "confidence": 0.8,
        },
        "route_plan": {
            "route": "clarify",
            "confidence": 0.8,
            "reason_codes": ["missing_required_slots"],
            "missing_slots": ["product", "time_range", "tool_or_chamber"],
            "task_decisions": [],
        },
    }

    frame = adapt_legacy_frame(state=state, prompts=CLARIFICATION_PROMPTS)

    assert frame.schema_version == "semikb-clarification-frame-v1"
    assert frame.original_request == state["request"]
    assert frame.pending_keys == state["missing_required_fields"]


@pytest.mark.asyncio
async def test_feature_flag_can_roll_back_to_legacy_clarification(seeded_services) -> None:
    store, _, retrieval, _, _ = seeded_services
    service = ConversationService(
        store,
        retrieval,
        Settings(
            _env_file=None,
            demo_mode=True,
            agent_clarification_frame_v1_enabled=False,
        ),
    )
    thread = service.create_thread("legacy clarification", ActorScope())

    first = await service.send_message(thread.thread_id, "最近良率下降，原因是什么？")
    completed = await service.send_message(
        thread.thread_id,
        "P-ALPHA 最近24小时 ETCH-03 Chamber B 出现 pressure alarm。",
    )

    assert first["clarification_required"] is True
    assert completed["clarification_required"] is False
    assert completed["route_decision"] is AgentRoute.RAG_AND_TOOL
