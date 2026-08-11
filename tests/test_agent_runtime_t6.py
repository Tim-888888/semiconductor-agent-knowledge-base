from __future__ import annotations

import pytest
from pydantic import ValidationError

from semikb.agent_runtime.graph import ConversationGraph
from semikb.agent_runtime.service import ConversationService
from semikb.agent_runtime.tools import ToolArguments
from semikb.config import Settings
from semikb.contracts.models import ActorScope, CreateMemoryRequest


@pytest.mark.asyncio
async def test_interrupted_graph_resumes_after_service_recreation(seeded_services) -> None:
    store, _, retrieval, service, _ = seeded_services
    thread = service.create_thread("Persistent investigation", ActorScope())

    first = await service.send_message(thread.thread_id, "最近蚀刻良率下降，帮我调查根因")
    assert first["clarification_required"] is True

    restarted = ConversationService(
        store,
        retrieval,
        Settings(_env_file=None, demo_mode=True),
        checkpointer=service.checkpointer,
        long_term_store=service.long_term_store,
    )
    second = await restarted.send_message(
        thread.thread_id,
        "P-ALPHA 最近24小时 ETCH-03 Chamber B 出现 pressure alarm。",
    )

    assert second["clarification_required"] is False
    assert second["trace_id"]
    assert second["evidence_ledger"]
    assert all(
        citation["evidence_id"]
        in {item["evidence_id"] for item in second["evidence_ledger"]}
        for citation in second["citations"]
    )


@pytest.mark.asyncio
async def test_two_clarification_rounds_stop_without_retrieval(seeded_services) -> None:
    store, _, _, service, _ = seeded_services
    thread = service.create_thread("Bounded clarification", ActorScope())

    first = await service.send_message(thread.thread_id, "最近良率下降，原因是什么？")
    second = await service.send_message(thread.thread_id, "暂时不知道")
    third = await service.send_message(thread.thread_id, "还是没有更多信息")

    assert first["clarification_round"] == 1
    assert second["clarification_round"] == 2
    assert third["status"] == "insufficient_information"
    assert third["trace_id"] is None
    assert store.traces == {}


@pytest.mark.asyncio
async def test_out_of_scope_constraints_do_not_retrieve(seeded_services) -> None:
    store, _, _, service, _ = seeded_services
    actor = ActorScope(products=["P-ALPHA"], tool_ids=["ETCH-03"])
    thread = service.create_thread("Authorization", actor)

    result = await service.send_message(
        thread.thread_id,
        "P-BETA 最近24小时 ETCH-03 Chamber B 良率异常。",
        actor,
    )

    assert result["status"] == "insufficient_information"
    assert "超出当前用户权限" in result["response"]
    assert result["trace_id"] is None
    assert store.traces == {}


def test_tool_arguments_reject_missing_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ToolArguments.model_validate(
            {
                "product": "P-ALPHA",
                "tool_id": "ETCH-03",
                "time_range": "最近24小时",
                "sql": "select * from secret_table",
            }
        )
    with pytest.raises(ValidationError):
        ToolArguments.model_validate({"product": "P-ALPHA", "tool_id": "ETCH-03"})


def test_long_term_memory_is_explicit_approved_and_user_scoped(seeded_services) -> None:
    _, _, _, service, _ = seeded_services
    actor = ActorScope(user_id="memory_owner")
    other = ActorScope(user_id="other_user")

    record = service.memory.create(
        CreateMemoryRequest(memory_type="preference", content="回答时先列出证据再给建议。"),
        actor,
    )

    assert record.approval_status == "approved"
    assert [item.memory_id for item in service.memory.list(actor)] == [record.memory_id]
    assert service.memory.list(other) == []
    with pytest.raises(PermissionError):
        service.memory.create(
            CreateMemoryRequest(
                memory_type="stable_rule",
                content="未经治理的工艺规则不得写入长期记忆。",
            ),
            actor,
        )


@pytest.mark.asyncio
async def test_thread_owner_is_enforced_before_resume(seeded_services) -> None:
    _, _, _, service, _ = seeded_services
    owner = ActorScope(user_id="owner")
    outsider = ActorScope(user_id="outsider")
    thread = service.create_thread("Owned thread", owner)
    await service.send_message(thread.thread_id, "最近良率下降，帮我调查", owner)

    with pytest.raises(KeyError):
        await service.send_message(thread.thread_id, "P-ALPHA 最近24小时 ETCH-03", outsider)


def test_verifier_rejects_external_only_fact_when_internal_evidence_exists() -> None:
    result = ConversationGraph._verify_answer(
        {
            "evidence_ledger": [
                {
                    "evidence_id": "chunk:C1",
                    "source_type": "internal_controlled",
                    "content": "现行 SOP 要求先检查 chamber pressure。",
                    "chunk_id": "C1",
                    "document_id": "SOP-1",
                },
                {
                    "evidence_id": "external:E1",
                    "source_type": "external",
                    "content": "外部网页给出了不同建议。",
                    "external_url": "https://example.com/a",
                },
            ],
            "answer": {
                "facts": [
                    {"text": "内部要求", "citation_ids": ["C1"]},
                    {"text": "外部说法覆盖内部", "citation_ids": ["external:E1"]},
                ],
                "hypotheses": [],
                "unknowns": [],
                "next_actions": [],
                "confidence": "medium",
            },
        }
    )

    assert [item["text"] for item in result["answer"]["facts"]] == ["内部要求"]
    assert result["answer"]["facts"][0]["citation_ids"] == ["chunk:C1"]
    assert "external_only_fact_removed_when_internal_evidence_exists" in result[
        "verification_warnings"
    ]
