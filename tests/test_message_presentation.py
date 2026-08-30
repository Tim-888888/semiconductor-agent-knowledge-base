from __future__ import annotations

import pytest

from semikb.agent_runtime.presentation import build_message_presentation
from semikb.contracts.models import (
    AgentAnswer,
    AgentRoute,
    AnswerClaim,
    AnswerMode,
    EvidenceLedgerEntry,
    MessageRenderMode,
)
from semikb.demo_factory import demo_actor_scope


def _answer() -> AgentAnswer:
    return AgentAnswer(
        facts=[AnswerClaim(text="受控事实", citation_ids=["chunk:SOP-001"])],
        next_actions=["复核设备状态"],
        confidence="high",
    )


@pytest.mark.parametrize(
    "route",
    [
        AgentRoute.REUSE_EVIDENCE,
        AgentRoute.INTERNAL_RAG,
        AgentRoute.TOOL_ONLY,
        AgentRoute.RAG_AND_TOOL,
        AgentRoute.RAG_AND_WEB,
    ],
)
def test_investigation_routes_use_structured_cards(route: AgentRoute) -> None:
    presentation = build_message_presentation(
        route=route,
        answer=_answer(),
        status="completed",
        trace_id="trace_1",
        verification_warnings=["warning"],
        image_asset_ids=["IMG-2", "IMG-1", "IMG-2"],
        evidence_ledger=[
            EvidenceLedgerEntry(
                evidence_id="chunk:SOP-001",
                source_type="internal_controlled",
                content="受控事实",
                document_id="SOP-001",
            ),
            {
                "evidence_id": "chunk:SOP-001",
                "source_type": "internal_controlled",
                "content": "重复证据",
            },
        ],
    )

    assert presentation.mode is MessageRenderMode.STRUCTURED_CARD
    assert presentation.route_decision == route.value
    assert presentation.answer == _answer()
    assert presentation.trace_id == "trace_1"
    assert presentation.verification_warnings == ["warning"]
    assert presentation.image_asset_ids == ["IMG-2", "IMG-1"]
    assert [item.evidence_id for item in presentation.evidence_ledger] == ["chunk:SOP-001"]


@pytest.mark.parametrize(
    "route",
    [
        AgentRoute.HISTORY_DIRECT,
        AgentRoute.CHAT_DIRECT,
        AgentRoute.CLARIFY,
        AgentRoute.REFUSE,
    ],
)
def test_direct_routes_use_bubbles_even_if_an_answer_is_supplied(route: AgentRoute) -> None:
    presentation = build_message_presentation(
        route=route,
        answer=_answer(),
        status="completed",
        trace_id="trace_should_not_leak",
        verification_warnings=["should_not_render"],
        image_asset_ids=["IMG-should-not-leak"],
        evidence_ledger=[
            {
                "evidence_id": "chunk:should-not-leak",
                "source_type": "internal_controlled",
                "content": "should not leak",
            }
        ],
    )

    assert presentation.mode is MessageRenderMode.BUBBLE
    assert presentation.route_decision == route.value
    assert presentation.answer is None
    assert presentation.trace_id is None
    assert presentation.verification_warnings == []
    assert presentation.image_asset_ids == []
    assert presentation.evidence_ledger == []


def test_public_knowledge_uses_bubble_but_keeps_trace_and_sources() -> None:
    presentation = build_message_presentation(
        route=AgentRoute.RAG_AND_WEB,
        answer=_answer(),
        answer_mode=AnswerMode.NATURAL_KNOWLEDGE,
        status="completed",
        trace_id="trace_public",
        image_asset_ids=["IMG-public"],
        evidence_ledger=[
            EvidenceLedgerEntry(
                evidence_id="external:1",
                source_type="external",
                content="公开知识",
                external_url="https://example.org/source",
                source_title="公开来源",
            )
        ],
    )

    assert presentation.mode is MessageRenderMode.BUBBLE
    assert presentation.answer_mode is AnswerMode.NATURAL_KNOWLEDGE
    assert presentation.answer is None
    assert presentation.trace_id == "trace_public"
    assert presentation.image_asset_ids == ["IMG-public"]
    assert presentation.evidence_ledger[0].source_title == "公开来源"


@pytest.mark.asyncio
async def test_structured_message_survives_later_direct_reply_and_thread_reload(
    seeded_services,
) -> None:
    store, _, _, service, _ = seeded_services
    scope = demo_actor_scope()
    thread = service.create_thread("presentation history", scope)

    investigation = await service.send_message(
        thread.thread_id,
        "当前 ETCH-03 SOP 怎么要求？",
        scope,
    )
    first_assistant = investigation["thread"]["messages"][-1]
    assert first_assistant["presentation"]["mode"] == "structured_card"
    assert first_assistant["presentation"]["route_decision"] == "internal_rag"
    assert first_assistant["presentation"]["answer"]
    assert first_assistant["presentation"]["evidence_ledger"]

    legacy_thread = store.get_thread(thread.thread_id)
    assert legacy_thread is not None
    legacy_assistant = legacy_thread.messages[-1]
    legacy_assistant.presentation = None

    direct = await service.send_message(thread.thread_id, "你好", scope)
    messages = direct["thread"]["messages"]
    assert messages[-1]["presentation"]["mode"] == "bubble"
    assert messages[-1]["presentation"]["route_decision"] == "chat_direct"
    assert messages[-3]["presentation"]["mode"] == "structured_card"

    persisted = store.get_thread(thread.thread_id)
    assert persisted is not None
    prior_assistant = persisted.messages[-3]
    assert prior_assistant.presentation is None

    restored = service.get_thread(thread.thread_id, scope)
    assert restored is not None
    restored_prior = restored.messages[-3]
    assert restored_prior.presentation is not None
    assert restored_prior.presentation.mode is MessageRenderMode.STRUCTURED_CARD
    assert restored_prior.presentation.answer is not None
    assert restored_prior.presentation.evidence_ledger
    assert prior_assistant.presentation is None


@pytest.mark.asyncio
async def test_final_image_order_survives_thread_reload(seeded_services) -> None:
    store, _, _, service, _ = seeded_services
    scope = demo_actor_scope()
    thread = service.create_thread("image presentation", scope)

    response = await service.send_message(
        thread.thread_id,
        "有没有 ETCH-03 Chamber B 的边缘环状缺陷晶圆图？",
        scope,
    )

    assert response["image_asset_ids"]
    latest = response["thread"]["messages"][-1]
    assert latest["presentation"]["image_asset_ids"] == response["image_asset_ids"]

    persisted = store.get_thread(thread.thread_id)
    assert persisted is not None
    assert persisted.messages[-1].presentation is not None
    persisted.messages[-1].presentation.evidence_ledger = []

    restored = service.get_thread(thread.thread_id, scope)
    assert restored is not None
    assert restored.messages[-1].presentation is not None
    assert (
        restored.messages[-1].presentation.image_asset_ids
        == response["image_asset_ids"]
    )
    assert restored.messages[-1].presentation.evidence_ledger
