from __future__ import annotations

import pytest

from semikb.contracts.models import ActorScope


@pytest.mark.asyncio
async def test_thread_clarifies_then_continues_with_evidence(seeded_services) -> None:
    store, _, _, conversation, _ = seeded_services
    thread = conversation.create_thread("Yield investigation", ActorScope())

    first = await conversation.send_message(thread.thread_id, "最近蚀刻良率下降，帮我调查根因")
    assert first["clarification_required"] is True
    assert set(first["missing_fields"]) == {"time_range", "affected_object"}

    second = await conversation.send_message(
        thread.thread_id,
        "P-ALPHA 最近24小时 ETCH-03 Chamber B 的首片异常和 pressure alarm。",
    )
    assert second["trace_id"]
    assert "受控证据" in second["response"]
    persisted = store.get_thread(thread.thread_id)
    assert persisted is not None
    assert len(persisted.messages) == 4
    assert persisted.messages[-1].citations
