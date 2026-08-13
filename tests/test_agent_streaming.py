from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from semikb.agent_runtime.llm_gateway import LLMCompletion
from semikb.agent_runtime.service import ConversationService
from semikb.api.main import app
from semikb.bootstrap import get_container
from semikb.config import Settings, get_settings
from semikb.contracts.models import ActorScope
from semikb.contracts.streaming import (
    AgentMessageRequestStatus,
    AgentStreamEventType,
    agent_stream_event_adapter,
    validate_stream_event_sequence,
)
from semikb.storage.conversations import MessageRequestInProgressError


@pytest.fixture(autouse=True)
def isolate_stream_api(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    get_settings.cache_clear()
    get_container.cache_clear()
    yield
    get_container.cache_clear()
    get_settings.cache_clear()


def _authenticated_client() -> tuple[TestClient, dict[str, str]]:
    get_container.cache_clear()
    client = TestClient(app)
    token = client.post(
        "/api/v1/auth/demo-token",
        json={
            "user_id": "stream_test_engineer",
            "roles": ["engineer"],
            "access_scope_keys": ["demo_engineering"],
            "fabs": ["FAB-01"],
            "products": ["P-ALPHA"],
            "tool_ids": ["ETCH-03"],
        },
    ).json()["access_token"]
    return client, {"Authorization": f"Bearer {token}"}


def _stream_events(
    client: TestClient,
    headers: dict[str, str],
    thread_id: str,
    request_id: str,
    content: str,
):
    with client.stream(
        "POST",
        f"/api/v1/threads/{thread_id}/messages/stream",
        json={"content": content, "request_id": request_id},
        headers=headers,
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        payloads = [
            json.loads(line[6:])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]
    return [agent_stream_event_adapter.validate_python(item) for item in payloads]


def test_stream_api_emits_ordered_events_and_persists_before_completed() -> None:
    client, headers = _authenticated_client()
    thread_id = client.post(
        "/api/v1/threads",
        json={"title": "stream test"},
        headers=headers,
    ).json()["thread_id"]

    events = _stream_events(
        client,
        headers,
        thread_id,
        "req_stream_api_001",
        "ETCH-03 Chamber B 清腔后首片异常，当前 SOP 怎么要求？",
    )

    validate_stream_event_sequence(events)
    assert events[0].event is AgentStreamEventType.ACCEPTED
    assert any(event.event is AgentStreamEventType.EVIDENCE for event in events)
    assert any(event.event is AgentStreamEventType.ANSWER_DELTA for event in events)
    completed = events[-1]
    assert completed.event is AgentStreamEventType.COMPLETED
    persisted = client.get(f"/api/v1/threads/{thread_id}", headers=headers).json()
    assert persisted["messages"][-1]["content"] == completed.data.result.response
    assert persisted["messages"][-1]["request_id"] == "req_stream_api_001"


def test_stream_api_validates_authentication_and_thread_before_sse_headers() -> None:
    client, headers = _authenticated_client()
    payload = {
        "content": "ETCH-03 Chamber B 清腔后首片异常，当前 SOP 怎么要求？",
        "request_id": "req_stream_preflight_001",
    }

    invalid_authentication = client.post(
        "/api/v1/threads/thread_missing/messages/stream",
        json=payload,
        headers={"Authorization": "Bearer invalid-token"},
    )
    missing_thread = client.post(
        "/api/v1/threads/thread_missing/messages/stream",
        json=payload,
        headers=headers,
    )

    assert invalid_authentication.status_code == 401
    assert not invalid_authentication.headers["content-type"].startswith("text/event-stream")
    assert missing_thread.status_code == 404
    assert not missing_thread.headers["content-type"].startswith("text/event-stream")


def test_completed_request_replays_without_duplicate_messages_and_conflicts_on_new_content() -> None:
    client, headers = _authenticated_client()
    thread_id = client.post(
        "/api/v1/threads",
        json={"title": "idempotency test"},
        headers=headers,
    ).json()["thread_id"]
    request_id = "req_stream_replay_001"
    content = "ETCH-03 Chamber B 清腔后首片异常，当前 SOP 怎么要求？"

    first = _stream_events(client, headers, thread_id, request_id, content)
    replay = _stream_events(client, headers, thread_id, request_id, content)

    assert replay[0].data.replayed is True
    assert [event.event for event in replay] == [
        AgentStreamEventType.ACCEPTED,
        AgentStreamEventType.COMPLETED,
    ]
    assert replay[-1].data.result.response == first[-1].data.result.response
    persisted = client.get(f"/api/v1/threads/{thread_id}", headers=headers).json()
    assert len(persisted["messages"]) == 2

    conflict = client.post(
        f"/api/v1/threads/{thread_id}/messages/stream",
        json={"content": "different content", "request_id": request_id},
        headers=headers,
    )
    assert conflict.status_code == 409


@pytest.mark.asyncio
async def test_disconnect_marks_request_cancelled_without_assistant_message(seeded_services) -> None:
    store, _, _, conversation, _ = seeded_services
    scope = ActorScope(user_id="cancel_test")
    thread = conversation.create_thread("cancel", scope)
    prepared = await conversation.prepare_stream_message(
        thread.thread_id,
        "ETCH-03 Chamber B 清腔后首片异常，当前 SOP 怎么要求？",
        "req_cancel_stream_001",
        scope,
    )
    stream = conversation.stream_message(prepared)

    await anext(stream)  # accepted
    await anext(stream)  # first graph stage, so cancellation occurs inside the guarded run
    await stream.aclose()

    record = store.get_message_request(thread.thread_id, scope.user_id, prepared.record.request_id)
    assert record is not None
    assert record.status is AgentMessageRequestStatus.CANCELLED
    persisted = store.get_thread(thread.thread_id)
    assert persisted is not None
    assert [message.role for message in persisted.messages] == ["user"]


@pytest.mark.asyncio
async def test_accepted_request_rejects_a_concurrent_duplicate(seeded_services) -> None:
    _, _, _, conversation, _ = seeded_services
    scope = ActorScope(user_id="concurrent_test")
    thread = conversation.create_thread("concurrent", scope)
    content = "ETCH-03 Chamber B 清腔后首片异常，当前 SOP 怎么要求？"
    await conversation.prepare_stream_message(
        thread.thread_id,
        content,
        "req_concurrent_stream_001",
        scope,
    )

    with pytest.raises(MessageRequestInProgressError):
        await conversation.prepare_stream_message(
            thread.thread_id,
            content,
            "req_concurrent_stream_001",
            scope,
        )


@pytest.mark.asyncio
async def test_explicit_cancel_interrupts_active_graph_and_allows_retry(seeded_services) -> None:
    store, _, _, conversation, _ = seeded_services
    scope = ActorScope(user_id="explicit_cancel_test")
    thread = conversation.create_thread("explicit cancel", scope)
    content = "ETCH-03 Chamber B 清腔后首片异常，当前 SOP 怎么要求？"
    request_id = "req_explicit_cancel_001"
    prepared = await conversation.prepare_stream_message(
        thread.thread_id,
        content,
        request_id,
        scope,
    )
    stream = conversation.stream_message(prepared)
    await anext(stream)
    await anext(stream)

    cancelled = await conversation.cancel_stream_message(thread.thread_id, request_id, scope)
    assert cancelled.status is AgentMessageRequestStatus.CANCELLED
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    retry = await conversation.prepare_stream_message(
        thread.thread_id,
        content,
        request_id,
        scope,
    )

    assert retry.record.attempt == 2
    persisted = store.get_thread(thread.thread_id)
    assert persisted is not None
    assert [message.role for message in persisted.messages] == ["user"]


@pytest.mark.asyncio
async def test_persistence_failure_never_emits_completed_or_saves_assistant(
    seeded_services,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _, _, conversation, _ = seeded_services
    scope = ActorScope(user_id="persistence_failure_test")
    thread = conversation.create_thread("persistence failure", scope)
    request_id = "req_persistence_failure_001"
    prepared = await conversation.prepare_stream_message(
        thread.thread_id,
        "ETCH-03 Chamber B 清腔后首片异常，当前 SOP 怎么要求？",
        request_id,
        scope,
    )

    def fail_finalize(*args, **kwargs):
        raise RuntimeError("injected persistence failure")

    monkeypatch.setattr(store, "finalize_stream_response", fail_finalize)
    events = [event async for event in conversation.stream_message(prepared)]

    assert events[-1].event is AgentStreamEventType.ERROR
    assert not any(event.event is AgentStreamEventType.COMPLETED for event in events)
    record = store.get_message_request(thread.thread_id, scope.user_id, request_id)
    assert record is not None
    assert record.status is AgentMessageRequestStatus.FAILED
    persisted = store.get_thread(thread.thread_id)
    assert persisted is not None
    assert [message.role for message in persisted.messages] == ["user"]


@pytest.mark.asyncio
async def test_production_graph_streams_multiple_verified_answer_units(seeded_services) -> None:
    store, _, retrieval, _, _ = seeded_services

    class FakeStreamingLLM:
        async def complete(self, *args, **kwargs):
            return LLMCompletion(
                content="{}",
                provider="fake",
                requested_model="fake-model",
                reported_model="fake-model",
                fallback_used=False,
                attempted_providers=("fake",),
                usage={},
            )

        async def stream_complete(self, messages, *, on_content_delta, **kwargs):
            payload = json.loads(messages[-1]["content"])
            evidence_id = payload["evidence_ledger"][0]["evidence_id"]
            content = "\n".join(
                (
                    json.dumps(
                        {"type": "fact", "text": "已核对受控要求", "citation_ids": [evidence_id]},
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {"type": "next_action", "text": "复核首片记录"},
                        ensure_ascii=False,
                    ),
                    '{"type":"confidence","value":"medium"}',
                )
            )
            for delta in (content[:24], content[24:57], content[57:]):
                on_content_delta(delta, "fake", "fake-model")
            return LLMCompletion(
                content=content,
                provider="fake",
                requested_model="fake-model",
                reported_model="fake-model",
                fallback_used=False,
                attempted_providers=("fake",),
                usage={},
            )

    settings = Settings(_env_file=None, demo_mode=False)
    conversation = ConversationService(
        store,
        retrieval,
        settings,
        llm=FakeStreamingLLM(),
    )
    scope = ActorScope(user_id="production_stream_test")
    thread = conversation.create_thread("production stream", scope)
    prepared = await conversation.prepare_stream_message(
        thread.thread_id,
        "ETCH-03 Chamber B 清腔后首片异常，当前 SOP 怎么要求？",
        "req_production_stream_001",
        scope,
    )

    events = [event async for event in conversation.stream_message(prepared)]
    answer_events = [event for event in events if event.event is AgentStreamEventType.ANSWER_DELTA]

    validate_stream_event_sequence(events)
    assert len(answer_events) >= 2
    assert "".join(event.data.delta for event in answer_events) == events[-1].data.result.response
    assert events[-1].data.result.model_metadata["answer_streamed"] is True
