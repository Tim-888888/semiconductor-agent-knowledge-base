from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from semikb.contracts.models import (
    AgentRoute,
    SendMessageResponse,
    TaskExecutionResult,
    TaskExecutionStatus,
    ThreadRecord,
)
from semikb.contracts.streaming import (
    AgentMessageRequestRecord,
    AgentMessageRequestStatus,
    AgentStreamErrorCode,
    AgentStreamEventType,
    AgentStreamStage,
    DirectReplyAudit,
    StreamAcceptedData,
    StreamAcceptedEvent,
    StreamAnswerDeltaData,
    StreamAnswerDeltaEvent,
    StreamCompletedData,
    StreamCompletedEvent,
    StreamErrorData,
    StreamErrorEvent,
    StreamMessageRequest,
    StreamStageData,
    StreamStageEvent,
    StreamTaskStatusData,
    StreamTaskStatusEvent,
    agent_stream_event_adapter,
    validate_stream_event_sequence,
)
from semikb.storage.conversations import MongoConversationRepository

NOW = datetime(2026, 8, 12, tzinfo=UTC)
REQUEST_ID = "req_12345678"
THREAD_ID = "thread_12345678"


def accepted(sequence: int = 1) -> StreamAcceptedEvent:
    return StreamAcceptedEvent(
        event_id=f"sse_{sequence}",
        request_id=REQUEST_ID,
        thread_id=THREAD_ID,
        sequence=sequence,
        emitted_at=NOW,
        data=StreamAcceptedData(message_id="msg_user", run_id="run_1"),
    )


def test_stream_request_generates_a_retryable_idempotency_key() -> None:
    request = StreamMessageRequest(content="检查 ETCH-03")

    assert request.request_id.startswith("req_")
    assert len(request.request_id) >= 8


def test_request_ledger_stores_a_hash_instead_of_duplicate_message_content() -> None:
    record = AgentMessageRequestRecord(
        request_id=REQUEST_ID,
        thread_id=THREAD_ID,
        actor_user_id="demo_engineer",
        content_sha256="a" * 64,
        user_message_id="msg_user",
        run_id="run_1",
        status=AgentMessageRequestStatus.RUNNING,
    )

    dumped = record.model_dump(mode="json")
    assert dumped["content_sha256"] == "a" * 64
    assert "content" not in dumped


def test_mongo_terminal_transition_persists_direct_reply_audit() -> None:
    record = AgentMessageRequestRecord(
        request_id=REQUEST_ID,
        thread_id=THREAD_ID,
        actor_user_id="demo_engineer",
        content_sha256="a" * 64,
        user_message_id="msg_user",
        run_id="run_1",
        status=AgentMessageRequestStatus.RUNNING,
        direct_reply_audit=DirectReplyAudit(
            reply_kind="history_recall",
            generation_mode="llm_stream",
            provider="test",
            model="test-model",
            latency_ms=12.5,
            verified_unit_count=2,
            context_message_count=1,
        ),
        task_results=[
            TaskExecutionResult(
                task_id="task_1",
                status=TaskExecutionStatus.COMPLETED,
                route=AgentRoute.CHAT_DIRECT,
                reason_code="route_contract_satisfied",
                message="历史回顾已完成。",
            )
        ],
    )

    class MessageRequests:
        def __init__(self) -> None:
            self.update: dict[str, object] | None = None

        def find_one_and_update(self, query, update, **kwargs):
            self.update = update
            document = record.model_dump(mode="python")
            document.update(update["$set"])
            return document

    class Threads:
        def update_one(self, *args, **kwargs):
            return None

    repository = object.__new__(MongoConversationRepository)
    repository.message_requests = MessageRequests()
    repository.threads = Threads()

    terminal = repository.mark_message_request_terminal(
        record,
        AgentMessageRequestStatus.COMPLETED,
    )

    assert terminal.direct_reply_audit == record.direct_reply_audit
    assert terminal.task_results == record.task_results
    assert repository.message_requests.update is not None
    assert repository.message_requests.update["$set"]["direct_reply_audit"]["reply_kind"] == (
        "history_recall"
    )
    assert repository.message_requests.update["$set"]["task_results"][0]["status"] == (
        TaskExecutionStatus.COMPLETED
    )


def test_task_status_event_is_part_of_the_discriminated_stream_contract() -> None:
    event = StreamTaskStatusEvent(
        event_id="sse_task_1",
        request_id=REQUEST_ID,
        thread_id=THREAD_ID,
        sequence=2,
        emitted_at=NOW,
        data=StreamTaskStatusData(
            task_id="task_1",
            status="running",
            route=AgentRoute.INTERNAL_RAG,
            message="正在检索受控知识",
        ),
    )

    parsed = agent_stream_event_adapter.validate_python(event.model_dump(mode="json"))

    assert parsed.event is AgentStreamEventType.TASK_STATUS
    assert parsed.data.task_id == "task_1"


def test_discriminated_event_adapter_rejects_mismatched_payload() -> None:
    with pytest.raises(ValidationError):
        agent_stream_event_adapter.validate_python(
            {
                "event": "accepted",
                "event_id": "sse_1",
                "request_id": REQUEST_ID,
                "thread_id": THREAD_ID,
                "sequence": 1,
                "data": {"stage": "analyzing_request", "message": "分析中"},
            }
        )


def test_stream_sequence_accepts_contiguous_terminal_flow() -> None:
    events = [
        accepted(),
        StreamStageEvent(
            event_id="sse_2",
            request_id=REQUEST_ID,
            thread_id=THREAD_ID,
            sequence=2,
            data=StreamStageData(
                stage=AgentStreamStage.ANALYZING_REQUEST,
                message="正在分析问题",
            ),
        ),
        StreamAnswerDeltaEvent(
            event_id="sse_3",
            request_id=REQUEST_ID,
            thread_id=THREAD_ID,
            sequence=3,
            data=StreamAnswerDeltaData(delta="建议先确认腔体状态。"),
        ),
        StreamCompletedEvent(
            event_id="sse_4",
            request_id=REQUEST_ID,
            thread_id=THREAD_ID,
            sequence=4,
            data=StreamCompletedData(
                run_id="run_1",
                result=SendMessageResponse(
                    thread=ThreadRecord(thread_id=THREAD_ID),
                    response="建议先确认腔体状态。",
                    clarification_required=False,
                ),
            ),
        ),
    ]

    validate_stream_event_sequence(events)
    assert events[-1].event == AgentStreamEventType.COMPLETED


def test_stream_sequence_rejects_events_after_terminal() -> None:
    events = [
        accepted(),
        StreamErrorEvent(
            event_id="sse_2",
            request_id=REQUEST_ID,
            thread_id=THREAD_ID,
            sequence=2,
            data=StreamErrorData(
                code=AgentStreamErrorCode.CANCELLED,
                message="生成已停止。",
            ),
        ),
        StreamAnswerDeltaEvent(
            event_id="sse_3",
            request_id=REQUEST_ID,
            thread_id=THREAD_ID,
            sequence=3,
            data=StreamAnswerDeltaData(delta="不应出现"),
        ),
    ]

    with pytest.raises(ValueError, match="terminal event"):
        validate_stream_event_sequence(events)
