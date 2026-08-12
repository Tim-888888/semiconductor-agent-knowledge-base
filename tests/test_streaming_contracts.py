from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from semikb.contracts.models import SendMessageResponse, ThreadRecord
from semikb.contracts.streaming import (
    AgentMessageRequestRecord,
    AgentMessageRequestStatus,
    AgentStreamErrorCode,
    AgentStreamEventType,
    AgentStreamStage,
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
    agent_stream_event_adapter,
    validate_stream_event_sequence,
)

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
