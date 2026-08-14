"""Frozen wire contracts for the T9-4 Agent SSE stream."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from semikb.contracts.models import (
    AffectSignals,
    AgentRoute,
    CancelScope,
    IntentTaskItem,
    InteractionMode,
    RouteTaskDecision,
    SendMessageResponse,
    SlotOperation,
    new_id,
    utc_now,
)

REQUEST_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$"


class AgentStreamEventType(StrEnum):
    ACCEPTED = "accepted"
    STAGE = "stage"
    EVIDENCE = "evidence"
    ANSWER_DELTA = "answer_delta"
    HEARTBEAT = "heartbeat"
    COMPLETED = "completed"
    ERROR = "error"


class AgentStreamStage(StrEnum):
    ANALYZING_REQUEST = "analyzing_request"
    ROUTING_REQUEST = "routing_request"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    RETRIEVING_EVIDENCE = "retrieving_evidence"
    SEARCHING_EXTERNAL = "searching_external"
    RERANKING_EVIDENCE = "reranking_evidence"
    GENERATING_ANSWER = "generating_answer"
    VERIFYING_ANSWER = "verifying_answer"
    PERSISTING_RESULT = "persisting_result"


class AgentStreamErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    THREAD_NOT_FOUND = "thread_not_found"
    REQUEST_CONFLICT = "request_conflict"
    REQUEST_IN_PROGRESS = "request_in_progress"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_RATE_LIMITED = "provider_rate_limited"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    VERIFICATION_FAILED = "verification_failed"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"


class AgentMessageRequestStatus(StrEnum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StreamMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=8000)
    request_id: str = Field(
        default_factory=lambda: new_id("req"),
        min_length=8,
        max_length=128,
        pattern=REQUEST_ID_PATTERN,
    )


class UnderstandingAudit(BaseModel):
    """Sanitized intent-governance evidence; prompts and card bodies are never persisted."""

    intent_catalog_version: str | None = Field(default=None, max_length=64)
    intent_catalog_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    active_intent_card_count: int = Field(default=0, ge=0)
    intent_card_selection: str | None = Field(default=None, max_length=32)
    intent_cards_in_prompt: int = Field(default=0, ge=0)
    intent_prompt_tokens: int = Field(default=0, ge=0)
    intent_prompt_tokens_source: str | None = Field(default=None, max_length=64)
    intent_catalog_capacity_warnings: list[str] = Field(default_factory=list, max_length=8)
    understanding_source: str | None = Field(default=None, max_length=64)
    understanding_calls: int = Field(default=0, ge=0, le=2)
    understanding_provider: str | None = Field(default=None, max_length=128)
    understanding_model: str | None = Field(default=None, max_length=128)
    understanding_fallback_used: bool = False
    understanding_repaired: bool = False
    understanding_warning: str | None = Field(default=None, max_length=128)
    understanding_latency_ms: float = Field(default=0, ge=0)


class DirectReplyAudit(BaseModel):
    """Credential-safe evidence for controlled direct-reply generation."""

    reply_kind: str = Field(min_length=1, max_length=64)
    generation_mode: str = Field(
        pattern="^(llm_stream|deterministic_fallback|partial_fallback)$"
    )
    provider: str | None = Field(default=None, max_length=128)
    model: str | None = Field(default=None, max_length=128)
    fallback_used: bool = False
    latency_ms: float = Field(default=0, ge=0)
    verified_unit_count: int = Field(default=0, ge=0, le=32)
    context_message_count: int = Field(default=0, ge=0, le=24)
    warning_codes: list[str] = Field(default_factory=list, max_length=16)
    usage: dict[str, int] = Field(default_factory=dict)


class AgentMessageRequestRecord(BaseModel):
    """Persistent idempotency ledger; message content remains in the thread only."""

    request_id: str = Field(min_length=8, max_length=128, pattern=REQUEST_ID_PATTERN)
    thread_id: str = Field(min_length=1, max_length=128)
    actor_user_id: str = Field(min_length=1, max_length=128)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    user_message_id: str
    run_id: str
    user_turn_seq: int | None = Field(default=None, ge=1)
    assistant_turn_seq: int | None = Field(default=None, ge=1)
    status: AgentMessageRequestStatus = AgentMessageRequestStatus.ACCEPTED
    attempt: int = Field(default=1, ge=1)
    assistant_message_id: str | None = None
    trace_id: str | None = None
    result_payload: dict[str, Any] = Field(default_factory=dict)
    interaction_mode: InteractionMode | None = None
    route_decision: AgentRoute | None = None
    route_confidence: float | None = Field(default=None, ge=0, le=1)
    task_items: list[IntentTaskItem] = Field(default_factory=list, max_length=3)
    task_decisions: list[RouteTaskDecision] = Field(default_factory=list, max_length=3)
    context_message_ids: list[str] = Field(default_factory=list, max_length=8)
    standalone_query: str = Field(default="", max_length=8000)
    retrieval_skipped_reason: str | None = None
    slot_operations: list[SlotOperation] = Field(default_factory=list, max_length=12)
    inherited_slots: dict[str, str] = Field(default_factory=dict)
    invalidated_context_refs: list[str] = Field(default_factory=list)
    cancel_scope: CancelScope | None = None
    affect: AffectSignals = Field(default_factory=AffectSignals)
    understanding_audit: UnderstandingAudit = Field(default_factory=UnderstandingAudit)
    direct_reply_audit: DirectReplyAudit | None = None
    error_code: AgentStreamErrorCode | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None


class StreamAcceptedData(BaseModel):
    message_id: str
    run_id: str
    attempt: int = Field(default=1, ge=1)
    replayed: bool = False


class StreamStageData(BaseModel):
    stage: AgentStreamStage
    message: str = Field(min_length=1, max_length=240)


class StreamEvidenceData(BaseModel):
    trace_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    image_asset_ids: list[str] = Field(default_factory=list)
    internal_count: int = Field(default=0, ge=0)
    external_count: int = Field(default=0, ge=0)


class StreamAnswerDeltaData(BaseModel):
    delta: str = Field(min_length=1)
    provider: str | None = None
    model: str | None = None


class StreamHeartbeatData(BaseModel):
    elapsed_ms: int = Field(ge=0)


class StreamCompletedData(BaseModel):
    run_id: str
    result: SendMessageResponse


class StreamErrorData(BaseModel):
    code: AgentStreamErrorCode
    message: str = Field(min_length=1, max_length=500)
    retryable: bool = False
    retry_after_seconds: int | None = Field(default=None, ge=1, le=3600)
    trace_id: str | None = None


class AgentStreamEventBase(BaseModel):
    event_id: str = Field(default_factory=lambda: new_id("sse"))
    request_id: str = Field(min_length=8, max_length=128, pattern=REQUEST_ID_PATTERN)
    thread_id: str = Field(min_length=1, max_length=128)
    sequence: int = Field(ge=1)
    emitted_at: datetime = Field(default_factory=utc_now)


class StreamAcceptedEvent(AgentStreamEventBase):
    event: Literal[AgentStreamEventType.ACCEPTED] = AgentStreamEventType.ACCEPTED
    data: StreamAcceptedData


class StreamStageEvent(AgentStreamEventBase):
    event: Literal[AgentStreamEventType.STAGE] = AgentStreamEventType.STAGE
    data: StreamStageData


class StreamEvidenceEvent(AgentStreamEventBase):
    event: Literal[AgentStreamEventType.EVIDENCE] = AgentStreamEventType.EVIDENCE
    data: StreamEvidenceData


class StreamAnswerDeltaEvent(AgentStreamEventBase):
    event: Literal[AgentStreamEventType.ANSWER_DELTA] = AgentStreamEventType.ANSWER_DELTA
    data: StreamAnswerDeltaData


class StreamHeartbeatEvent(AgentStreamEventBase):
    event: Literal[AgentStreamEventType.HEARTBEAT] = AgentStreamEventType.HEARTBEAT
    data: StreamHeartbeatData


class StreamCompletedEvent(AgentStreamEventBase):
    event: Literal[AgentStreamEventType.COMPLETED] = AgentStreamEventType.COMPLETED
    data: StreamCompletedData


class StreamErrorEvent(AgentStreamEventBase):
    event: Literal[AgentStreamEventType.ERROR] = AgentStreamEventType.ERROR
    data: StreamErrorData


AgentStreamEvent = Annotated[
    StreamAcceptedEvent
    | StreamStageEvent
    | StreamEvidenceEvent
    | StreamAnswerDeltaEvent
    | StreamHeartbeatEvent
    | StreamCompletedEvent
    | StreamErrorEvent,
    Field(discriminator="event"),
]

agent_stream_event_adapter = TypeAdapter(AgentStreamEvent)


def encode_sse_event(event: AgentStreamEvent) -> bytes:
    """Serialize one complete event so proxies and clients can flush it immediately."""

    payload = event.model_dump_json()
    return f"id: {event.event_id}\nevent: {event.event}\ndata: {payload}\n\n".encode()


def validate_stream_event_sequence(events: list[AgentStreamEvent]) -> None:
    """Reject a stream that cannot be replayed deterministically by a client."""

    if not events:
        raise ValueError("stream must contain at least one event")
    if events[0].event != AgentStreamEventType.ACCEPTED:
        raise ValueError("stream must begin with accepted")

    request_id = events[0].request_id
    thread_id = events[0].thread_id
    terminal_seen = False
    for expected_sequence, event in enumerate(events, start=1):
        if event.request_id != request_id or event.thread_id != thread_id:
            raise ValueError("all stream events must belong to the same request and thread")
        if event.sequence != expected_sequence:
            raise ValueError("stream event sequence must be contiguous and start at 1")
        if terminal_seen:
            raise ValueError("terminal event must be the final stream event")
        terminal_seen = event.event in {
            AgentStreamEventType.COMPLETED,
            AgentStreamEventType.ERROR,
        }

    if not terminal_seen:
        raise ValueError("stream must end with completed or error")
