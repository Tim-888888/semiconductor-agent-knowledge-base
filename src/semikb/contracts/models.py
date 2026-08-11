"""Stable domain and API models for the Phase 1 knowledge base."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class ApprovalStatus(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    REJECTED = "rejected"


class DocumentLifecycle(StrEnum):
    STAGED = "staged"
    PUBLISHED = "published"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    QUARANTINED = "quarantined"


class ChunkType(StrEnum):
    TEXT = "text"
    TABLE = "table"
    IMAGE_TEXT = "image_text"


class IngestionStatus(StrEnum):
    QUEUED = "queued"
    VALIDATING = "validating"
    PARSING = "parsing"
    QUALITY_CHECK = "quality_check"
    EMBEDDING = "embedding"
    STAGED = "staged"
    PUBLISHED = "published"
    FAILED = "failed"


class EvaluationStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ActorScope(BaseModel):
    user_id: str = "demo_engineer"
    roles: list[str] = Field(default_factory=lambda: ["engineer"])
    access_scope_keys: list[str] = Field(default_factory=lambda: ["demo_engineering"])
    fabs: list[str] = Field(default_factory=lambda: ["FAB-01"])
    products: list[str] = Field(default_factory=lambda: ["P-ALPHA"])
    tool_ids: list[str] = Field(default_factory=lambda: ["ETCH-03"])


class ObjectRef(BaseModel):
    bucket: str
    object_key: str
    content_type: str
    sha256: str
    version_id: str | None = None


class DocumentRevision(BaseModel):
    document_id: str
    revision: str
    title: str
    document_type: str
    approval_status: ApprovalStatus = ApprovalStatus.APPROVED
    lifecycle: DocumentLifecycle = DocumentLifecycle.STAGED
    effective_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    supersedes_revision: str | None = None
    source_hash: str
    source_ref: ObjectRef
    access_scope_key: str = "demo_engineering"
    fab: str = "FAB-01"
    product: str = "P-ALPHA"
    process_layer: str | None = None
    tool_id: str | None = None
    chamber: str | None = None
    recipe_id: str | None = None
    recipe_version: str | None = None
    parser_version: str = "demo-parser-v1"
    chunker_version: str = "semantic-v1"
    index_version: str = "v1"
    created_at: datetime = Field(default_factory=utc_now)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    revision: str
    parent_chunk_id: str | None = None
    chunk_type: ChunkType = ChunkType.TEXT
    chunk_text: str
    title_path: list[str] = Field(default_factory=list)
    page_or_section: str
    approval_status: ApprovalStatus = ApprovalStatus.APPROVED
    lifecycle: DocumentLifecycle = DocumentLifecycle.STAGED
    effective_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    access_scope_key: str = "demo_engineering"
    fab: str = "FAB-01"
    product: str = "P-ALPHA"
    process_layer: str | None = None
    tool_id: str | None = None
    chamber: str | None = None
    recipe_id: str | None = None
    recipe_version: str | None = None
    image_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    parser_version: str = "demo-parser-v1"
    chunker_version: str = "semantic-v1"
    embedding_version: str = "bge-m3-demo-v1"
    index_version: str = "v1"
    created_at: datetime = Field(default_factory=utc_now)


class ImageAsset(BaseModel):
    image_id: str
    document_id: str
    revision: str
    parent_chunk_id: str | None = None
    object_ref: ObjectRef
    image_type: str
    caption: str
    caption_source: str = "human"
    caption_confidence: float = Field(default=1.0, ge=0, le=1)
    ocr_text: str = ""
    detection_summary: str = ""
    source_page: str = ""
    related_case_id: str | None = None
    access_scope_key: str = "demo_engineering"
    approval_status: ApprovalStatus = ApprovalStatus.APPROVED
    lifecycle: DocumentLifecycle = DocumentLifecycle.STAGED
    effective_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None


class IngestionEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: new_id("ing_evt"))
    stage: IngestionStatus
    message: str
    created_at: datetime = Field(default_factory=utc_now)


class IngestionJob(BaseModel):
    job_id: str = Field(default_factory=lambda: new_id("ing"))
    document_id: str
    revision: str
    filename: str
    file_type: str
    source_hash: str
    status: IngestionStatus = IngestionStatus.QUEUED
    current_stage: IngestionStatus = IngestionStatus.QUEUED
    progress: int = Field(default=0, ge=0, le=100)
    attempt: int = 1
    idempotency_key: str
    parser_version: str = "demo-parser-v1"
    chunker_version: str = "semantic-v1"
    embedding_version: str = "bge-m3-demo-v1"
    index_version: str = "v1"
    chunks_count: int = 0
    images_count: int = 0
    tables_count: int = 0
    error_code: str | None = None
    safe_error_summary: str | None = None
    failed_stage: IngestionStatus | None = None
    created_by: str = "demo_admin"
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    events: list[IngestionEvent] = Field(default_factory=list)


class RetrievalCandidate(BaseModel):
    chunk_id: str
    document_id: str
    revision: str
    title: str
    page_or_section: str
    routes: list[str]
    dense_score: float
    sparse_score: float
    rrf_score: float
    rerank_score: float
    selected: bool = False
    exclusion_reason: str | None = None
    protected_evidence: bool = False


class RetrievalTrace(BaseModel):
    trace_id: str = Field(default_factory=lambda: new_id("trace"))
    thread_id: str | None = None
    actor_user_id: str
    access_scope_keys: list[str] = Field(default_factory=list)
    original_query: str
    rewritten_query: str | None = None
    hyde_query: str | None = None
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    routes: list[str] = Field(default_factory=list)
    candidates: list[RetrievalCandidate] = Field(default_factory=list)
    cutoff_reason: str = "top_k"
    final_evidence_ids: list[str] = Field(default_factory=list)
    image_asset_ids: list[str] = Field(default_factory=list)
    external_evidence: list[dict[str, Any]] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ChatMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: new_id("msg"))
    role: str
    content: str
    created_at: datetime = Field(default_factory=utc_now)
    citations: list[dict[str, Any]] = Field(default_factory=list)


class ThreadRecord(BaseModel):
    thread_id: str = Field(default_factory=lambda: new_id("thread"))
    title: str = "New investigation"
    actor_scope: ActorScope = Field(default_factory=ActorScope)
    status: str = "active"
    summary: str = ""
    clarification_round: int = 0
    pending_fields: list[str] = Field(default_factory=list)
    messages: list[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class EvaluationCase(BaseModel):
    case_id: str
    question: str
    expected_chunk_ids: list[str]
    actor_scope: ActorScope = Field(default_factory=ActorScope)
    tags: list[str] = Field(default_factory=list)


class EvaluationRun(BaseModel):
    evaluation_run_id: str = Field(default_factory=lambda: new_id("eval"))
    dataset_version: str
    baseline_run_id: str | None = None
    status: EvaluationStatus = EvaluationStatus.QUEUED
    retrieval_config: dict[str, Any] = Field(default_factory=dict)
    aggregate_metrics: dict[str, float] = Field(default_factory=dict)
    case_results: list[dict[str, Any]] = Field(default_factory=list)
    failure_tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None


class CreateThreadRequest(BaseModel):
    title: str = "New investigation"
    actor_scope: ActorScope = Field(default_factory=ActorScope)


class SendMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    actor_scope: ActorScope = Field(default_factory=ActorScope)
    thread_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class IngestDocumentRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=120)
    revision: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=500)
    document_type: str = Field(min_length=1, max_length=64)
    content: str = Field(min_length=1)
    approval_status: ApprovalStatus = ApprovalStatus.APPROVED
    lifecycle: DocumentLifecycle = DocumentLifecycle.PUBLISHED
    fab: str = "FAB-01"
    product: str = "P-ALPHA"
    process_layer: str | None = None
    tool_id: str | None = None
    chamber: str | None = None
    recipe_id: str | None = None
    recipe_version: str | None = None
    images: list[dict[str, Any]] = Field(default_factory=list)


class IngestUploadMetadata(BaseModel):
    document_id: str = Field(min_length=1, max_length=120)
    revision: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=500)
    document_type: str = Field(min_length=1, max_length=64)
    approval_status: ApprovalStatus = ApprovalStatus.APPROVED
    lifecycle: DocumentLifecycle = DocumentLifecycle.PUBLISHED
    fab: str = "FAB-01"
    product: str = "P-ALPHA"
    process_layer: str | None = None
    tool_id: str | None = None
    chamber: str | None = None
    recipe_id: str | None = None
    recipe_version: str | None = None
    images: list[dict[str, Any]] = Field(default_factory=list)


class CreateEvaluationRunRequest(BaseModel):
    dataset_version: str = "demo-v1"
    baseline_run_id: str | None = None
