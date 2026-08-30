"""Stable domain and API models for the Phase 1 knowledge base."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from semikb_provider_resilience import ProviderAttemptAudit


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
    WITHDRAWN = "withdrawn"


class SourceManifestStatus(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    RETIRED = "retired"


class SourceManifestType(StrEnum):
    DATASET = "dataset"
    PAPER = "paper"
    REPOSITORY = "repository"
    ONTOLOGY = "ontology"
    DOCUMENTATION = "documentation"
    CURATED_CORPUS = "curated_corpus"
    OTHER = "other"


class SourceContentOrigin(StrEnum):
    REAL = "real"
    SYNTHETIC = "synthetic"
    DERIVED = "derived"


class SourceLicenseStatus(StrEnum):
    VERIFIED = "verified"
    DECLARED = "declared"
    UNCLEAR = "unclear"
    RESTRICTED = "restricted"


class RedistributionPolicy(StrEnum):
    ALLOWED = "allowed"
    RESTRICTED = "restricted"
    PROHIBITED = "prohibited"
    UNKNOWN = "unknown"


class SourceIngestionMode(StrEnum):
    DOCUMENT_RAG = "document_rag"
    TABULAR_PROFILE_AND_TOOL = "tabular_profile_and_tool"
    IMAGE_CORPUS = "image_corpus"
    MIXED_CURATED_CORPUS = "mixed_curated_corpus"
    REFERENCE_ONLY = "reference_only"


class SourceIndexArtifact(StrEnum):
    DOCUMENT_CHUNKS = "document_chunks"
    DATA_DICTIONARY = "data_dictionary"
    DATASET_PROFILE = "dataset_profile"
    ANALYSIS_REPORT = "analysis_report"
    IMAGE_TEXT = "image_text"


class DocumentLifecycleAction(StrEnum):
    WITHDRAW = "withdraw"
    RESTORE = "restore"


class DocumentLifecycleOperationStatus(StrEnum):
    REQUESTED = "requested"
    BLOCKING = "blocking"
    VECTOR_CLEANUP = "vector_cleanup"
    WITHDRAWN = "withdrawn"
    RESTORE_VALIDATING = "restore_validating"
    RESTORE_INDEXING = "restore_indexing"
    RESTORED = "restored"
    COMPENSATION_REQUIRED = "compensation_required"
    FAILED = "failed"


class CompensationStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ChunkType(StrEnum):
    TEXT = "text"
    TABLE = "table"
    IMAGE_TEXT = "image_text"


class RetrievalPolicy(StrEnum):
    STANDARD = "standard"
    PROTECTED = "protected"


class RetrievalMode(StrEnum):
    STANDARD = "standard"
    DIAGNOSTIC = "diagnostic"
    IMAGE = "image"


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


class EvaluationDatasetPurpose(StrEnum):
    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    REGRESSION = "regression"
    HOLDOUT = "holdout"


class EvaluationLeakageStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    CLEARED = "cleared"
    CONTAMINATED = "contaminated"


class ActorScope(BaseModel):
    user_id: str = "anonymous"
    roles: list[str] = Field(default_factory=list)
    access_scope_keys: list[str] = Field(default_factory=list)
    fabs: list[str] = Field(default_factory=list)
    products: list[str] = Field(default_factory=list)
    tool_ids: list[str] = Field(default_factory=list)


class ObjectRef(BaseModel):
    bucket: str
    object_key: str
    content_type: str
    sha256: str
    version_id: str | None = None


class SourceIngestionPolicy(BaseModel):
    """Controls which derived representations may enter RAG for one source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: SourceIngestionMode
    raw_storage: Literal["minio_private"] = "minio_private"
    raw_row_vectorization: bool = False
    index_artifacts: list[SourceIndexArtifact] = Field(default_factory=list)
    analysis_tool_required: bool = False

    @model_validator(mode="after")
    def validate_representation_boundary(self) -> SourceIngestionPolicy:
        if self.raw_row_vectorization:
            raise ValueError("Raw tabular rows must never be vectorized as individual RAG chunks.")
        if len(self.index_artifacts) != len(set(self.index_artifacts)):
            raise ValueError("index_artifacts must not contain duplicates.")
        artifacts = set(self.index_artifacts)
        if self.mode is SourceIngestionMode.DOCUMENT_RAG:
            if SourceIndexArtifact.DOCUMENT_CHUNKS not in artifacts:
                raise ValueError("document_rag requires document_chunks.")
        elif self.mode is SourceIngestionMode.TABULAR_PROFILE_AND_TOOL:
            required = {
                SourceIndexArtifact.DATA_DICTIONARY,
                SourceIndexArtifact.DATASET_PROFILE,
            }
            if not required.issubset(artifacts) or not self.analysis_tool_required:
                raise ValueError(
                    "tabular_profile_and_tool requires data_dictionary, dataset_profile, "
                    "and analysis_tool_required=true."
                )
        elif self.mode is SourceIngestionMode.IMAGE_CORPUS:
            if SourceIndexArtifact.IMAGE_TEXT not in artifacts:
                raise ValueError("image_corpus requires image_text.")
        elif self.mode is SourceIngestionMode.MIXED_CURATED_CORPUS:
            if not artifacts:
                raise ValueError("mixed_curated_corpus requires at least one index artifact.")
        elif self.mode is SourceIngestionMode.REFERENCE_ONLY and artifacts:
            raise ValueError("reference_only sources cannot declare index artifacts.")
        return self


class SourceExpectedAssets(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_files_min: int = Field(default=1, ge=1)
    documents_min: int = Field(default=0, ge=0)
    images_min: int = Field(default=0, ge=0)
    tables_min: int = Field(default=0, ge=0)
    records_estimate: int | None = Field(default=None, ge=0)
    expected_formats: list[str] = Field(default_factory=list)


class SourceManifest(BaseModel):
    """Immutable, versioned provenance card for one acquired source snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_schema_version: Literal["semikb-source-manifest-v1"] = (
        "semikb-source-manifest-v1"
    )
    source_id: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9._:-]+$")
    manifest_version: str = Field(min_length=1, max_length=64)
    status: SourceManifestStatus = SourceManifestStatus.DRAFT
    title: str = Field(min_length=1, max_length=500)
    source_type: SourceManifestType
    source_url: str = Field(min_length=1, max_length=2000)
    doi_or_repo: str | None = Field(default=None, max_length=2000)
    retrieved_at: datetime
    source_hash: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    hash_scope: str = Field(min_length=1, max_length=1000)
    content_origin: SourceContentOrigin
    license_name: str = Field(min_length=1, max_length=200)
    license_status: SourceLicenseStatus
    redistribution_policy: RedistributionPolicy
    license_notes: str = Field(default="", max_length=4000)
    ingestion_policy: SourceIngestionPolicy
    parser_hint: str | None = Field(default=None, max_length=200)
    expected_assets: SourceExpectedAssets = Field(default_factory=SourceExpectedAssets)
    dataset_version: str = Field(min_length=1, max_length=160)
    access_scope_key: str = Field(min_length=1, max_length=160)
    source_snapshot_ref: ObjectRef | None = None
    supersedes_manifest_version: str | None = Field(default=None, max_length=64)
    created_by: str = Field(min_length=1, max_length=160)
    created_at: datetime = Field(default_factory=utc_now)
    notes: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def validate_version_chain(self) -> SourceManifest:
        if self.supersedes_manifest_version == self.manifest_version:
            raise ValueError("A source manifest cannot supersede itself.")
        return self


def _validate_source_manifest_link(
    source_id: str | None,
    source_manifest_version: str | None,
) -> None:
    if bool(source_id) != bool(source_manifest_version):
        raise ValueError("source_id and source_manifest_version must be provided together.")


def _validate_publication_request(request: Any) -> None:
    if request.lifecycle is not DocumentLifecycle.PUBLISHED:
        return
    if request.approval_status is not ApprovalStatus.APPROVED:
        raise ValueError("A published upload must be explicitly approved.")
    required = {
        "access_scope_key": request.access_scope_key,
        "source_id": request.source_id,
        "source_manifest_version": request.source_manifest_version,
        "dataset_version": request.dataset_version,
        "source_license_status": request.source_license_status,
        "redistribution_policy": request.redistribution_policy,
    }
    missing = sorted(name for name, value in required.items() if value is None or value == "")
    if missing:
        raise ValueError(
            "Published uploads require reviewed governance fields: " + ", ".join(missing)
        )


class DocumentRevision(BaseModel):
    document_id: str
    revision: str
    title: str
    document_type: str
    approval_status: ApprovalStatus = ApprovalStatus.DRAFT
    lifecycle: DocumentLifecycle = DocumentLifecycle.STAGED
    effective_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    supersedes_revision: str | None = None
    source_hash: str
    source_ref: ObjectRef
    parsed_ref: ObjectRef | None = None
    source_kind: str = "unknown"
    source_uri: str = ""
    source_license: str = "unknown"
    source_id: str | None = None
    source_manifest_version: str | None = None
    dataset_version: str | None = None
    source_license_status: SourceLicenseStatus | None = None
    redistribution_policy: RedistributionPolicy | None = None
    access_scope_key: str | None = None
    fab: str | None = None
    product: str | None = None
    process_layer: str | None = None
    tool_id: str | None = None
    chamber: str | None = None
    recipe_id: str | None = None
    recipe_version: str | None = None
    parse_contract_version: str = "legacy-ingestion-v1"
    parser_name: str = "legacy"
    parser_version: str = "unknown"
    provider_name: str | None = None
    provider_version: str | None = None
    upstream_project: str | None = None
    upstream_commit: str | None = None
    detected_title: str | None = None
    detected_language: str | None = None
    parse_warning_codes: list[str] = Field(default_factory=list)
    parse_metrics: dict[str, Any] = Field(default_factory=dict)
    chunker_version: str = "semantic-v1"
    embedding_version: str = "unknown"
    index_version: str = "v1"
    retrieval_policy: RetrievalPolicy = RetrievalPolicy.STANDARD
    source_index_artifacts: list[SourceIndexArtifact] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_source_manifest_link(self) -> DocumentRevision:
        _validate_source_manifest_link(self.source_id, self.source_manifest_version)
        return self


class DocumentRevisionSelector(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1, max_length=120)
    revision: str = Field(min_length=1, max_length=64)


class AffectedRecordCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    documents: int = Field(default=0, ge=0)
    chunks: int = Field(default=0, ge=0)
    images: int = Field(default=0, ge=0)
    tables: int = Field(default=0, ge=0)
    vectors: int = Field(default=0, ge=0)


class KnowledgeDocumentRevisionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    revision: str
    title: str
    document_type: str
    approval_status: ApprovalStatus
    lifecycle: DocumentLifecycle
    effective_at: datetime
    expires_at: datetime | None = None
    source_id: str | None = None
    source_manifest_version: str | None = None
    dataset_version: str | None = None
    source_uri: str = ""
    source_license: str = "internal"
    source_license_status: SourceLicenseStatus | None = None
    redistribution_policy: RedistributionPolicy | None = None
    access_scope_key: str
    counts: AffectedRecordCounts = Field(default_factory=AffectedRecordCounts)
    created_at: datetime


class KnowledgeDocumentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    title: str
    document_type: str
    current_revision: str | None = None
    current_lifecycle: DocumentLifecycle | None = None
    revision_count: int = Field(ge=0)
    source_id: str | None = None
    dataset_version: str | None = None
    updated_at: datetime


class KnowledgeDocumentListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[KnowledgeDocumentSummary]
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=200)
    offset: int = Field(ge=0)


class WithdrawDocumentRevisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=160)
    reason: str = Field(min_length=8, max_length=1000)


class RestoreDocumentRevisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=160)
    reason: str = Field(min_length=8, max_length=1000)
    target_index_version: str | None = Field(default=None, min_length=1, max_length=64)


class DocumentLifecycleOperationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(default_factory=lambda: new_id("doc_lifecycle"))
    request_id: str
    action: DocumentLifecycleAction
    status: DocumentLifecycleOperationStatus
    selector: DocumentRevisionSelector
    actor_user_id: str
    reason: str
    before_lifecycle: DocumentLifecycle
    after_lifecycle: DocumentLifecycle | None = None
    target_index_version: str | None = Field(default=None, min_length=1, max_length=64)
    affected: AffectedRecordCounts = Field(default_factory=AffectedRecordCounts)
    compensation_status: CompensationStatus = CompensationStatus.NOT_REQUIRED
    warning_codes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    revision: str
    parent_chunk_id: str | None = None
    chunk_type: ChunkType = ChunkType.TEXT
    chunk_text: str
    title_path: list[str] = Field(default_factory=list)
    page_or_section: str
    approval_status: ApprovalStatus = ApprovalStatus.DRAFT
    lifecycle: DocumentLifecycle = DocumentLifecycle.STAGED
    effective_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    access_scope_key: str | None = None
    fab: str | None = None
    product: str | None = None
    process_layer: str | None = None
    tool_id: str | None = None
    chamber: str | None = None
    recipe_id: str | None = None
    recipe_version: str | None = None
    image_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    parser_name: str = "legacy"
    parser_version: str = "unknown"
    upstream_commit: str | None = None
    chunker_version: str = "semantic-v1"
    embedding_version: str = "unknown"
    index_version: str = "v1"
    retrieval_policy: RetrievalPolicy = RetrievalPolicy.STANDARD
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
    source_asset_id: str | None = None
    source_location: dict[str, Any] = Field(default_factory=dict)
    parser_name: str = "legacy"
    parser_version: str = "unknown"
    provider_name: str | None = None
    provider_version: str | None = None
    related_case_id: str | None = None
    demo_source_path: str | None = None
    access_scope_key: str | None = None
    approval_status: ApprovalStatus = ApprovalStatus.DRAFT
    lifecycle: DocumentLifecycle = DocumentLifecycle.STAGED
    effective_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None


class TableAsset(BaseModel):
    """Governed table representation with one durable derived-object reference."""

    table_id: str
    document_id: str
    revision: str
    parent_chunk_id: str | None = None
    object_ref: ObjectRef
    title: str = ""
    markdown: str
    html: str
    headers: list[str] = Field(default_factory=list)
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    source_asset_id: str | None = None
    source_page: str = ""
    source_location: dict[str, Any] = Field(default_factory=dict)
    parser_name: str = "legacy"
    parser_version: str = "unknown"
    access_scope_key: str | None = None
    approval_status: ApprovalStatus = ApprovalStatus.DRAFT
    lifecycle: DocumentLifecycle = DocumentLifecycle.STAGED
    effective_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None


class IngestionEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: new_id("ing_evt"))
    job_id: str | None = None
    stage: IngestionStatus
    message: str
    attempt: int = Field(default=1, ge=1)
    progress: int = Field(default=0, ge=0, le=100)
    provider_attempts: list[ProviderAttemptAudit] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class IngestionJob(BaseModel):
    job_id: str = Field(default_factory=lambda: new_id("ing"))
    document_id: str
    revision: str
    filename: str
    file_type: str
    source_hash: str
    source_ref: ObjectRef | None = None
    parsed_ref: ObjectRef | None = None
    status: IngestionStatus = IngestionStatus.QUEUED
    current_stage: IngestionStatus = IngestionStatus.QUEUED
    progress: int = Field(default=0, ge=0, le=100)
    attempt: int = 1
    idempotency_key: str
    parse_contract_version: str = "legacy-ingestion-v1"
    parser_name: str = "legacy"
    parser_version: str = "demo-parser-v1"
    provider_name: str | None = None
    provider_version: str | None = None
    upstream_project: str | None = None
    upstream_commit: str | None = None
    parse_warning_codes: list[str] = Field(default_factory=list)
    parse_metrics: dict[str, Any] = Field(default_factory=dict)
    provider_attempts: list[ProviderAttemptAudit] = Field(default_factory=list)
    chunker_version: str = "semantic-v1"
    embedding_version: str = "deterministic-demo-v1"
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
    hyde_score: float = 0.0
    rrf_score: float
    rerank_score: float
    route_ranks: dict[str, int] = Field(default_factory=dict)
    context_selection_reason: str | None = None
    selected: bool = False
    exclusion_reason: str | None = None
    protected_evidence: bool = False
    answer_eligible: bool = False
    answer_eligibility_reasons: list[str] = Field(default_factory=list)
    answer_term_coverage: float = Field(default=0.0, ge=0, le=1)


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
    web_search_audit: dict[str, Any] = Field(default_factory=dict)
    evidence_sufficiency: dict[str, Any] = Field(default_factory=dict)
    component_versions: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    provider_attempts: list[ProviderAttemptAudit] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class AnswerClaim(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    citation_ids: list[str] = Field(default_factory=list)


class AgentAnswer(BaseModel):
    facts: list[AnswerClaim] = Field(default_factory=list)
    hypotheses: list[AnswerClaim] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    confidence: str = Field(default="low", pattern="^(low|medium|high)$")


class EvidenceLedgerEntry(BaseModel):
    evidence_id: str
    source_type: str
    content: str
    chunk_id: str | None = None
    document_id: str | None = None
    revision: str | None = None
    approval_status: str | None = None
    effective_at: datetime | None = None
    source_uri: str = ""
    page_or_section: str = ""
    tool_id: str | None = None
    chamber: str | None = None
    recipe_version: str | None = None
    retrieval_routes: list[str] = Field(default_factory=list)
    retrieval_score: float | None = None
    rerank_score: float | None = None
    context_selection_reason: str | None = None
    image_ids: list[str] = Field(default_factory=list)
    external_url: str = ""
    source_title: str = ""
    source_domain: str = ""


class AnswerMode(StrEnum):
    STRUCTURED_INVESTIGATION = "structured_investigation"
    NATURAL_KNOWLEDGE = "natural_knowledge"


class MessageRenderMode(StrEnum):
    BUBBLE = "bubble"
    STRUCTURED_CARD = "structured_card"


class MessagePresentation(BaseModel):
    """Persisted UI projection selected by server-owned route policy."""

    mode: MessageRenderMode = MessageRenderMode.BUBBLE
    answer_mode: AnswerMode | None = None
    route_decision: str | None = None
    status: str | None = None
    answer: AgentAnswer | None = None
    trace_id: str | None = None
    verification_warnings: list[str] = Field(default_factory=list)
    task_results: list[TaskExecutionResult] = Field(default_factory=list, max_length=3)
    image_asset_ids: list[str] = Field(default_factory=list, max_length=64)
    evidence_ledger: list[EvidenceLedgerEntry] = Field(default_factory=list, max_length=64)


class ChatMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: new_id("msg"))
    request_id: str | None = None
    run_id: str | None = None
    turn_seq: int | None = Field(default=None, ge=1)
    role: str
    content: str
    created_at: datetime = Field(default_factory=utc_now)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    presentation: MessagePresentation | None = None


class ContextSlot(BaseModel):
    """A sourced conversation value that can be invalidated without deleting history."""

    value: str
    source_message_id: str
    source_kind: str = "explicit"
    depends_on: list[str] = Field(default_factory=list)
    valid: bool = True
    invalidated_by_message_id: str | None = None
    invalidation_reason: str | None = None
    updated_at: datetime = Field(default_factory=utc_now)


class ContextEvidenceRef(BaseModel):
    evidence_id: str
    source_type: str
    source_message_id: str
    trace_id: str | None = None
    valid: bool = True
    invalidated_by_message_id: str | None = None
    updated_at: datetime = Field(default_factory=utc_now)


class ActiveConversationContext(BaseModel):
    """Small, sourced working context; exact messages remain the audit authority."""

    topic: str | None = None
    slots: dict[str, ContextSlot] = Field(default_factory=dict)
    evidence_refs: list[ContextEvidenceRef] = Field(default_factory=list)
    trace_id: str | None = None
    updated_at: datetime = Field(default_factory=utc_now)


class ConversationContextMessage(BaseModel):
    message_id: str
    turn_seq: int = Field(ge=1)
    role: str
    content: str
    created_at: datetime


class AssembledConversationContext(BaseModel):
    """Bounded graph input assembled from one owned thread."""

    thread_id: str
    context_version: int = Field(default=1, ge=1)
    summary: str = ""
    summary_upto_message_id: str | None = None
    recent_messages: list[ConversationContextMessage] = Field(default_factory=list)
    active_context: ActiveConversationContext = Field(default_factory=ActiveConversationContext)
    approved_preferences: list[str] = Field(default_factory=list)
    current_message_id: str | None = None


class InteractionMode(StrEnum):
    TASK = "task"
    CONVERSATION = "conversation"
    FEEDBACK = "feedback"
    CONTROL = "control"
    CLARIFICATION_ANSWER = "clarification_answer"
    MIXED = "mixed"


class PrimaryIntent(StrEnum):
    CONVERSATION = "conversation"
    KNOWLEDGE_QUERY = "knowledge_query"
    INVESTIGATION = "investigation"
    DATA_QUERY = "data_query"
    ACTION_REQUEST = "action_request"
    CONTENT_TASK = "content_task"


class SemanticTemporalScope(StrEnum):
    UNSPECIFIED = "unspecified"
    TIMELESS = "timeless"
    RELATIVE = "relative"
    EXPLICIT = "explicit"
    CURRENT = "current"


class ExpectedOutput(StrEnum):
    UNSPECIFIED = "unspecified"
    EXPLANATION = "explanation"
    ENUMERATION = "enumeration"
    RECORDS = "records"
    RANKING = "ranking"
    TREND = "trend"
    DIAGNOSIS = "diagnosis"
    ACTION = "action"
    TRANSFORMATION = "transformation"
    CONVERSATION = "conversation"


class KnowledgeScope(StrEnum):
    UNSPECIFIED = "unspecified"
    PUBLIC_GENERAL = "public_general"
    INTERNAL_CONTROLLED = "internal_controlled"
    MIXED = "mixed"
    NOT_APPLICABLE = "not_applicable"


class EvidenceSufficiencyStatus(StrEnum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class EvidenceSufficiencyAssessment(BaseModel):
    schema_version: str = "semikb-evidence-sufficiency-v1"
    status: EvidenceSufficiencyStatus
    reason_codes: list[str] = Field(default_factory=list)
    selected_count: int = Field(default=0, ge=0)
    high_score_count: int = Field(default=0, ge=0)
    query_term_coverage: float = Field(default=0.0, ge=0, le=1)
    top_rerank_score: float | None = Field(default=None, ge=0)
    supported_aspects: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    knowledge_scope: KnowledgeScope = KnowledgeScope.UNSPECIFIED
    web_fallback_allowed: bool = False
    judge_source: str = "deterministic"
    provider: str | None = None
    model: str | None = None
    warning_codes: list[str] = Field(default_factory=list)
    answer_eligible_ids: list[str] = Field(default_factory=list)
    answer_rejected_ids: list[str] = Field(default_factory=list)
    answer_gate_profile: str = "semikb-answer-gate-v1"
    answer_score_threshold: float | None = Field(default=None, ge=0, le=1)


class TaskShape(StrEnum):
    UNSPECIFIED = "unspecified"
    DIRECT = "direct"
    CONCEPT_EXPLANATION = "concept_explanation"
    ENTITY_LOOKUP = "entity_lookup"
    AGGREGATE_RANKING = "aggregate_ranking"
    EVENT_LIST = "event_list"
    TREND_ANALYSIS = "trend_analysis"
    CAUSAL_INVESTIGATION = "causal_investigation"
    CONTENT_TRANSFORM = "content_transform"
    CONTROL = "control"


class GroupingDimension(StrEnum):
    PRODUCT = "product"
    TOOL = "tool"
    CHAMBER = "chamber"
    LOT = "lot"
    WAFER = "wafer"
    ALARM = "alarm"


class TaskGroundingSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quote: str = Field(min_length=1, max_length=1000)
    message_id: str | None = Field(default=None, max_length=128)


class SemanticFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temporal_scope: SemanticTemporalScope = SemanticTemporalScope.UNSPECIFIED
    expected_output: ExpectedOutput = ExpectedOutput.UNSPECIFIED
    knowledge_scope: KnowledgeScope = KnowledgeScope.UNSPECIFIED


class IntentTarget(StrEnum):
    PREVIOUS_USER_MESSAGE = "previous_user_message"
    PREVIOUS_ANSWER = "previous_answer"
    SOP = "sop"
    RECIPE = "recipe"
    FDC = "fdc"
    SPC = "spc"
    WAFER_MAP = "wafer_map"
    LOT = "lot"
    CASE = "case"
    ALARM = "alarm"
    REPORT = "report"
    GENERAL = "general"


class IntentTaskAction(StrEnum):
    RECALL = "recall"
    SUMMARIZE = "summarize"
    SIMPLIFY = "simplify"
    TRANSLATE = "translate"
    LOOKUP = "lookup"
    COMPARE = "compare"
    DIAGNOSE = "diagnose"
    EXPLAIN = "explain"
    EXECUTE = "execute"
    GENERATE = "generate"


class TaskExecutionDecision(StrEnum):
    EXECUTE = "execute"
    CLARIFY = "clarify"
    REFUSE = "refuse"
    DEFER = "defer"


class TaskExecutionStatus(StrEnum):
    COMPLETED = "completed"
    CLARIFY = "clarify"
    REFUSED = "refused"
    DEFERRED = "deferred"
    FAILED = "failed"


class SlotOperationKind(StrEnum):
    SET = "set"
    INHERIT = "inherit"
    CORRECT = "correct"
    CLEAR = "clear"


class CancelScope(StrEnum):
    CURRENT_GENERATION = "current_generation"
    CURRENT_TASK = "current_task"
    TASK_ITEM = "task_item"
    CLARIFICATION = "clarification"


class ClarificationKind(StrEnum):
    INTENT_DISAMBIGUATION = "intent_disambiguation"
    SLOT_COLLECTION = "slot_collection"
    HISTORY_REFERENCE = "history_reference"
    CONTROL_CONFIRMATION = "control_confirmation"


class ClarificationFrameStatus(StrEnum):
    WAITING = "waiting"
    PAUSED = "paused"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


class ClarificationItemType(StrEnum):
    SLOT = "slot"
    CHOICE = "choice"


class ClarificationTurnRelation(StrEnum):
    CONTINUE_CURRENT = "continue_current"
    CANCEL_CURRENT = "cancel_current"
    REPLACE_WITH_NEW_REQUEST = "replace_with_new_request"
    SIDE_CONVERSATION = "side_conversation"
    AMBIGUOUS = "ambiguous"


class AgentRoute(StrEnum):
    HISTORY_DIRECT = "history_direct"
    CHAT_DIRECT = "chat_direct"
    REUSE_EVIDENCE = "reuse_evidence"
    INTERNAL_RAG = "internal_rag"
    TOOL_ONLY = "tool_only"
    RAG_AND_TOOL = "rag_and_tool"
    RAG_AND_WEB = "rag_and_web"
    CLARIFY = "clarify"
    REFUSE = "refuse"


class AffectSignals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentiment: str = Field(default="neutral", pattern="^(neutral|positive|negative)$")
    urgency: str = Field(default="normal", pattern="^(normal|urgent)$")
    complaint_signal: bool = False


class IntentTaskItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(pattern=r"^task_[1-3]$")
    primary_intent: PrimaryIntent
    target_type: IntentTarget
    action: IntentTaskAction
    task_shape: TaskShape = TaskShape.UNSPECIFIED
    group_by: list[GroupingDimension] = Field(default_factory=list, max_length=3)
    supporting_spans: list[TaskGroundingSpan] = Field(default_factory=list, max_length=3)
    depends_on: list[str] = Field(default_factory=list, max_length=2)
    execution_policy: TaskExecutionDecision = TaskExecutionDecision.EXECUTE


class SlotOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: SlotOperationKind
    slot_name: str = Field(min_length=1, max_length=64)
    value: str | None = Field(default=None, max_length=256)
    source_message_id: str | None = None


class ConversationUnderstanding(BaseModel):
    """Validated semantic interpretation; it is not an executable tool plan."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "semikb-understanding-v1"
    classifier_source: str = Field(pattern="^(l0|llm|deterministic_fallback)$")
    interaction_mode: InteractionMode
    primary_intent: PrimaryIntent
    task_items: list[IntentTaskItem] = Field(default_factory=list, max_length=3)
    semantic_frame: SemanticFrame = Field(default_factory=SemanticFrame)
    affect: AffectSignals = Field(default_factory=AffectSignals)
    slot_operations: list[SlotOperation] = Field(default_factory=list, max_length=12)
    explicit_slots: dict[str, str] = Field(default_factory=dict)
    inherited_slots: dict[str, str] = Field(default_factory=dict)
    missing_slots: list[str] = Field(default_factory=list, max_length=3)
    context_message_ids: list[str] = Field(default_factory=list, max_length=8)
    standalone_query: str = Field(default="", max_length=8000)
    cancel_scope: CancelScope | None = None
    clarification_relation: ClarificationTurnRelation | None = None
    suggested_route: AgentRoute
    confidence: float = Field(ge=0, le=1)


class ClarificationPendingItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=64)
    item_type: ClarificationItemType
    prompt: str = Field(min_length=1, max_length=500)
    allowed_values: list[str] = Field(default_factory=list, max_length=12)


class ClarificationResolvedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=64)
    value: str = Field(min_length=1, max_length=512)
    source_message_id: str | None = Field(default=None, max_length=128)


class ClarificationFrame(BaseModel):
    """Versioned task-local truth for a bounded clarification lifecycle."""

    model_config = ConfigDict(extra="forbid")

    frame_id: str = Field(default_factory=lambda: new_id("clarify"), min_length=1, max_length=128)
    schema_version: Literal["semikb-clarification-frame-v1"] = "semikb-clarification-frame-v1"
    kind: ClarificationKind
    original_request: str = Field(min_length=1, max_length=8000)
    candidate_route: AgentRoute
    task_ids: list[str] = Field(default_factory=list, max_length=3)
    pending_items: list[ClarificationPendingItem] = Field(default_factory=list, max_length=3)
    resolved_items: list[ClarificationResolvedItem] = Field(default_factory=list, max_length=12)
    round: int = Field(default=0, ge=0, le=3)
    no_progress_count: int = Field(default=0, ge=0, le=3)
    signature: str = Field(min_length=8, max_length=128)
    status: ClarificationFrameStatus = ClarificationFrameStatus.WAITING
    last_transition: ClarificationTurnRelation | None = None
    last_source_message_id: str | None = Field(default=None, max_length=128)
    superseded_by_request_id: str | None = Field(default=None, max_length=128)
    base_understanding: dict[str, Any] = Field(default_factory=dict)
    base_route_plan: dict[str, Any] = Field(default_factory=dict)

    @property
    def pending_keys(self) -> list[str]:
        return [item.key for item in self.pending_items]


class ClarificationTransitionAudit(BaseModel):
    """Prompt-free request audit for one clarification state transition."""

    model_config = ConfigDict(extra="forbid")

    frame_id: str = Field(min_length=1, max_length=128)
    relation: ClarificationTurnRelation
    classifier_source: str = Field(pattern="^(l0|llm|deterministic_fallback|server_policy)$")
    previous_status: ClarificationFrameStatus
    next_status: ClarificationFrameStatus
    pending_before: list[str] = Field(default_factory=list, max_length=3)
    resolved_by_answer: list[str] = Field(default_factory=list, max_length=3)
    pending_after: list[str] = Field(default_factory=list, max_length=3)
    made_progress: bool = False
    warning_codes: list[str] = Field(default_factory=list, max_length=12)


class RouteTaskDecision(BaseModel):
    task_id: str
    decision: TaskExecutionDecision
    route: AgentRoute | None = None
    reason_code: str


class TaskExecutionResult(BaseModel):
    """Terminal, user-safe result for one planned task item."""

    task_id: str = Field(pattern=r"^task_[1-3]$")
    status: TaskExecutionStatus
    route: AgentRoute | None = None
    reason_code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(default_factory=list, max_length=64)
    tool_fact_ids: list[str] = Field(default_factory=list, max_length=32)
    external_evidence_ids: list[str] = Field(default_factory=list, max_length=32)
    validation_warnings: list[str] = Field(default_factory=list, max_length=16)


class RoutePlan(BaseModel):
    """Deterministic policy output persisted for audit and later controlled execution."""

    route: AgentRoute
    confidence: float = Field(ge=0, le=1)
    reason_codes: list[str] = Field(default_factory=list)
    task_decisions: list[RouteTaskDecision] = Field(default_factory=list, max_length=3)
    missing_slots: list[str] = Field(default_factory=list, max_length=3)
    retrieval_skipped_reason: str | None = None
    invalidated_context_refs: list[str] = Field(default_factory=list)


class ThreadRecord(BaseModel):
    thread_id: str = Field(default_factory=lambda: new_id("thread"))
    title: str = "New investigation"
    actor_scope: ActorScope = Field(default_factory=ActorScope)
    status: str = "active"
    summary: str = ""
    summary_upto_message_id: str | None = None
    context_version: int = Field(default=1, ge=1)
    active_context: ActiveConversationContext = Field(default_factory=ActiveConversationContext)
    next_turn_seq: int = Field(default=1, ge=1)
    last_turn_seq: int = Field(default=0, ge=0)
    active_request_id: str | None = None
    active_request_started_at: datetime | None = None
    clarification_round: int = 0
    pending_fields: list[str] = Field(default_factory=list)
    messages: list[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SendMessageResponse(BaseModel):
    """Canonical completed result shared by synchronous and streaming messages."""

    thread: ThreadRecord
    response: str
    clarification_required: bool
    missing_fields: list[str] = Field(default_factory=list)
    clarification_round: int | None = None
    status: str | None = None
    answer: AgentAnswer | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    trace_id: str | None = None
    image_asset_ids: list[str] = Field(default_factory=list)
    tool_facts: list[dict[str, Any]] = Field(default_factory=list)
    external_evidence: list[dict[str, Any]] = Field(default_factory=list)
    evidence_ledger: list[dict[str, Any]] = Field(default_factory=list)
    model_metadata: dict[str, Any] = Field(default_factory=dict)
    verification_warnings: list[str] = Field(default_factory=list)
    interaction_mode: InteractionMode | None = None
    route_decision: AgentRoute | None = None
    route_confidence: float | None = Field(default=None, ge=0, le=1)
    task_items: list[IntentTaskItem] = Field(default_factory=list, max_length=3)
    task_decisions: list[RouteTaskDecision] = Field(default_factory=list, max_length=3)
    task_results: list[TaskExecutionResult] = Field(default_factory=list, max_length=3)
    retrieval_skipped_reason: str | None = None
    answer_mode: AnswerMode | None = None


class MemoryRecord(BaseModel):
    memory_id: str = Field(default_factory=lambda: new_id("memory"))
    user_id: str
    memory_type: str = Field(pattern="^(preference|case_summary|stable_rule)$")
    content: str = Field(min_length=1, max_length=4000)
    scope: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0, le=1)
    approval_status: ApprovalStatus = ApprovalStatus.APPROVED
    expires_at: datetime | None = None
    created_by: str
    created_at: datetime = Field(default_factory=utc_now)


class AuditEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: new_id("audit"))
    event_type: str
    actor_user_id: str
    thread_id: str | None = None
    trace_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class EvaluationCase(BaseModel):
    case_id: str
    question: str
    expected_chunk_ids: list[str]
    expected_outcome: str = Field(default="evidence", pattern="^(evidence|no_evidence)$")
    actor_scope: ActorScope = Field(default_factory=ActorScope)
    tags: list[str] = Field(default_factory=list)
    failure_labels: list[str] = Field(default_factory=list)


class EvaluationDataset(BaseModel):
    dataset_version: str
    dataset_hash: str
    source_kind: str = "synthetic"
    description: str = ""
    purpose: EvaluationDatasetPurpose = EvaluationDatasetPurpose.REGRESSION
    sealed_at: datetime | None = None
    opened_at: datetime | None = None
    source_snapshot_hash: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    leakage_status: EvaluationLeakageStatus = EvaluationLeakageStatus.UNREVIEWED
    case_count: int = Field(ge=1)
    cases: list[EvaluationCase]
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_holdout_governance(self) -> EvaluationDataset:
        if self.opened_at and not self.sealed_at:
            raise ValueError("opened_at requires sealed_at.")
        if self.opened_at and self.sealed_at and self.opened_at < self.sealed_at:
            raise ValueError("opened_at cannot be earlier than sealed_at.")
        if self.purpose is EvaluationDatasetPurpose.HOLDOUT:
            if self.sealed_at is None:
                raise ValueError("A holdout dataset must be sealed before use.")
            if self.leakage_status is not EvaluationLeakageStatus.CLEARED:
                raise ValueError("A holdout dataset must pass leakage review before use.")
        return self


class EvaluationRun(BaseModel):
    evaluation_run_id: str = Field(default_factory=lambda: new_id("eval"))
    dataset_version: str
    dataset_hash: str = ""
    case_count: int = 0
    dataset_purpose: EvaluationDatasetPurpose = EvaluationDatasetPurpose.REGRESSION
    dataset_sealed_at: datetime | None = None
    dataset_opened_at: datetime | None = None
    dataset_leakage_status: EvaluationLeakageStatus = EvaluationLeakageStatus.UNREVIEWED
    source_snapshot_hash: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    release_freeze_id: str | None = None
    release_freeze_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    baseline_run_id: str | None = None
    requested_by: str = "system"
    status: EvaluationStatus = EvaluationStatus.QUEUED
    retrieval_profile: str = Field(
        default="full",
        pattern="^(dense|hybrid|reranked|full)$",
    )
    retrieval_config: dict[str, Any] = Field(default_factory=dict)
    component_versions: dict[str, str] = Field(default_factory=dict)
    aggregate_metrics: dict[str, float] = Field(default_factory=dict)
    baseline_comparison: dict[str, Any] = Field(default_factory=dict)
    case_results: list[dict[str, Any]] = Field(default_factory=list)
    failure_tags: list[str] = Field(default_factory=list)
    safe_error_summary: str | None = None
    worker_task_id: str | None = None
    attempt: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class CreateThreadRequest(BaseModel):
    title: str = "New investigation"
    actor_scope: ActorScope = Field(default_factory=ActorScope)


class SendMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


class CreateMemoryRequest(BaseModel):
    memory_type: str = Field(pattern="^(preference|case_summary|stable_rule)$")
    content: str = Field(min_length=1, max_length=4000)
    scope: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0, le=1)
    expires_at: datetime | None = None


class RetrievalConstraints(BaseModel):
    fab: str | None = None
    product: str | None = None
    process_layer: str | None = None
    tool_id: str | None = None
    chamber: str | None = None
    recipe_id: str | None = None
    recipe_version: str | None = None
    as_of: datetime | None = None
    use_hyde: bool | None = None
    retrieval_mode: RetrievalMode | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    actor_scope: ActorScope = Field(default_factory=ActorScope)
    thread_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    constraints: RetrievalConstraints | None = None


class IngestDocumentRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=120)
    revision: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=500)
    document_type: str = Field(min_length=1, max_length=64)
    content: str = Field(min_length=1)
    approval_status: ApprovalStatus = ApprovalStatus.DRAFT
    lifecycle: DocumentLifecycle = DocumentLifecycle.STAGED
    supersedes_revision: str | None = None
    source_kind: str = "user_upload"
    source_uri: str = ""
    source_license: str = "unknown"
    source_id: str | None = None
    source_manifest_version: str | None = None
    dataset_version: str | None = None
    source_license_status: SourceLicenseStatus | None = None
    redistribution_policy: RedistributionPolicy | None = None
    access_scope_key: str | None = None
    fab: str | None = None
    product: str | None = None
    process_layer: str | None = None
    tool_id: str | None = None
    chamber: str | None = None
    recipe_id: str | None = None
    recipe_version: str | None = None
    retrieval_policy: RetrievalPolicy = RetrievalPolicy.STANDARD
    source_index_artifacts: list[SourceIndexArtifact] = Field(default_factory=list)
    images: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source_manifest_link(self) -> IngestDocumentRequest:
        _validate_source_manifest_link(self.source_id, self.source_manifest_version)
        _validate_publication_request(self)
        return self


class IngestUploadMetadata(BaseModel):
    document_id: str = Field(min_length=1, max_length=120)
    revision: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=500)
    document_type: str = Field(min_length=1, max_length=64)
    approval_status: ApprovalStatus = ApprovalStatus.DRAFT
    lifecycle: DocumentLifecycle = DocumentLifecycle.STAGED
    supersedes_revision: str | None = None
    source_kind: str = "user_upload"
    source_uri: str = ""
    source_license: str = "unknown"
    source_id: str | None = None
    source_manifest_version: str | None = None
    dataset_version: str | None = None
    source_license_status: SourceLicenseStatus | None = None
    redistribution_policy: RedistributionPolicy | None = None
    access_scope_key: str | None = None
    fab: str | None = None
    product: str | None = None
    process_layer: str | None = None
    tool_id: str | None = None
    chamber: str | None = None
    recipe_id: str | None = None
    recipe_version: str | None = None
    retrieval_policy: RetrievalPolicy = RetrievalPolicy.STANDARD
    source_index_artifacts: list[SourceIndexArtifact] = Field(default_factory=list)
    images: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source_manifest_link(self) -> IngestUploadMetadata:
        _validate_source_manifest_link(self.source_id, self.source_manifest_version)
        _validate_publication_request(self)
        return self


class CreateEvaluationRunRequest(BaseModel):
    dataset_version: str = "demo-v2"
    retrieval_profile: str = Field(
        default="full",
        pattern="^(dense|hybrid|reranked|full)$",
    )
    baseline_run_id: str | None = None
