export type ActorScope = {
  user_id: string;
  roles: string[];
  access_scope_keys: string[];
  fabs: string[];
  products: string[];
  tool_ids: string[];
};

export type Citation = {
  evidence_id?: string;
  source_type?: string;
  chunk_id?: string | null;
  document_id?: string | null;
  revision?: string | null;
  page_or_section?: string;
  image_ids?: string[];
};

export type Message = {
  message_id: string;
  request_id?: string | null;
  run_id?: string | null;
  turn_seq?: number | null;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  presentation?: MessagePresentation | null;
  created_at: string;
};

export type Thread = {
  thread_id: string;
  title: string;
  actor_scope: ActorScope;
  status: string;
  summary: string;
  summary_upto_message_id?: string | null;
  context_version: number;
  active_context: {
    topic?: string | null;
    slots: Record<string, {
      value: string;
      source_message_id: string;
      source_kind: string;
      depends_on: string[];
      valid: boolean;
    }>;
    evidence_refs: Array<Record<string, unknown>>;
    trace_id?: string | null;
  };
  next_turn_seq: number;
  last_turn_seq: number;
  clarification_round: number;
  messages: Message[];
  pending_fields: string[];
  created_at: string;
  updated_at: string;
};

export type AnswerClaim = {
  text: string;
  citation_ids: string[];
};

export type AgentAnswer = {
  facts: AnswerClaim[];
  hypotheses: AnswerClaim[];
  unknowns: string[];
  next_actions: string[];
  confidence: "low" | "medium" | "high";
};

export type AgentRoute =
  | "history_direct"
  | "chat_direct"
  | "reuse_evidence"
  | "internal_rag"
  | "tool_only"
  | "rag_and_tool"
  | "rag_and_web"
  | "clarify"
  | "refuse";

export type MessagePresentation = {
  mode: "bubble" | "structured_card";
  route_decision?: AgentRoute | string | null;
  status?: string | null;
  answer?: AgentAnswer | null;
  trace_id?: string | null;
  verification_warnings: string[];
  task_results: TaskExecutionResult[];
  image_asset_ids: string[];
  evidence_ledger: EvidenceLedgerEntry[];
};

export type IntentTaskItem = {
  task_id: string;
  primary_intent: string;
  target_type: string;
  action: string;
  depends_on: string[];
  execution_policy: "execute" | "clarify" | "refuse" | "defer";
};

export type RouteTaskDecision = {
  task_id: string;
  decision: "execute" | "clarify" | "refuse" | "defer";
  route?: AgentRoute | null;
  reason_code: string;
};

export type TaskExecutionStatus = "completed" | "clarify" | "refused" | "deferred" | "failed";

export type TaskExecutionResult = {
  task_id: string;
  status: TaskExecutionStatus;
  route?: AgentRoute | null;
  reason_code: string;
  message: string;
  evidence_ids: string[];
  tool_fact_ids: string[];
  external_evidence_ids: string[];
  validation_warnings: string[];
};

export type EvidenceLedgerEntry = {
  evidence_id: string;
  source_type: string;
  content: string;
  chunk_id?: string | null;
  document_id?: string | null;
  revision?: string | null;
  approval_status?: string | null;
  effective_at?: string | null;
  source_uri?: string;
  page_or_section?: string;
  tool_id?: string | null;
  chamber?: string | null;
  recipe_version?: string | null;
  retrieval_routes: string[];
  retrieval_score?: number | null;
  rerank_score?: number | null;
  context_selection_reason?: string | null;
  image_ids: string[];
  external_url?: string;
};

export type AgentResponse = {
  thread: Thread;
  response: string;
  clarification_required: boolean;
  missing_fields?: string[];
  clarification_round?: number;
  status?: string;
  answer?: AgentAnswer;
  citations?: Citation[];
  trace_id?: string | null;
  image_asset_ids?: string[];
  tool_facts?: Record<string, unknown>[];
  external_evidence?: Record<string, unknown>[];
  evidence_ledger?: EvidenceLedgerEntry[];
  model_metadata?: Record<string, unknown>;
  verification_warnings?: string[];
  interaction_mode?: string | null;
  route_decision?: AgentRoute | null;
  route_confidence?: number | null;
  task_items?: IntentTaskItem[];
  task_decisions?: RouteTaskDecision[];
  task_results?: TaskExecutionResult[];
  retrieval_skipped_reason?: string | null;
};

export type AgentStreamStage =
  | "analyzing_request"
  | "routing_request"
  | "awaiting_clarification"
  | "retrieving_evidence"
  | "searching_external"
  | "reranking_evidence"
  | "generating_answer"
  | "verifying_answer"
  | "persisting_result";

export type AgentStreamEvent = {
  event_id: string;
  request_id: string;
  thread_id: string;
  sequence: number;
  emitted_at: string;
  event: "accepted" | "stage" | "task_status" | "evidence" | "answer_delta" | "heartbeat" | "completed" | "error";
  data: Record<string, unknown> & {
    message_id?: string;
    run_id?: string;
    attempt?: number;
    replayed?: boolean;
    stage?: AgentStreamStage;
    message?: string;
    task_id?: string;
    status?: TaskProgressStatus;
    route?: AgentRoute | null;
    trace_id?: string | null;
    evidence_ids?: string[];
    image_asset_ids?: string[];
    internal_count?: number;
    external_count?: number;
    delta?: string;
    provider?: string | null;
    model?: string | null;
    elapsed_ms?: number;
    result?: AgentResponse;
    code?: string;
    retryable?: boolean;
  };
};

export type TaskProgressStatus = "queued" | "running" | TaskExecutionStatus;

export type TaskProgress = {
  taskId: string;
  status: TaskProgressStatus;
  route?: AgentRoute | null;
  message: string;
};

export type StreamUiState = {
  requestId: string;
  content: string;
  status: "running" | "stopped" | "error";
  stage?: AgentStreamStage;
  stageMessage: string;
  partialAnswer: string;
  internalEvidenceCount: number;
  externalEvidenceCount: number;
  elapsedMs: number;
  retryable: boolean;
  tasks: TaskProgress[];
};

export type IngestionEvent = {
  event_id: string;
  stage: string;
  message: string;
  attempt: number;
  progress: number;
  created_at: string;
};

export type IngestionJob = {
  job_id: string;
  document_id: string;
  revision: string;
  filename: string;
  file_type: string;
  source_hash: string;
  status: string;
  current_stage: string;
  progress: number;
  attempt: number;
  parser_version: string;
  chunker_version: string;
  embedding_version: string;
  index_version: string;
  chunks_count: number;
  images_count: number;
  tables_count: number;
  error_code?: string | null;
  safe_error_summary?: string | null;
  failed_stage?: string | null;
  created_by: string;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  events: IngestionEvent[];
};

export type UploadMetadata = {
  document_id: string;
  revision: string;
  title: string;
  document_type: string;
  approval_status: "draft" | "approved" | "rejected";
  lifecycle: "staged" | "published" | "quarantined";
  source_kind: string;
  source_license: string;
  source_id?: string;
  source_manifest_version?: string;
  dataset_version?: string;
  source_license_status?: SourceLicenseStatus;
  redistribution_policy?: RedistributionPolicy;
  access_scope_key?: string;
  fab?: string;
  product?: string;
  process_layer?: string;
  tool_id?: string;
  chamber?: string;
  recipe_id?: string;
  recipe_version?: string;
  retrieval_policy: "standard" | "protected";
};

export type CorpusStandardizationMetadata = {
  metadata_schema_version?: "semikb-corpus-metadata-v1";
  corpus_id: string;
  snapshot_version: string;
  display_name: string;
  source_kind: string;
  source_uri?: string;
  source_license: string;
  use_restrictions?: string;
  access_scope_key?: string;
  corpus_kind: "auto" | "document_collection" | "tabular_dataset" | "image_corpus" | "mixed";
};

export type CorpusFileManifest = {
  file_id: string;
  relative_path: string;
  sha256: string;
  size_bytes: number;
  content_type: string;
  role: "document" | "table" | "image" | "label" | "archive" | "unsupported";
  source_format?: string | null;
  parser_name?: string | null;
  parser_version?: string | null;
  warning_codes: string[];
  description: string;
  standardized_ref?: { bucket: string; object_key: string } | null;
  tabular_profile?: {
    sheets: Array<{
      name: string;
      observed_rows: number;
      column_count: number;
      sample_truncated: boolean;
      columns_truncated: boolean;
    }>;
    raw_rows_vectorized: false;
  } | null;
};

export type CorpusStandardizationJob = {
  job_id: string;
  metadata: CorpusStandardizationMetadata;
  snapshot_hash: string;
  status: "queued" | "validating" | "snapshotting" | "inventorying" | "standardizing" | "review_required" | "failed";
  progress: number;
  attempt: number;
  files_count: number;
  documents_count: number;
  tables_count: number;
  images_count: number;
  unsupported_count: number;
  report?: {
    inferred_corpus_kind: string;
    warning_codes: string[];
    review_reasons: string[];
    files: CorpusFileManifest[];
    relations: Array<{
      relation_id: string;
      from_file_id: string;
      to_file_id: string;
      relation_type: string;
    }>;
  } | null;
  error_code?: string | null;
  safe_error_summary?: string | null;
  created_by: string;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  events: Array<{
    event_id: string;
    status: string;
    message: string;
    progress: number;
    attempt: number;
    created_at: string;
  }>;
};

export type CorpusPublicationReview = {
  request_id: string;
  standardization_job_id: string;
  expected_snapshot_hash: string;
  selected_file_ids: string[];
  acknowledged_warning_codes: string[];
  source_type: SourceManifestType;
  content_origin: SourceContentOrigin;
  source_url: string;
  license_name: string;
  license_status: "verified" | "declared";
  redistribution_policy: "allowed" | "restricted";
  license_notes: string;
  access_scope_key: string;
  review_note: string;
  retrieval_policy: "standard" | "protected";
};

export type CorpusPublicationBatch = {
  batch_id: string;
  review: CorpusPublicationReview;
  status: "queued" | "preflight" | "publishing" | "reconciling" | "completed" | "failed";
  progress: number;
  attempt: number;
  published_count: number;
  failed_count: number;
  error_code?: string | null;
  safe_error_summary?: string | null;
  created_at: string;
  finished_at?: string | null;
  items: Array<{
    file_id: string;
    relative_path: string;
    artifact_kind: "document" | "tabular_profile" | "image_text";
    document_id: string;
    revision: string;
    status: "pending" | "publishing" | "published" | "failed";
    ingestion_job_id?: string | null;
    error_code?: string | null;
    safe_error_summary?: string | null;
    reconciliation?: {
      document_count: number;
      chunk_count: number;
      image_count: number;
      table_count: number;
      vector_count: number;
      object_count: number;
      passed: boolean;
      warning_codes: string[];
    } | null;
  }>;
};

export type TraceCandidate = {
  chunk_id: string;
  document_id: string;
  revision: string;
  title: string;
  page_or_section: string;
  routes: string[];
  dense_score: number;
  sparse_score: number;
  hyde_score: number;
  rrf_score: number;
  rerank_score: number;
  route_ranks: Record<string, number>;
  context_selection_reason?: string | null;
  selected: boolean;
  exclusion_reason?: string | null;
  protected_evidence: boolean;
};

export type ProviderAttemptAudit = {
  schema_version: "semikb-provider-attempt-v1";
  provider: string;
  operation: string;
  attempt: number;
  max_attempts: number;
  outcome: "succeeded" | "retrying" | "failed";
  failure_kind?: string | null;
  status_code?: number | null;
  retryable: boolean;
  retry_after_seconds?: number | null;
  latency_ms: number;
};

export type RetrievalTrace = {
  trace_id: string;
  thread_id?: string | null;
  actor_user_id: string;
  access_scope_keys: string[];
  original_query: string;
  rewritten_query?: string | null;
  hyde_query?: string | null;
  metadata_filters: Record<string, unknown>;
  routes: string[];
  cutoff_reason: string;
  final_evidence_ids: string[];
  image_asset_ids: string[];
  external_evidence: Record<string, unknown>[];
  candidates: TraceCandidate[];
  component_versions: Record<string, string>;
  warnings: string[];
  provider_attempts: ProviderAttemptAudit[];
  timings_ms: Record<string, number>;
  created_at: string;
};

export type BaselineMetric = {
  current: number;
  baseline: number;
  delta: number;
  direction: "higher_better" | "lower_better";
  outcome: "improved" | "regressed" | "unchanged";
};

export type EvaluationCaseResult = {
  case_id: string;
  question: string;
  tags: string[];
  expected_chunk_ids: string[];
  expected_outcome: "evidence" | "no_evidence";
  actual_chunk_ids: string[];
  missing_expected_chunk_ids: string[];
  unexpected_chunk_ids: string[];
  passed: boolean;
  recall_at_5: number;
  reciprocal_rank: number;
  ndcg_at_5: number;
  trace_id: string;
  routes: string[];
  cutoff_reason: string;
  image_asset_ids: string[];
  warnings: string[];
  latency_ms: number;
  failure_tags: string[];
  baseline_passed?: boolean;
  baseline_outcome?: string;
};

export type EvaluationRun = {
  evaluation_run_id: string;
  dataset_version: string;
  dataset_hash: string;
  case_count: number;
  dataset_purpose: "development" | "calibration" | "regression" | "holdout";
  dataset_sealed_at?: string | null;
  dataset_opened_at?: string | null;
  dataset_leakage_status: "unreviewed" | "cleared" | "contaminated";
  source_snapshot_hash?: string | null;
  release_freeze_id?: string | null;
  release_freeze_hash?: string | null;
  baseline_run_id?: string | null;
  requested_by: string;
  status: string;
  retrieval_profile: "dense" | "hybrid" | "reranked" | "full";
  retrieval_config: Record<string, unknown>;
  component_versions: Record<string, string>;
  aggregate_metrics: Record<string, number>;
  baseline_comparison: Record<string, BaselineMetric>;
  case_results: EvaluationCaseResult[];
  failure_tags: string[];
  safe_error_summary?: string | null;
  attempt: number;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
};

export type EvaluationReleaseFreeze = {
  freeze_id: string;
  release_version: string;
  source_commit: string;
  publication_batch_ids: string[];
  dataset_hashes: Record<string, string>;
  holdout_dataset_version: string;
  holdout_dataset_hash: string;
  retrieval_config: Record<string, unknown>;
  component_versions: Record<string, string>;
  freeze_hash: string;
  status: "frozen" | "opened";
  created_by: string;
  notes: string;
  created_at: string;
  opened_at?: string | null;
};

export type EvaluationDataset = {
  dataset_version: string;
  dataset_hash: string;
  source_kind: string;
  description: string;
  purpose: "development" | "calibration" | "regression" | "holdout";
  sealed_at?: string | null;
  opened_at?: string | null;
  source_snapshot_hash?: string | null;
  leakage_status: "unreviewed" | "cleared" | "contaminated";
  case_count: number;
  created_at: string;
};

export type SourceManifestStatus = "draft" | "approved" | "retired";
export type SourceManifestType =
  | "dataset"
  | "paper"
  | "repository"
  | "ontology"
  | "documentation"
  | "curated_corpus"
  | "other";
export type SourceContentOrigin = "real" | "synthetic" | "derived";
export type SourceLicenseStatus = "verified" | "declared" | "unclear" | "restricted";
export type RedistributionPolicy = "allowed" | "restricted" | "prohibited" | "unknown";
export type SourceIngestionMode =
  | "document_rag"
  | "tabular_profile_and_tool"
  | "image_corpus"
  | "mixed_curated_corpus"
  | "reference_only";
export type SourceIndexArtifact =
  | "document_chunks"
  | "data_dictionary"
  | "dataset_profile"
  | "analysis_report"
  | "image_text";

export type SourceIngestionPolicy = {
  mode: SourceIngestionMode;
  raw_storage: "minio_private";
  raw_row_vectorization: false;
  index_artifacts: SourceIndexArtifact[];
  analysis_tool_required: boolean;
};

export type SourceManifest = {
  manifest_schema_version: "semikb-source-manifest-v1";
  source_id: string;
  manifest_version: string;
  status: SourceManifestStatus;
  title: string;
  source_type: SourceManifestType;
  source_url: string;
  doi_or_repo?: string | null;
  retrieved_at: string;
  source_hash: string;
  hash_scope: string;
  content_origin: SourceContentOrigin;
  license_name: string;
  license_status: SourceLicenseStatus;
  redistribution_policy: RedistributionPolicy;
  license_notes: string;
  ingestion_policy: SourceIngestionPolicy;
  parser_hint?: string | null;
  expected_assets: {
    raw_files_min: number;
    documents_min: number;
    images_min: number;
    tables_min: number;
    records_estimate?: number | null;
    expected_formats: string[];
  };
  dataset_version: string;
  access_scope_key: string;
  source_snapshot_ref?: {
    bucket: string;
    object_key: string;
    content_type: string;
    sha256: string;
    version_id?: string | null;
  } | null;
  supersedes_manifest_version?: string | null;
  created_by: string;
  created_at: string;
  notes: string;
};

export type DocumentLifecycle =
  | "staged"
  | "published"
  | "superseded"
  | "expired"
  | "quarantined"
  | "withdrawn";
export type DocumentLifecycleAction = "withdraw" | "restore";
export type DocumentLifecycleOperationStatus =
  | "requested"
  | "blocking"
  | "vector_cleanup"
  | "withdrawn"
  | "restore_validating"
  | "restore_indexing"
  | "restored"
  | "compensation_required"
  | "failed";

export type AffectedRecordCounts = {
  documents: number;
  chunks: number;
  images: number;
  tables: number;
  vectors: number;
};

export type KnowledgeDocumentRevisionSummary = {
  document_id: string;
  revision: string;
  title: string;
  document_type: string;
  approval_status: "draft" | "approved" | "rejected";
  lifecycle: DocumentLifecycle;
  effective_at: string;
  expires_at?: string | null;
  source_id?: string | null;
  source_manifest_version?: string | null;
  dataset_version?: string | null;
  source_uri: string;
  source_license: string;
  source_license_status?: SourceLicenseStatus | null;
  redistribution_policy?: RedistributionPolicy | null;
  access_scope_key: string;
  counts: AffectedRecordCounts;
  created_at: string;
};

export type KnowledgeDocumentSummary = {
  document_id: string;
  title: string;
  document_type: string;
  current_revision?: string | null;
  current_lifecycle?: DocumentLifecycle | null;
  revision_count: number;
  source_id?: string | null;
  dataset_version?: string | null;
  updated_at: string;
};

export type KnowledgeDocumentListResponse = {
  items: KnowledgeDocumentSummary[];
  total: number;
  limit: number;
  offset: number;
};

export type WithdrawDocumentRevisionRequest = {
  request_id: string;
  reason: string;
};

export type RestoreDocumentRevisionRequest = {
  request_id: string;
  reason: string;
  target_index_version?: string | null;
};

export type DocumentLifecycleOperationRecord = {
  operation_id: string;
  request_id: string;
  action: DocumentLifecycleAction;
  status: DocumentLifecycleOperationStatus;
  selector: { document_id: string; revision: string };
  actor_user_id: string;
  reason: string;
  before_lifecycle: DocumentLifecycle;
  after_lifecycle?: DocumentLifecycle | null;
  target_index_version?: string | null;
  affected: AffectedRecordCounts;
  compensation_status: "not_required" | "pending" | "running" | "completed" | "failed";
  warning_codes: string[];
  created_at: string;
  updated_at: string;
  completed_at?: string | null;
};

export type AssetAccess = {
  image_id: string;
  url: string;
  expires_at: string;
  object_key: string;
  local_object_url?: boolean;
};
