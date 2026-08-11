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
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  created_at: string;
};

export type Thread = {
  thread_id: string;
  title: string;
  actor_scope: ActorScope;
  status: string;
  summary: string;
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
  source_kind: string;
  source_license: string;
  access_scope_key: string;
  fab: string;
  product: string;
  process_layer?: string;
  tool_id?: string;
  chamber?: string;
  recipe_id?: string;
  recipe_version?: string;
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

export type EvaluationDataset = {
  dataset_version: string;
  dataset_hash: string;
  source_kind: string;
  description: string;
  case_count: number;
  created_at: string;
};

export type AssetAccess = {
  image_id: string;
  url: string;
  expires_at: string;
  object_key: string;
  local_object_url?: boolean;
};
