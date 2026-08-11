export type Message = {
  message_id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  created_at: string;
};

export type Citation = {
  chunk_id: string;
  document_id: string;
  revision: string;
  page_or_section: string;
  image_ids: string[];
};

export type Thread = {
  thread_id: string;
  title: string;
  summary: string;
  messages: Message[];
  pending_fields: string[];
};

export type IngestionJob = {
  job_id: string;
  document_id: string;
  revision: string;
  filename: string;
  status: string;
  current_stage: string;
  progress: number;
  chunks_count: number;
  images_count: number;
  safe_error_summary?: string;
  events: { event_id: string; stage: string; message: string; created_at: string }[];
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
  rrf_score: number;
  rerank_score: number;
  selected: boolean;
  exclusion_reason?: string;
};

export type RetrievalTrace = {
  trace_id: string;
  original_query: string;
  routes: string[];
  cutoff_reason: string;
  final_evidence_ids: string[];
  image_asset_ids: string[];
  candidates: TraceCandidate[];
  timings_ms: Record<string, number>;
  created_at: string;
};

export type EvaluationRun = {
  evaluation_run_id: string;
  dataset_version: string;
  status: string;
  aggregate_metrics: Record<string, number>;
  case_results: { case_id: string; recall_at_5: number; reciprocal_rank: number; trace_id: string }[];
  created_at: string;
};
