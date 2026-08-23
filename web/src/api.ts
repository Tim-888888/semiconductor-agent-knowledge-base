import type {
  AgentResponse,
  AgentStreamEvent,
  AssetAccess,
  CorpusStandardizationJob,
  CorpusStandardizationMetadata,
  CorpusPublicationBatch,
  CorpusPublicationReview,
  EvaluationDataset,
  EvaluationReleaseFreeze,
  EvaluationRun,
  IngestionJob,
  KnowledgeDocumentListResponse,
  KnowledgeDocumentRevisionSummary,
  DocumentLifecycleOperationRecord,
  RetrievalTrace,
  Thread,
  UploadMetadata
} from "./types";

const apiBase = "/api/v1";
let token = "";

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "ApiError";
  }
}

export class AgentStreamError extends Error {
  constructor(
    message: string,
    readonly code: string,
    readonly retryable: boolean
  ) {
    super(message);
    this.name = "AgentStreamError";
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  if (!(options.body instanceof FormData)) headers.set("Content-Type", "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${apiBase}${path}`, { ...options, headers });
  if (!response.ok) {
    const payload = await response.text();
    let detail = payload;
    try {
      const parsed = JSON.parse(payload) as {
        detail?: string | { message?: string };
      };
      detail = typeof parsed.detail === "string"
        ? parsed.detail
        : parsed.detail?.message ?? payload;
    } catch (error) {
      if (!(error instanceof SyntaxError)) throw error;
    }
    throw new ApiError(detail || `HTTP ${response.status}`, response.status);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export async function bootstrapToken(accessKey = ""): Promise<void> {
  const headers = new Headers();
  if (accessKey) headers.set("X-Demo-Access-Key", accessKey);
  const response = await request<{ access_token: string }>("/auth/demo-token", {
    method: "POST",
    headers
  });
  token = response.access_token;
}

export const createThread = (title: string) =>
  request<Thread>("/threads", { method: "POST", body: JSON.stringify({ title }) });
export const listThreads = () => request<Thread[]>("/threads");
export const getThread = (threadId: string) => request<Thread>(`/threads/${threadId}`);
export const sendMessage = (threadId: string, content: string) =>
  request<AgentResponse>(`/threads/${threadId}/messages`, {
    method: "POST",
    body: JSON.stringify({ content })
  });

export async function sendMessageStream(
  threadId: string,
  content: string,
  requestId: string,
  onEvent: (event: AgentStreamEvent) => void,
  signal?: AbortSignal
): Promise<AgentResponse> {
  const headers = new Headers({
    Accept: "text/event-stream",
    "Content-Type": "application/json"
  });
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${apiBase}/threads/${threadId}/messages/stream`, {
    method: "POST",
    headers,
    body: JSON.stringify({ content, request_id: requestId }),
    signal
  });
  if (!response.ok) {
    const payload = await response.text();
    let detail = payload;
    try {
      const parsed = JSON.parse(payload) as { detail?: string };
      detail = parsed.detail ?? payload;
    } catch (error) {
      if (!(error instanceof SyntaxError)) throw error;
    }
    throw new ApiError(detail || `HTTP ${response.status}`, response.status);
  }
  if (!response.body) throw new AgentStreamError("浏览器未收到响应流。", "empty_stream", true);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completed: AgentResponse | undefined;
  let terminalSeen = false;

  function consumeBlock(block: string) {
    const data = block
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n");
    if (!data) return;
    const event = JSON.parse(data) as AgentStreamEvent;
    onEvent(event);
    if (event.event === "completed") {
      completed = event.data.result;
      terminalSeen = true;
    }
    if (event.event === "error") {
      terminalSeen = true;
      throw new AgentStreamError(
        event.data.message ?? "流式请求失败。",
        event.data.code ?? "stream_error",
        Boolean(event.data.retryable)
      );
    }
  }

  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done }).replace(/\r\n/g, "\n");
      let boundary = buffer.indexOf("\n\n");
      while (boundary >= 0) {
        consumeBlock(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        boundary = buffer.indexOf("\n\n");
      }
      if (done) break;
    }
    if (buffer.trim()) consumeBlock(buffer);
  } finally {
    reader.releaseLock();
  }

  if (!terminalSeen || !completed) {
    throw new AgentStreamError("连接提前结束，最终结果未确认保存。", "incomplete_stream", true);
  }
  return completed;
}

export const cancelMessageRequest = (threadId: string, requestId: string, keepalive = false) =>
  request<{ status: string }>(`/threads/${threadId}/message-requests/${requestId}/cancel`, {
    method: "POST",
    keepalive
  });

export const listJobs = () => request<IngestionJob[]>("/ingestion-jobs");
export const getJob = (jobId: string) => request<IngestionJob>(`/ingestion-jobs/${jobId}`);
export const retryJob = (jobId: string) =>
  request<IngestionJob>(`/ingestion-jobs/${jobId}/retry`, { method: "POST" });
export async function uploadDocument(file: File, metadata: UploadMetadata): Promise<IngestionJob> {
  const body = new FormData();
  body.set("file", file);
  body.set("metadata", JSON.stringify(metadata));
  return request<IngestionJob>("/ingestion-jobs/upload", { method: "POST", body });
}

export const listCorpusStandardizationJobs = () =>
  request<CorpusStandardizationJob[]>("/corpus-standardization-jobs");
export const getCorpusStandardizationJob = (jobId: string) =>
  request<CorpusStandardizationJob>(`/corpus-standardization-jobs/${jobId}`);
export const retryCorpusStandardizationJob = (jobId: string) =>
  request<CorpusStandardizationJob>(`/corpus-standardization-jobs/${jobId}/retry`, {
    method: "POST"
  });
export async function uploadCorpus(
  files: File[],
  metadata: CorpusStandardizationMetadata,
  sidecar?: string
): Promise<CorpusStandardizationJob> {
  const body = new FormData();
  files.forEach((file) => {
    body.append("files", file, file.webkitRelativePath || file.name);
  });
  body.set("metadata", JSON.stringify(metadata));
  if (sidecar?.trim()) body.set("sidecar", sidecar.trim());
  return request<CorpusStandardizationJob>("/corpus-standardization-jobs/upload", {
    method: "POST",
    body
  });
}

export const listCorpusPublicationBatches = () =>
  request<CorpusPublicationBatch[]>("/corpus-publication-batches");
export const getCorpusPublicationBatch = (batchId: string) =>
  request<CorpusPublicationBatch>(`/corpus-publication-batches/${batchId}`);
export const publishCorpus = (review: CorpusPublicationReview) =>
  request<CorpusPublicationBatch>("/corpus-publication-batches", {
    method: "POST",
    body: JSON.stringify(review)
  });
export const retryCorpusPublication = (batchId: string) =>
  request<CorpusPublicationBatch>(`/corpus-publication-batches/${batchId}/retry`, {
    method: "POST"
  });

export function listKnowledgeDocuments(input: {
  query?: string;
  lifecycle?: string;
  limit?: number;
  offset?: number;
} = {}): Promise<KnowledgeDocumentListResponse> {
  const params = new URLSearchParams();
  if (input.query?.trim()) params.set("query", input.query.trim());
  if (input.lifecycle) params.set("lifecycle", input.lifecycle);
  params.set("limit", String(input.limit ?? 100));
  params.set("offset", String(input.offset ?? 0));
  return request<KnowledgeDocumentListResponse>(`/knowledge-documents?${params.toString()}`);
}

export const listKnowledgeDocumentRevisions = (documentId: string) =>
  request<KnowledgeDocumentRevisionSummary[]>(
    `/knowledge-documents/${encodeURIComponent(documentId)}/revisions`
  );

export const withdrawKnowledgeDocumentRevision = (
  documentId: string,
  revision: string,
  reason: string,
  requestId: string
) => request<DocumentLifecycleOperationRecord>(
  `/knowledge-documents/${encodeURIComponent(documentId)}/revisions/${encodeURIComponent(revision)}/withdraw`,
  {
    method: "POST",
    body: JSON.stringify({ request_id: requestId, reason })
  }
);

export const restoreKnowledgeDocumentRevision = (
  documentId: string,
  revision: string,
  reason: string,
  requestId: string
) => request<DocumentLifecycleOperationRecord>(
  `/knowledge-documents/${encodeURIComponent(documentId)}/revisions/${encodeURIComponent(revision)}/restore`,
  {
    method: "POST",
    body: JSON.stringify({ request_id: requestId, reason })
  }
);

export const getKnowledgeDocumentOperation = (operationId: string) =>
  request<DocumentLifecycleOperationRecord>(
    `/knowledge-document-operations/${encodeURIComponent(operationId)}`
  );

export const retryKnowledgeDocumentOperation = (operationId: string) =>
  request<DocumentLifecycleOperationRecord>(
    `/knowledge-document-operations/${encodeURIComponent(operationId)}/retry`,
    { method: "POST" }
  );

export const listTraces = () => request<RetrievalTrace[]>("/retrieval-traces");
export const getTrace = (traceId: string) =>
  request<RetrievalTrace>(`/retrieval-traces/${traceId}`);

export const listEvaluationDatasets = () =>
  request<EvaluationDataset[]>("/evaluation-datasets");
export const listEvaluationReleaseFreezes = () =>
  request<EvaluationReleaseFreeze[]>("/evaluation-release-freezes");
export const listEvaluations = () => request<EvaluationRun[]>("/evaluation-runs");
export const getEvaluation = (runId: string) =>
  request<EvaluationRun>(`/evaluation-runs/${runId}`);
export const runEvaluation = (input: {
  dataset_version: string;
  retrieval_profile: EvaluationRun["retrieval_profile"];
  baseline_run_id?: string;
}) =>
  request<EvaluationRun>("/evaluation-runs", {
    method: "POST",
    body: JSON.stringify(input)
  });
export const retryEvaluation = (runId: string) =>
  request<EvaluationRun>(`/evaluation-runs/${runId}/retry`, { method: "POST" });
export const getEvaluationCaseTrace = (runId: string, caseId: string) =>
  request<RetrievalTrace>(`/evaluation-runs/${runId}/cases/${caseId}/trace`);

export const getAsset = (imageId: string) =>
  request<AssetAccess>(`/assets/${imageId}/access`);

export async function resolveAsset(imageId: string): Promise<AssetAccess> {
  const access = await getAsset(imageId);
  if (!access.url.startsWith("/")) return access;
  const headers = new Headers();
  if (token && !access.url.startsWith("/objects/")) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  const response = await fetch(access.url, { headers });
  if (!response.ok) throw new Error("图片预览访问失败。");
  const blob = await response.blob();
  return { ...access, url: URL.createObjectURL(blob), local_object_url: true };
}
