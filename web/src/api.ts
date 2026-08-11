import type {
  AgentResponse,
  AssetAccess,
  EvaluationDataset,
  EvaluationRun,
  IngestionJob,
  RetrievalTrace,
  Thread,
  UploadMetadata
} from "./types";

const apiBase = "/api/v1";
let token = "";

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  if (!(options.body instanceof FormData)) headers.set("Content-Type", "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${apiBase}${path}`, { ...options, headers });
  if (!response.ok) {
    const payload = await response.text();
    try {
      const parsed = JSON.parse(payload) as { detail?: string };
      throw new Error(parsed.detail ?? payload);
    } catch (error) {
      if (error instanceof SyntaxError) throw new Error(payload || `HTTP ${response.status}`);
      throw error;
    }
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export async function bootstrapToken(): Promise<void> {
  const response = await request<{ access_token: string }>("/auth/demo-token", {
    method: "POST",
    body: JSON.stringify({
      user_id: "demo_engineer",
      roles: ["engineer", "knowledge_admin"],
      access_scope_keys: ["demo_engineering"],
      fabs: ["FAB-01"],
      products: ["P-ALPHA"],
      tool_ids: ["ETCH-03"]
    })
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

export const listTraces = () => request<RetrievalTrace[]>("/retrieval-traces");
export const getTrace = (traceId: string) =>
  request<RetrievalTrace>(`/retrieval-traces/${traceId}`);

export const listEvaluationDatasets = () =>
  request<EvaluationDataset[]>("/evaluation-datasets");
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
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(access.url, { headers });
  if (!response.ok) throw new Error("图片预览访问失败。");
  const blob = await response.blob();
  return { ...access, url: URL.createObjectURL(blob), local_object_url: true };
}
