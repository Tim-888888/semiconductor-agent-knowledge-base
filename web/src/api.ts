import type { EvaluationRun, IngestionJob, RetrievalTrace, Thread } from "./types";

const apiBase = "/api/v1";
let token = "";

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  headers.set("Content-Type", "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${apiBase}${path}`, { ...options, headers });
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<T>;
}

export async function bootstrapToken(): Promise<void> {
  const payload = {
    user_id: "demo_engineer",
    roles: ["engineer", "knowledge_admin"],
    access_scope_keys: ["demo_engineering"],
    fabs: ["FAB-01"],
    products: ["P-ALPHA"],
    tool_ids: ["ETCH-03"]
  };
  const response = await request<{ access_token: string }>("/auth/demo-token", {
    method: "POST",
    body: JSON.stringify(payload)
  });
  token = response.access_token;
}

export const createThread = (title: string) => request<Thread>("/threads", { method: "POST", body: JSON.stringify({ title }) });
export const getThread = (threadId: string) => request<Thread>(`/threads/${threadId}`);
export const sendMessage = (threadId: string, content: string) => request<{ thread: Thread; response: string; trace_id?: string; image_asset_ids: string[] }>(`/threads/${threadId}/messages`, { method: "POST", body: JSON.stringify({ content }) });
export const listJobs = () => request<IngestionJob[]>("/ingestion-jobs");
export const listTraces = () => request<RetrievalTrace[]>("/retrieval-traces");
export const listEvaluations = () => request<EvaluationRun[]>("/evaluation-runs");
export const runEvaluation = () => request<EvaluationRun>("/evaluation-runs", { method: "POST", body: JSON.stringify({ dataset_version: "demo-v1" }) });
export const getAsset = (imageId: string) => request<{ url: string }>(`/assets/${imageId}/access`);

export async function createDemoDocument(): Promise<IngestionJob> {
  return request<IngestionJob>("/ingestion-jobs", {
    method: "POST",
    body: JSON.stringify({
      document_id: "SOP-ETCH-03-TRAINING",
      revision: "R1",
      title: "ETCH-03 首片训练补充说明",
      document_type: "training_note",
      content: "# 首片训练说明\n\n当 ETCH-03 Chamber B 出现首片异常时，先核对 SOP、FDC 报警与最近 Recipe 变更。",
      tool_id: "ETCH-03",
      chamber: "B",
      process_layer: "ETCH"
    })
  });
}
