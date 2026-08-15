import { type FormEvent, type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ClipboardCheck,
  Image as ImageIcon,
  KeyRound,
  LoaderCircle,
  MessageSquareText,
  SearchCheck,
  Upload,
  X
} from "lucide-react";
import {
  ApiError,
  AgentStreamError,
  bootstrapToken,
  cancelMessageRequest,
  createThread,
  getEvaluation,
  getEvaluationCaseTrace,
  getJob,
  getThread,
  getTrace,
  listEvaluationDatasets,
  listEvaluations,
  listJobs,
  listThreads,
  listTraces,
  resolveAsset,
  retryEvaluation,
  retryJob,
  runEvaluation,
  sendMessageStream,
  uploadDocument
} from "./api";
import { EvaluationPanel } from "./components/EvaluationPanel";
import { IngestionPanel } from "./components/IngestionPanel";
import { TracePanel } from "./components/TracePanel";
import { Workbench } from "./components/Workbench";
import type {
  AgentResponse,
  AssetAccess,
  EvaluationDataset,
  EvaluationRun,
  IngestionJob,
  RetrievalTrace,
  StreamUiState,
  Thread,
  UploadMetadata
} from "./types";

type View = "workbench" | "trace" | "ingestion" | "evaluation";

const starterQuestion = "ETCH-03 Chamber B 清腔后首片异常，当前 SOP 怎么要求？";

function App() {
  const [view, setView] = useState<View>("workbench");
  const [threads, setThreads] = useState<Thread[]>([]);
  const [thread, setThread] = useState<Thread | null>(null);
  const [agentResult, setAgentResult] = useState<AgentResponse | null>(null);
  const [traces, setTraces] = useState<RetrievalTrace[]>([]);
  const [selectedTrace, setSelectedTrace] = useState<RetrievalTrace>();
  const [jobs, setJobs] = useState<IngestionJob[]>([]);
  const [selectedJob, setSelectedJob] = useState<IngestionJob>();
  const [datasets, setDatasets] = useState<EvaluationDataset[]>([]);
  const [evaluations, setEvaluations] = useState<EvaluationRun[]>([]);
  const [selectedEvaluation, setSelectedEvaluation] = useState<EvaluationRun>();
  const [query, setQuery] = useState(starterQuestion);
  const [asset, setAsset] = useState<AssetAccess | null>(null);
  const [assetError, setAssetError] = useState("");
  const [assetModalOpen, setAssetModalOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [accessRequired, setAccessRequired] = useState(false);
  const [accessKey, setAccessKey] = useState("");
  const [actionLoading, setActionLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [streamState, setStreamState] = useState<StreamUiState | null>(null);
  const streamAbortRef = useRef<AbortController | null>(null);
  const streamStopRequestedRef = useRef(false);
  const streamLifecycleRef = useRef<{ threadId: string; requestId: string } | null>(null);
  const manualAssetSelectionRef = useRef<{ resultKey: string; imageId: string } | null>(null);

  const traceOptions = useMemo(() => {
    if (!selectedTrace || traces.some((item) => item.trace_id === selectedTrace.trace_id)) return traces;
    return [selectedTrace, ...traces];
  }, [selectedTrace, traces]);
  const hasPendingJobs = jobs.some((job) => !["published", "failed"].includes(job.status));
  const hasPendingEvaluations = evaluations.some((run) => ["queued", "running"].includes(run.status));
  const activeImageSelection = useMemo(() => {
    const latestAssistant = [...(thread?.messages ?? [])]
      .reverse()
      .find((message) => message.role === "assistant");
    const imageIds = agentResult?.image_asset_ids
      ?? latestAssistant?.presentation?.image_asset_ids
      ?? [];
    return {
      resultKey: `${thread?.thread_id ?? "none"}:${latestAssistant?.request_id ?? latestAssistant?.message_id ?? "empty"}`,
      imageIds: [...new Set(imageIds)]
    };
  }, [agentResult, thread]);
  const activeImageIdsKey = activeImageSelection.imageIds.join("\u001f");

  async function initializeWorkspace(demoAccessKey: string) {
    await bootstrapToken(demoAccessKey);
    const [existingThreads, nextJobs, nextTraces, nextEvaluations, nextDatasets] = await Promise.all([
      listThreads(),
      listJobs(),
      listTraces(),
      listEvaluations(),
      listEvaluationDatasets()
    ]);
    const activeThread = existingThreads[0]
      ? await getThread(existingThreads[0].thread_id)
      : await createThread("ETCH-03 异常调查");
    setThreads(existingThreads.length ? existingThreads : [activeThread]);
    setThread(activeThread);
    setJobs(nextJobs);
    setSelectedJob(nextJobs[0]);
    setTraces(nextTraces);
    setSelectedTrace(nextTraces[0]);
    setEvaluations(nextEvaluations);
    setSelectedEvaluation(nextEvaluations[0]);
    setDatasets(nextDatasets);
  }

  useEffect(() => {
    async function initialize() {
      const savedAccessKey = window.sessionStorage.getItem("semikb_demo_access_key") ?? "";
      try {
        await initializeWorkspace(savedAccessKey);
      } catch (caught) {
        if (caught instanceof ApiError && caught.status === 401) {
          window.sessionStorage.removeItem("semikb_demo_access_key");
          setAccessRequired(true);
          setError(savedAccessKey ? "访问码无效，请重新输入。" : "请输入测试访问码。" );
        } else {
          setError(messageFrom(caught, "无法连接后端服务。"));
        }
      } finally {
        setLoading(false);
      }
    }
    void initialize();
  }, []);

  async function submitAccessKey(event: FormEvent) {
    event.preventDefault();
    if (!accessKey.trim()) return;
    setLoading(true);
    setError("");
    try {
      await initializeWorkspace(accessKey.trim());
      window.sessionStorage.setItem("semikb_demo_access_key", accessKey.trim());
      setAccessRequired(false);
    } catch (caught) {
      setError(
        caught instanceof ApiError && caught.status === 401
          ? "访问码无效，请重新输入。"
          : messageFrom(caught, "无法连接后端服务。")
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!hasPendingJobs && !hasPendingEvaluations) return;
    const timer = window.setInterval(() => {
      if (hasPendingJobs) void refreshJobs();
      if (hasPendingEvaluations) void refreshEvaluations();
    }, 2500);
    return () => window.clearInterval(timer);
  }, [hasPendingEvaluations, hasPendingJobs]);

  useEffect(() => () => {
    if (asset?.local_object_url) URL.revokeObjectURL(asset.url);
  }, [asset]);

  useEffect(() => {
    let cancelled = false;
    manualAssetSelectionRef.current = null;
    setAsset(null);
    setAssetError("");
    setAssetModalOpen(false);
    const firstImageId = activeImageSelection.imageIds[0];
    if (!firstImageId) return () => { cancelled = true; };

    void resolveAsset(firstImageId)
      .then((nextAsset) => {
        if (!cancelled && manualAssetSelectionRef.current === null) {
          setAsset(nextAsset);
        } else if (nextAsset.local_object_url) {
          URL.revokeObjectURL(nextAsset.url);
        }
      })
      .catch(() => {
        if (!cancelled && manualAssetSelectionRef.current === null) {
          setAsset(null);
          setAssetError("当前结果的图片无法加载，文本答案仍可正常使用。");
        }
      });
    return () => { cancelled = true; };
  }, [activeImageSelection.resultKey, activeImageIdsKey]);

  useEffect(() => {
    function cancelOnPageHide() {
      const active = streamLifecycleRef.current;
      if (active) {
        void cancelMessageRequest(active.threadId, active.requestId, true).catch(() => undefined);
      }
      streamAbortRef.current?.abort();
    }

    window.addEventListener("pagehide", cancelOnPageHide);
    return () => {
      window.removeEventListener("pagehide", cancelOnPageHide);
      cancelOnPageHide();
    };
  }, []);

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: "auto" });
    setNotice("");
  }, [view]);

  async function refreshJobs(preferredId?: string) {
    const nextJobs = await listJobs();
    setJobs(nextJobs);
    const targetId = preferredId ?? selectedJob?.job_id;
    const target = nextJobs.find((job) => job.job_id === targetId) ?? nextJobs[0];
    setSelectedJob(target);
  }

  async function refreshTraces(preferredId?: string) {
    const nextTraces = await listTraces();
    setTraces(nextTraces);
    const targetId = preferredId ?? selectedTrace?.trace_id;
    const target = nextTraces.find((trace) => trace.trace_id === targetId) ?? nextTraces[0];
    if (target) setSelectedTrace(target);
  }

  async function refreshEvaluations(preferredId?: string) {
    const [nextRuns, nextDatasets] = await Promise.all([listEvaluations(), listEvaluationDatasets()]);
    setEvaluations(nextRuns);
    setDatasets(nextDatasets);
    const targetId = preferredId ?? selectedEvaluation?.evaluation_run_id;
    const target = nextRuns.find((run) => run.evaluation_run_id === targetId) ?? nextRuns[0];
    setSelectedEvaluation(target);
  }

  async function submitQuestion(event: FormEvent) {
    event.preventDefault();
    if (!thread || !query.trim()) return;
    await runStream(query.trim(), newStreamRequestId(), true);
  }

  async function runStream(content: string, requestId: string, optimistic: boolean) {
    if (!thread || streamAbortRef.current) return;
    const threadId = thread.thread_id;
    const controller = new AbortController();
    streamAbortRef.current = controller;
    streamStopRequestedRef.current = false;
    streamLifecycleRef.current = { threadId, requestId };
    setActionLoading(true);
    setError("");
    setNotice("");
    setStreamState({
      requestId,
      content,
      status: "running",
      stage: "analyzing_request",
      stageMessage: "请求已提交，等待服务端接受",
      partialAnswer: "",
      internalEvidenceCount: 0,
      externalEvidenceCount: 0,
      elapsedMs: 0,
      retryable: false,
      tasks: []
    });
    if (optimistic) {
      setThread({
        ...thread,
        messages: [...thread.messages, {
          message_id: `optimistic_${requestId}`,
          request_id: requestId,
          role: "user",
          content,
          citations: [],
          created_at: new Date().toISOString()
        }]
      });
      setQuery("");
    }

    try {
      const response = await sendMessageStream(
        threadId,
        content,
        requestId,
        (streamEvent) => {
          setStreamState((current) => {
            if (!current || current.requestId !== requestId) return current;
            if (streamEvent.event === "stage") {
              return {
                ...current,
                stage: streamEvent.data.stage,
                stageMessage: streamEvent.data.message ?? current.stageMessage
              };
            }
            if (streamEvent.event === "evidence") {
              return {
                ...current,
                internalEvidenceCount: streamEvent.data.internal_count ?? 0,
                externalEvidenceCount: streamEvent.data.external_count ?? 0
              };
            }
            if (streamEvent.event === "task_status" && streamEvent.data.task_id && streamEvent.data.status) {
              const task = {
                taskId: streamEvent.data.task_id,
                status: streamEvent.data.status,
                route: streamEvent.data.route,
                message: streamEvent.data.message ?? "任务状态已更新"
              };
              return {
                ...current,
                tasks: [...current.tasks.filter((item) => item.taskId !== task.taskId), task]
                  .sort((left, right) => left.taskId.localeCompare(right.taskId))
              };
            }
            if (streamEvent.event === "answer_delta") {
              return {
                ...current,
                partialAnswer: current.partialAnswer + (streamEvent.data.delta ?? "")
              };
            }
            if (streamEvent.event === "heartbeat") {
              return { ...current, elapsedMs: streamEvent.data.elapsed_ms ?? current.elapsedMs };
            }
            return current;
          });
        },
        controller.signal
      );
      setThread(response.thread);
      setThreads((current) => [response.thread, ...current.filter((item) => item.thread_id !== response.thread.thread_id)]);
      setAgentResult(response);
      if (response.trace_id) await refreshTraces(response.trace_id);
      setNotice(response.clarification_required ? "线程已暂停，等待补充信息" : "调查结果已写入线程");
      setStreamState(null);
    } catch (caught) {
      const stopped = streamStopRequestedRef.current
        || (caught instanceof DOMException && caught.name === "AbortError");
      try {
        const refreshed = await getThread(threadId);
        setThread(refreshed);
        setThreads((current) => [refreshed, ...current.filter((item) => item.thread_id !== threadId)]);
      } catch {
        // Keep the local conversation visible if reconciliation is temporarily unavailable.
      }
      setStreamState((current) => current ? {
        ...current,
        status: stopped ? "stopped" : "error",
        stageMessage: stopped
          ? "已停止，未完成内容没有写入会话"
          : messageFrom(caught, "流式请求失败。"),
        retryable: stopped || (caught instanceof AgentStreamError && caught.retryable)
      } : current);
      if (stopped) setNotice("生成已停止，可使用同一请求重试");
      else setError(messageFrom(caught, "问题发送失败。"));
    } finally {
      if (streamLifecycleRef.current?.requestId === requestId) {
        streamLifecycleRef.current = null;
      }
      streamAbortRef.current = null;
      setActionLoading(false);
    }
  }

  async function stopStream() {
    const controller = streamAbortRef.current;
    if (!controller || !thread || !streamState) return;
    streamStopRequestedRef.current = true;
    setStreamState((current) => current ? {
      ...current,
      stageMessage: "正在停止生成并同步服务端状态"
    } : current);
    try {
      await cancelMessageRequest(thread.thread_id, streamState.requestId);
    } catch (caught) {
      setError(messageFrom(caught, "服务端取消确认失败。"));
    } finally {
      controller.abort();
    }
  }

  async function retryStream() {
    if (!streamState || streamState.status === "running") return;
    await runStream(streamState.content, streamState.requestId, false);
  }

  async function selectThread(threadId: string) {
    await runAction(async () => {
      setThread(await getThread(threadId));
      setAgentResult(null);
      setAsset(null);
      setAssetError("");
      setStreamState(null);
    }, "线程恢复失败。");
  }

  async function addThread() {
    await runAction(async () => {
      const created = await createThread(`新调查 ${new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}`);
      setThreads((current) => [created, ...current]);
      setThread(created);
      setAgentResult(null);
      setStreamState(null);
      setQuery("");
    }, "线程创建失败。");
  }

  async function openTrace(traceId?: string | null) {
    await runAction(async () => {
      if (traceId) {
        const existing = traces.find((item) => item.trace_id === traceId);
        setSelectedTrace(existing ?? await getTrace(traceId));
      }
      setView("trace");
    }, "Trace 读取失败。");
  }

  async function selectTrace(traceId: string) {
    const existing = traceOptions.find((item) => item.trace_id === traceId);
    if (existing) setSelectedTrace(existing);
    else await openTrace(traceId);
  }

  async function openImage(imageId: string) {
    const resultKey = activeImageSelection.resultKey;
    manualAssetSelectionRef.current = { resultKey, imageId };
    setAssetError("");
    try {
      const nextAsset = await resolveAsset(imageId);
      const selection = manualAssetSelectionRef.current;
      if (selection?.resultKey === resultKey && selection.imageId === imageId) {
        setAsset(nextAsset);
        setAssetModalOpen(true);
      } else if (nextAsset.local_object_url) {
        URL.revokeObjectURL(nextAsset.url);
      }
    } catch (caught) {
      const selection = manualAssetSelectionRef.current;
      if (selection?.resultKey === resultKey && selection.imageId === imageId) {
        setAsset(null);
        setAssetModalOpen(false);
        setAssetError("图片无权访问、已失效或暂时无法加载，文本答案不受影响。");
      }
      setError(messageFrom(caught, "图片访问失败。"));
    }
  }

  function handleAssetLoadError() {
    setAsset(null);
    setAssetModalOpen(false);
    setAssetError("图片链接已失效或加载失败，文本答案仍可正常使用。");
  }

  async function upload(file: File, metadata: UploadMetadata) {
    let succeeded = false;
    await runAction(async () => {
      const job = await uploadDocument(file, metadata);
      await refreshJobs(job.job_id);
      setNotice(`入库任务 ${job.job_id.slice(-8)} 已创建`);
      succeeded = true;
    }, "文档上传失败。");
    return succeeded;
  }

  async function retryIngestion(jobId: string) {
    await runAction(async () => {
      const job = await retryJob(jobId);
      await refreshJobs(job.job_id);
      setNotice("失败任务已重新排队");
    }, "任务重试失败。");
  }

  async function selectIngestionJob(jobId: string) {
    await runAction(async () => setSelectedJob(await getJob(jobId)), "任务详情读取失败。");
  }

  async function createEvaluation(input: { dataset_version: string; retrieval_profile: EvaluationRun["retrieval_profile"]; baseline_run_id?: string }) {
    await runAction(async () => {
      const run = await runEvaluation(input);
      await refreshEvaluations(run.evaluation_run_id);
      setNotice(`评估 ${run.evaluation_run_id.slice(-8)} 已排队`);
    }, "评估创建失败。");
  }

  async function selectEvaluation(runId: string) {
    await runAction(async () => setSelectedEvaluation(await getEvaluation(runId)), "评估详情读取失败。");
  }

  async function retryEvaluationRun(runId: string) {
    await runAction(async () => {
      const run = await retryEvaluation(runId);
      await refreshEvaluations(run.evaluation_run_id);
      setNotice("评估已重新排队");
    }, "评估重试失败。");
  }

  async function openEvaluationTrace(runId: string, caseId: string) {
    await runAction(async () => {
      const trace = await getEvaluationCaseTrace(runId, caseId);
      setSelectedTrace(trace);
      setView("trace");
    }, "评估 Trace 读取失败。");
  }

  async function runAction(action: () => Promise<void>, fallback: string) {
    setActionLoading(true);
    setError("");
    setNotice("");
    try {
      await action();
    } catch (caught) {
      setError(messageFrom(caught, fallback));
    } finally {
      setActionLoading(false);
    }
  }

  if (loading) return <div className="loading-screen"><LoaderCircle className="spin" size={28} />正在连接知识服务</div>;

  if (accessRequired) return <main className="access-screen">
    <form className="access-panel" onSubmit={submitAccessKey}>
      <div className="access-heading"><span className="brand-mark">S</span><div><span className="section-label">SEMIKB</span><h1>测试环境访问</h1></div></div>
      <label htmlFor="demo-access-key">访问码</label>
      <div className="access-input-row"><KeyRound size={18} /><input id="demo-access-key" type="password" autoComplete="current-password" value={accessKey} onChange={(event) => setAccessKey(event.target.value)} autoFocus /></div>
      {error && <div className="error-banner" role="alert">{error}</div>}
      <button className="command-button" type="submit" disabled={!accessKey.trim()}>进入工作台</button>
    </form>
  </main>;

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark">S</span><span>SEMIKB</span></div>
      <nav aria-label="主导航">
        <NavButton active={view === "workbench"} icon={<MessageSquareText />} label="工程师工作台" onClick={() => setView("workbench")} />
        <NavButton active={view === "trace"} icon={<SearchCheck />} label="可解释检索" onClick={() => setView("trace")} />
        <NavButton active={view === "ingestion"} icon={<Upload />} label="入库任务中心" onClick={() => setView("ingestion")} />
        <NavButton active={view === "evaluation"} icon={<ClipboardCheck />} label="离线评估" onClick={() => setView("evaluation")} />
      </nav>
      <div className="sidebar-status"><span className="status-dot" /><span>FAB-01 · scoped access</span></div>
    </aside>

    <main className="main-area">
      <header className="topbar">
        <div><p className="eyebrow">FAB-01 / P-ALPHA / ETCH</p><h1>{viewTitle(view)}</h1></div>
        <div className="topbar-right"><Activity size={16} /><span>Evidence-first</span></div>
      </header>
      {error && <div className="error-banner" role="alert">{error}</div>}
      {notice && <div className="notice-banner" role="status">{notice}</div>}

      {view === "workbench" && <Workbench
        threads={threads}
        thread={thread}
        result={agentResult}
        query={query}
        loading={actionLoading}
        asset={asset}
        assetError={assetError}
        imageIds={activeImageSelection.imageIds}
        streamState={streamState}
        onQueryChange={setQuery}
        onSubmit={submitQuestion}
        onSelectThread={(threadId) => void selectThread(threadId)}
        onNewThread={() => void addThread()}
        onOpenImage={(imageId) => void openImage(imageId)}
        onAssetLoadError={handleAssetLoadError}
        onOpenTrace={(traceId) => void openTrace(traceId)}
        onStopStream={() => void stopStream()}
        onRetryStream={() => void retryStream()}
      />}
      {view === "trace" && <TracePanel traces={traceOptions} trace={selectedTrace} onSelect={(traceId) => void selectTrace(traceId)} onOpenImage={(imageId) => void openImage(imageId)} />}
      {view === "ingestion" && <IngestionPanel jobs={jobs} selectedJob={selectedJob} loading={actionLoading} onSelect={(jobId) => void selectIngestionJob(jobId)} onUpload={upload} onRetry={retryIngestion} onRefresh={() => refreshJobs()} />}
      {view === "evaluation" && <EvaluationPanel datasets={datasets} runs={evaluations} selectedRun={selectedEvaluation} loading={actionLoading} onSelect={(runId) => void selectEvaluation(runId)} onRun={createEvaluation} onRetry={retryEvaluationRun} onRefresh={() => refreshEvaluations()} onOpenTrace={openEvaluationTrace} />}
    </main>

    {asset && assetModalOpen && <div className="asset-modal" role="dialog" aria-modal="true" aria-label="受控图片预览" data-testid="asset-modal">
      <div className="asset-modal-toolbar"><div><ImageIcon size={17} /><strong>{asset.image_id}</strong><span>{asset.object_key}</span></div><button className="icon-button" type="button" title="关闭图片" onClick={() => setAssetModalOpen(false)}><X size={18} /></button></div>
      <img src={asset.url} alt={`受控图片 ${asset.image_id}`} onError={handleAssetLoadError} />
    </div>}
  </div>;
}

function NavButton({ active, icon, label, onClick }: { active: boolean; icon: ReactNode; label: string; onClick: () => void }) {
  return <button type="button" className={`nav-button ${active ? "active" : ""}`} onClick={onClick} title={label}>{icon}<span>{label}</span></button>;
}

function viewTitle(view: View) {
  return ({ workbench: "工程师工作台", trace: "可解释检索", ingestion: "入库任务中心", evaluation: "离线评估" } as Record<View, string>)[view];
}

function messageFrom(value: unknown, fallback: string): string {
  return value instanceof Error ? value.message : fallback;
}

function newStreamRequestId(): string {
  const words = new Uint32Array(4);
  window.crypto.getRandomValues(words);
  const random = Array.from(words, (word) => word.toString(16).padStart(8, "0")).join("");
  return `web_${Date.now().toString(36)}_${random}`;
}

export default App;
