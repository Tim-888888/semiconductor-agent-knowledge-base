import { type FormEvent, type ReactNode, useEffect, useMemo, useState } from "react";
import {
  Activity,
  ClipboardCheck,
  Image as ImageIcon,
  LoaderCircle,
  MessageSquareText,
  SearchCheck,
  Upload,
  X
} from "lucide-react";
import {
  bootstrapToken,
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
  sendMessage,
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
  const [assetModalOpen, setAssetModalOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const traceOptions = useMemo(() => {
    if (!selectedTrace || traces.some((item) => item.trace_id === selectedTrace.trace_id)) return traces;
    return [selectedTrace, ...traces];
  }, [selectedTrace, traces]);
  const hasPendingJobs = jobs.some((job) => !["published", "failed"].includes(job.status));
  const hasPendingEvaluations = evaluations.some((run) => ["queued", "running"].includes(run.status));

  useEffect(() => {
    async function initialize() {
      try {
        await bootstrapToken();
        const [existingThreads, nextJobs, nextTraces, nextEvaluations, nextDatasets] = await Promise.all([
          listThreads(),
          listJobs(),
          listTraces(),
          listEvaluations(),
          listEvaluationDatasets()
        ]);
        const activeThread = existingThreads[0] ?? await createThread("ETCH-03 异常调查");
        setThreads(existingThreads.length ? existingThreads : [activeThread]);
        setThread(activeThread);
        setJobs(nextJobs);
        setSelectedJob(nextJobs[0]);
        setTraces(nextTraces);
        setSelectedTrace(nextTraces[0]);
        setEvaluations(nextEvaluations);
        setSelectedEvaluation(nextEvaluations[0]);
        setDatasets(nextDatasets);
      } catch (caught) {
        setError(messageFrom(caught, "无法连接后端服务。"));
      } finally {
        setLoading(false);
      }
    }
    void initialize();
  }, []);

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
    await runAction(async () => {
      const response = await sendMessage(thread.thread_id, query.trim());
      setThread(response.thread);
      setThreads((current) => [response.thread, ...current.filter((item) => item.thread_id !== response.thread.thread_id)]);
      setAgentResult(response);
      setQuery("");
      if (response.trace_id) await refreshTraces(response.trace_id);
      setNotice(response.clarification_required ? "线程已暂停，等待补充信息" : "调查结果已写入线程");
    }, "问题发送失败。");
  }

  async function selectThread(threadId: string) {
    await runAction(async () => {
      setThread(await getThread(threadId));
      setAgentResult(null);
      setAsset(null);
    }, "线程恢复失败。");
  }

  async function addThread() {
    await runAction(async () => {
      const created = await createThread(`新调查 ${new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}`);
      setThreads((current) => [created, ...current]);
      setThread(created);
      setAgentResult(null);
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
    await runAction(async () => {
      const nextAsset = await resolveAsset(imageId);
      setAsset(nextAsset);
      setAssetModalOpen(true);
    }, "图片访问被拒绝。");
  }

  async function upload(file: File, metadata: UploadMetadata) {
    await runAction(async () => {
      const job = await uploadDocument(file, metadata);
      await refreshJobs(job.job_id);
      setNotice(`入库任务 ${job.job_id.slice(-8)} 已创建`);
    }, "文档上传失败。");
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
        onQueryChange={setQuery}
        onSubmit={submitQuestion}
        onSelectThread={(threadId) => void selectThread(threadId)}
        onNewThread={() => void addThread()}
        onOpenImage={(imageId) => void openImage(imageId)}
        onOpenTrace={(traceId) => void openTrace(traceId)}
      />}
      {view === "trace" && <TracePanel traces={traceOptions} trace={selectedTrace} onSelect={(traceId) => void selectTrace(traceId)} onOpenImage={(imageId) => void openImage(imageId)} />}
      {view === "ingestion" && <IngestionPanel jobs={jobs} selectedJob={selectedJob} loading={actionLoading} onSelect={(jobId) => void selectIngestionJob(jobId)} onUpload={upload} onRetry={retryIngestion} onRefresh={() => refreshJobs()} />}
      {view === "evaluation" && <EvaluationPanel datasets={datasets} runs={evaluations} selectedRun={selectedEvaluation} loading={actionLoading} onSelect={(runId) => void selectEvaluation(runId)} onRun={createEvaluation} onRetry={retryEvaluationRun} onRefresh={() => refreshEvaluations()} onOpenTrace={openEvaluationTrace} />}
    </main>

    {asset && assetModalOpen && <div className="asset-modal" role="dialog" aria-modal="true" aria-label="受控图片预览" data-testid="asset-modal">
      <div className="asset-modal-toolbar"><div><ImageIcon size={17} /><strong>{asset.image_id}</strong><span>{asset.object_key}</span></div><button className="icon-button" type="button" title="关闭图片" onClick={() => setAssetModalOpen(false)}><X size={18} /></button></div>
      <img src={asset.url} alt={`受控图片 ${asset.image_id}`} />
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

export default App;
