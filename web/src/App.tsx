import { FormEvent, type ReactNode, useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowUp,
  Bot,
  ClipboardCheck,
  Database,
  FileSearch,
  Image as ImageIcon,
  LoaderCircle,
  MessageSquareText,
  Plus,
  SearchCheck,
  Upload
} from "lucide-react";
import {
  bootstrapToken,
  createDemoDocument,
  createThread,
  getAsset,
  listEvaluations,
  listJobs,
  listTraces,
  runEvaluation,
  sendMessage
} from "./api";
import type { EvaluationRun, IngestionJob, RetrievalTrace, Thread } from "./types";

type View = "workbench" | "trace" | "ingestion" | "evaluation";

const starterQuestion = "ETCH-03 Chamber B 清腔后首片异常，当前 SOP 怎么要求？";

function App() {
  const [view, setView] = useState<View>("workbench");
  const [thread, setThread] = useState<Thread | null>(null);
  const [traces, setTraces] = useState<RetrievalTrace[]>([]);
  const [jobs, setJobs] = useState<IngestionJob[]>([]);
  const [evaluations, setEvaluations] = useState<EvaluationRun[]>([]);
  const [query, setQuery] = useState(starterQuestion);
  const [assetUrl, setAssetUrl] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [error, setError] = useState("");

  const activeTrace = traces[0];
  const activeImages = useMemo(
    () => thread?.messages.flatMap((message) => message.citations.flatMap((citation) => citation.image_ids)) ?? [],
    [thread]
  );

  async function refreshOperations() {
    const [nextJobs, nextTraces, nextEvaluations] = await Promise.all([listJobs(), listTraces(), listEvaluations()]);
    setJobs(nextJobs);
    setTraces(nextTraces);
    setEvaluations(nextEvaluations);
  }

  useEffect(() => {
    async function initialize() {
      try {
        await bootstrapToken();
        const nextThread = await createThread("ETCH-03 首片异常调查");
        setThread(nextThread);
        await refreshOperations();
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "无法连接后端服务。");
      } finally {
        setLoading(false);
      }
    }
    void initialize();
  }, []);

  async function submitQuestion(event: FormEvent) {
    event.preventDefault();
    if (!thread || !query.trim()) return;
    setActionLoading(true);
    setError("");
    try {
      const response = await sendMessage(thread.thread_id, query.trim());
      setThread(response.thread);
      setQuery("");
      setView("workbench");
      await refreshOperations();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "问题发送失败。");
    } finally {
      setActionLoading(false);
    }
  }

  async function openImage(imageId: string) {
    try {
      const result = await getAsset(imageId);
      setAssetUrl(result.url);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "图片访问被拒绝。");
    }
  }

  async function addTrainingDocument() {
    setActionLoading(true);
    try {
      await createDemoDocument();
      await refreshOperations();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "入库任务创建失败。");
    } finally {
      setActionLoading(false);
    }
  }

  async function startEvaluation() {
    setActionLoading(true);
    try {
      await runEvaluation();
      await refreshOperations();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "评估运行失败。");
    } finally {
      setActionLoading(false);
    }
  }

  if (loading) {
    return <div className="loading-screen"><LoaderCircle className="spin" size={28} /> 正在连接知识服务</div>;
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">S</span><span>SEMIKB</span></div>
        <nav aria-label="Primary navigation">
          <NavButton active={view === "workbench"} icon={<MessageSquareText />} label="工程师工作台" onClick={() => setView("workbench")} />
          <NavButton active={view === "trace"} icon={<SearchCheck />} label="可解释检索" onClick={() => setView("trace")} />
          <NavButton active={view === "ingestion"} icon={<Upload />} label="入库任务中心" onClick={() => setView("ingestion")} />
          <NavButton active={view === "evaluation"} icon={<ClipboardCheck />} label="离线评估" onClick={() => setView("evaluation")} />
        </nav>
        <div className="sidebar-status"><span className="status-dot" /> Demo data · controlled mode</div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div>
            <p className="eyebrow">FAB-01 / P-ALPHA / ETCH</p>
            <h1>{viewTitle(view)}</h1>
          </div>
          <div className="topbar-right"><Activity size={16} /><span>Evidence-first</span></div>
        </header>
        {error && <div className="error-banner">{error}</div>}
        {view === "workbench" && (
          <Workbench
            thread={thread}
            query={query}
            loading={actionLoading}
            assetUrl={assetUrl}
            imageIds={activeImages}
            onQueryChange={setQuery}
            onSubmit={submitQuestion}
            onOpenImage={openImage}
            onOpenTrace={() => setView("trace")}
          />
        )}
        {view === "trace" && <TracePanel trace={activeTrace} onOpenImage={openImage} />}
        {view === "ingestion" && <IngestionPanel jobs={jobs} loading={actionLoading} onAdd={addTrainingDocument} />}
        {view === "evaluation" && <EvaluationPanel runs={evaluations} loading={actionLoading} onRun={startEvaluation} />}
      </main>
    </div>
  );
}

function NavButton({ active, icon, label, onClick }: { active: boolean; icon: ReactNode; label: string; onClick: () => void }) {
  return <button type="button" className={`nav-button ${active ? "active" : ""}`} onClick={onClick}>{icon}<span>{label}</span></button>;
}

function Workbench({ thread, query, loading, assetUrl, imageIds, onQueryChange, onSubmit, onOpenImage, onOpenTrace }: {
  thread: Thread | null;
  query: string;
  loading: boolean;
  assetUrl: string;
  imageIds: string[];
  onQueryChange: (value: string) => void;
  onSubmit: (event: FormEvent) => void;
  onOpenImage: (imageId: string) => void;
  onOpenTrace: () => void;
}) {
  return <section className="workbench-grid">
    <div className="conversation-panel">
      <div className="panel-heading"><div><span className="section-label">ACTIVE THREAD</span><h2>{thread?.title ?? "Loading thread"}</h2></div><span className="thread-id">{thread?.thread_id.slice(-8)}</span></div>
      <div className="message-list">
        {thread?.messages.length ? thread.messages.map((message) => <article className={`message ${message.role}`} key={message.message_id}>
          <div className="message-avatar">{message.role === "assistant" ? <Bot size={17} /> : "E"}</div>
          <div className="message-body"><div className="message-role">{message.role === "assistant" ? "SEMIKB" : "ENGINEER"}</div><p>{message.content}</p>
            {message.citations.length > 0 && <div className="citation-row">{message.citations.map((citation) => <button key={citation.chunk_id} className="citation" type="button" onClick={onOpenTrace}>{citation.document_id} {citation.revision} · {citation.page_or_section}</button>)}</div>}
          </div>
        </article>) : <div className="empty-conversation">使用下方问题开始受控调查。</div>}
      </div>
      <form className="query-box" onSubmit={onSubmit}>
        <textarea value={query} onChange={(event) => onQueryChange(event.target.value)} placeholder="描述异常、SOP 或 Recipe 问题" rows={3} />
        <button type="submit" className="icon-command" title="发送问题" disabled={loading || !query.trim()}><ArrowUp size={19} /></button>
      </form>
    </div>
    <aside className="evidence-panel">
      <div className="panel-heading"><div><span className="section-label">EVIDENCE DRAWER</span><h2>真实图文证据</h2></div><ImageIcon size={19} /></div>
      {assetUrl ? <img className="wafer-image" src={assetUrl} alt="Synthetic wafer edge-ring inspection" /> : <div className="asset-placeholder"><ImageIcon size={32} /><span>检索命中图片后在此展示</span></div>}
      <div className="asset-list">{[...new Set(imageIds)].map((imageId) => <button className="asset-item" type="button" key={imageId} onClick={() => onOpenImage(imageId)}><ImageIcon size={16} /><span>{imageId}</span></button>)}</div>
      <p className="evidence-note">图片展示前会重新执行权限、revision 和有效期检查。Demo 页面显示的是合成晶圆图。</p>
    </aside>
  </section>;
}

function TracePanel({ trace, onOpenImage }: { trace?: RetrievalTrace; onOpenImage: (imageId: string) => void }) {
  if (!trace) return <section className="empty-state"><FileSearch size={34} /><h2>尚无检索 Trace</h2><p>在工程师工作台完成一次查询后，这里会展示完整检索路径。</p></section>;
  return <section className="trace-layout">
    <div className="trace-summary"><div><span className="section-label">RETRIEVAL TRACE</span><h2>{trace.original_query}</h2></div><div className="route-list">{trace.routes.map((route) => <span key={route}>{route}</span>)}</div></div>
    <div className="trace-kpis"><Metric label="Candidates" value={trace.candidates.length} /><Metric label="Final evidence" value={trace.final_evidence_ids.length} /><Metric label="Cutoff" value={trace.cutoff_reason} /><Metric label="Latency" value={`${trace.timings_ms.retrieval ?? 0} ms`} /></div>
    <div className="data-table-wrap"><table><thead><tr><th>Evidence</th><th>Routes</th><th>Dense</th><th>Sparse</th><th>RRF</th><th>Rerank</th><th>Status</th></tr></thead><tbody>{trace.candidates.map((candidate) => <tr key={candidate.chunk_id}><td><strong>{candidate.document_id} {candidate.revision}</strong><span>{candidate.page_or_section}</span></td><td>{candidate.routes.join(" + ")}</td><td>{candidate.dense_score.toFixed(3)}</td><td>{candidate.sparse_score.toFixed(3)}</td><td>{candidate.rrf_score.toFixed(3)}</td><td>{candidate.rerank_score.toFixed(3)}</td><td><span className={`status-pill ${candidate.selected ? "selected" : "excluded"}`}>{candidate.selected ? "Selected" : candidate.exclusion_reason ?? "Excluded"}</span></td></tr>)}</tbody></table></div>
    {trace.image_asset_ids.length > 0 && <div className="trace-images">{trace.image_asset_ids.map((imageId) => <button type="button" onClick={() => onOpenImage(imageId)} key={imageId}><ImageIcon size={16} />打开 {imageId}</button>)}</div>}
  </section>;
}

function IngestionPanel({ jobs, loading, onAdd }: { jobs: IngestionJob[]; loading: boolean; onAdd: () => void }) {
  return <section className="operations-layout"><div className="operations-heading"><div><span className="section-label">INGESTION OPERATIONS</span><h2>入库任务中心</h2><p>每一份文档都在发布前经过解析、质量检查、暂存索引与幂等控制。</p></div><button className="command-button" type="button" onClick={onAdd} disabled={loading}><Plus size={17} />添加训练文档</button></div>
    <div className="data-table-wrap"><table><thead><tr><th>Document</th><th>Stage</th><th>Progress</th><th>Chunks</th><th>Images</th><th>Latest event</th></tr></thead><tbody>{jobs.map((job) => <tr key={job.job_id}><td><strong>{job.document_id} {job.revision}</strong><span>{job.filename}</span></td><td><span className="status-pill selected">{job.current_stage}</span></td><td><div className="progress"><span style={{ width: `${job.progress}%` }} /></div>{job.progress}%</td><td>{job.chunks_count}</td><td>{job.images_count}</td><td>{job.events.at(-1)?.message ?? "Queued"}</td></tr>)}</tbody></table></div>
  </section>;
}

function EvaluationPanel({ runs, loading, onRun }: { runs: EvaluationRun[]; loading: boolean; onRun: () => void }) {
  const latest = runs[0];
  return <section className="operations-layout"><div className="operations-heading"><div><span className="section-label">OFFLINE EVALUATION</span><h2>黄金集与基线比较</h2><p>运行固定数据集，检查检索优化是否真实提升命中率。</p></div><button className="command-button" type="button" onClick={onRun} disabled={loading}><ClipboardCheck size={17} />运行 demo-v1</button></div>
    {latest ? <><div className="metric-grid"><Metric label="Recall@5" value={latest.aggregate_metrics.recall_at_5 ?? 0} /><Metric label="MRR" value={latest.aggregate_metrics.mrr ?? 0} /><Metric label="Cases" value={latest.case_results.length} /><Metric label="Status" value={latest.status} /></div><div className="data-table-wrap"><table><thead><tr><th>Case</th><th>Recall@5</th><th>MRR</th><th>Trace</th></tr></thead><tbody>{latest.case_results.map((item) => <tr key={item.case_id}><td>{item.case_id}</td><td>{item.recall_at_5}</td><td>{item.reciprocal_rank}</td><td>{item.trace_id}</td></tr>)}</tbody></table></div></> : <div className="empty-state"><Database size={34} /><h2>尚未运行评估</h2><p>使用 demo-v1 固定黄金集创建第一条可复现基线。</p></div>}
  </section>;
}

function Metric({ label, value }: { label: string; value: string | number }) { return <div className="metric"><span>{label}</span><strong>{value}</strong></div>; }
function viewTitle(view: View) { return ({ workbench: "工程师工作台", trace: "可解释检索", ingestion: "入库任务中心", evaluation: "离线评估" } as Record<View, string>)[view]; }

export default App;
