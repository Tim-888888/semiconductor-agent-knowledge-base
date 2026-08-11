import { type FormEvent, useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowUp,
  Bot,
  CheckCircle2,
  FileText,
  FlaskConical,
  Image as ImageIcon,
  ListChecks,
  Plus,
  SearchCheck
} from "lucide-react";
import type { AgentResponse, AssetAccess, EvidenceLedgerEntry, Thread } from "../types";
import { StatusPill } from "./Common";

type Props = {
  threads: Thread[];
  thread: Thread | null;
  result: AgentResponse | null;
  query: string;
  loading: boolean;
  asset: AssetAccess | null;
  onQueryChange: (value: string) => void;
  onSubmit: (event: FormEvent) => void;
  onSelectThread: (threadId: string) => void;
  onNewThread: () => void;
  onOpenImage: (imageId: string) => void;
  onOpenTrace: (traceId?: string | null) => void;
};

export function Workbench({
  threads,
  thread,
  result,
  query,
  loading,
  asset,
  onQueryChange,
  onSubmit,
  onSelectThread,
  onNewThread,
  onOpenImage,
  onOpenTrace
}: Props) {
  const ledger = result?.evidence_ledger ?? [];
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string>("");
  const selectedEvidence = ledger.find((item) => item.evidence_id === selectedEvidenceId) ?? ledger[0];
  const imageIds = useMemo(() => {
    const fromResult = result?.image_asset_ids ?? [];
    const fromLedger = ledger.flatMap((item) => item.image_ids ?? []);
    const fromMessages = thread?.messages.flatMap((message) =>
      (message.citations ?? []).flatMap((citation) => citation.image_ids ?? [])
    ) ?? [];
    return [...new Set([...fromResult, ...fromLedger, ...fromMessages])];
  }, [ledger, result, thread]);

  return <section className="workbench-grid">
    <div className="conversation-panel">
      <div className="panel-heading workbench-heading">
        <div>
          <span className="section-label">ACTIVE INVESTIGATION</span>
          <h2>{thread?.title ?? "调查线程"}</h2>
        </div>
        <div className="thread-controls">
          <select
            aria-label="切换调查线程"
            data-testid="thread-select"
            value={thread?.thread_id ?? ""}
            onChange={(event) => onSelectThread(event.target.value)}
          >
            {threads.map((item) => <option key={item.thread_id} value={item.thread_id}>{item.title}</option>)}
          </select>
          <button className="icon-button" type="button" title="新建调查线程" onClick={onNewThread}>
            <Plus size={18} />
          </button>
        </div>
      </div>

      {thread?.status === "waiting_for_clarification" && <div className="clarification-band" data-testid="clarification-band">
        <AlertTriangle size={18} />
        <div><strong>等待补充信息 · 第 {thread.clarification_round} 轮</strong><span>{thread.pending_fields.join(" / ")}</span></div>
      </div>}

      <div className="message-list" data-testid="message-list">
        {thread?.messages.length ? thread.messages.map((message) => <article className={`message ${message.role}`} key={message.message_id}>
          <div className="message-avatar">{message.role === "assistant" ? <Bot size={17} /> : "E"}</div>
          <div className="message-body">
            <div className="message-role">{message.role === "assistant" ? "SEMIKB" : "ENGINEER"}</div>
            <p>{message.content}</p>
            {(message.citations ?? []).length > 0 && <div className="citation-row">
              {message.citations.map((citation, index) => <button
                key={citation.evidence_id ?? citation.chunk_id ?? index}
                className="citation"
                type="button"
                onClick={() => onOpenTrace(result?.trace_id)}
              >
                <FileText size={13} />
                {citation.document_id ?? citation.source_type ?? "证据"} {citation.revision ?? ""}
                {citation.page_or_section ? ` · ${citation.page_or_section}` : ""}
              </button>)}
            </div>}
          </div>
        </article>) : <div className="empty-conversation">当前线程尚无消息</div>}

        {result?.answer && !result.clarification_required && <StructuredAnswer result={result} onOpenTrace={onOpenTrace} />}
      </div>

      <form className="query-box" onSubmit={onSubmit}>
        <textarea
          data-testid="question-input"
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder={thread?.status === "waiting_for_clarification" ? "补充缺失的产品、机台、时间或 Lot 信息" : "输入半导体制造或检测问题"}
          rows={3}
        />
        <button type="submit" className="icon-command" title="发送问题" disabled={loading || !query.trim()}>
          <ArrowUp size={19} />
        </button>
      </form>
    </div>

    <aside className="evidence-panel">
      <div className="panel-heading">
        <div><span className="section-label">EVIDENCE DRAWER</span><h2>证据与资产</h2></div>
        {result?.trace_id && <button className="icon-button" type="button" title="打开检索 Trace" onClick={() => onOpenTrace(result.trace_id)}><SearchCheck size={18} /></button>}
      </div>

      {asset ? <figure className="asset-preview">
        <img className="wafer-image" src={asset.url} alt={`受控图片 ${asset.image_id}`} />
        <figcaption><strong>{asset.image_id}</strong><span>授权有效至 {new Date(asset.expires_at).toLocaleTimeString("zh-CN")}</span></figcaption>
      </figure> : <div className="asset-placeholder"><ImageIcon size={30} /><span>暂无已打开图片</span></div>}

      {imageIds.length > 0 && <div className="asset-list">
        {imageIds.map((imageId) => <button className="asset-item" type="button" key={imageId} onClick={() => onOpenImage(imageId)}>
          <ImageIcon size={16} /><span>{imageId}</span>
        </button>)}
      </div>}

      <div className="evidence-list" data-testid="evidence-list">
        {ledger.map((item) => <button
          className={`evidence-item ${selectedEvidence?.evidence_id === item.evidence_id ? "active" : ""}`}
          type="button"
          key={item.evidence_id}
          onClick={() => setSelectedEvidenceId(item.evidence_id)}
        >
          <span>{sourceLabel(item.source_type)}</span>
          <strong>{item.document_id ?? item.evidence_id}</strong>
          <small>{item.page_or_section || item.tool_id || "结构化证据"}</small>
        </button>)}
      </div>

      {selectedEvidence && <EvidenceDetail evidence={selectedEvidence} />}
    </aside>
  </section>;
}

function StructuredAnswer({ result, onOpenTrace }: { result: AgentResponse; onOpenTrace: (traceId?: string | null) => void }) {
  const answer = result.answer!;
  return <section className="structured-answer" data-testid="structured-answer">
    <div className="structured-heading">
      <div><span className="section-label">VERIFIED INVESTIGATION</span><h3>结构化调查结果</h3></div>
      <div className="answer-badges"><StatusPill value={result.status ?? "completed"} /><span className={`confidence confidence-${answer.confidence}`}>{answer.confidence}</span></div>
    </div>
    <AnswerSection icon={<CheckCircle2 size={17} />} title="已知事实" claims={answer.facts.map((item) => item.text)} />
    <AnswerSection icon={<FlaskConical size={17} />} title="待验证假设" claims={answer.hypotheses.map((item) => item.text)} />
    <AnswerSection icon={<AlertTriangle size={17} />} title="未知项" claims={answer.unknowns} />
    <AnswerSection icon={<ListChecks size={17} />} title="下一步" claims={answer.next_actions} />
    {(result.verification_warnings ?? []).length > 0 && <div className="verification-warning"><AlertTriangle size={15} />{result.verification_warnings!.join(" / ")}</div>}
    {result.trace_id && <button type="button" className="text-command" onClick={() => onOpenTrace(result.trace_id)}><SearchCheck size={16} />查看证据选择过程</button>}
  </section>;
}

function AnswerSection({ icon, title, claims }: { icon: React.ReactNode; title: string; claims: string[] }) {
  if (!claims.length) return null;
  return <div className="answer-section"><h4>{icon}{title}</h4><ul>{claims.map((claim, index) => <li key={`${title}-${index}`}>{claim}</li>)}</ul></div>;
}

function EvidenceDetail({ evidence }: { evidence: EvidenceLedgerEntry }) {
  return <div className="evidence-detail">
    <div className="detail-title"><span className="section-label">SELECTED EVIDENCE</span><StatusPill value={evidence.approval_status ?? evidence.source_type} /></div>
    <p>{evidence.content}</p>
    <dl>
      <div><dt>来源</dt><dd>{evidence.document_id ?? evidence.source_type}</dd></div>
      <div><dt>Revision</dt><dd>{evidence.revision ?? "-"}</dd></div>
      <div><dt>位置</dt><dd>{evidence.page_or_section || "-"}</dd></div>
      <div><dt>Rerank</dt><dd>{evidence.rerank_score?.toFixed(3) ?? "-"}</dd></div>
      <div><dt>路由</dt><dd>{evidence.retrieval_routes?.join(" + ") || "-"}</dd></div>
      <div><dt>机台</dt><dd>{[evidence.tool_id, evidence.chamber && `Chamber ${evidence.chamber}`].filter(Boolean).join(" / ") || "-"}</dd></div>
    </dl>
  </div>;
}

function sourceLabel(value: string): string {
  return ({ internal_chunk: "受控文档", tool: "只读工具", external: "外部资料" } as Record<string, string>)[value] ?? value;
}
