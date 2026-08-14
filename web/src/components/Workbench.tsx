import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  ArrowUp,
  Bot,
  CheckCircle2,
  FileText,
  FlaskConical,
  Image as ImageIcon,
  ListChecks,
  LoaderCircle,
  Plus,
  RotateCcw,
  SearchCheck,
  Square
} from "lucide-react";
import type { AgentResponse, AssetAccess, EvidenceLedgerEntry, MessagePresentation, StreamUiState, Thread } from "../types";
import { StatusPill } from "./Common";

type Props = {
  threads: Thread[];
  thread: Thread | null;
  result: AgentResponse | null;
  query: string;
  loading: boolean;
  asset: AssetAccess | null;
  streamState: StreamUiState | null;
  onQueryChange: (value: string) => void;
  onSubmit: (event: FormEvent) => void;
  onSelectThread: (threadId: string) => void;
  onNewThread: () => void;
  onOpenImage: (imageId: string) => void;
  onOpenTrace: (traceId?: string | null) => void;
  onStopStream: () => void;
  onRetryStream: () => void;
};

export function Workbench({
  threads,
  thread,
  result,
  query,
  loading,
  asset,
  streamState,
  onQueryChange,
  onSubmit,
  onSelectThread,
  onNewThread,
  onOpenImage,
  onOpenTrace,
  onStopStream,
  onRetryStream
}: Props) {
  const ledger = result?.evidence_ledger ?? [];
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string>("");
  const selectedEvidence = ledger.find((item) => item.evidence_id === selectedEvidenceId) ?? ledger[0];
  const messageListRef = useRef<HTMLDivElement>(null);
  const imageIds = useMemo(() => {
    const fromResult = result?.image_asset_ids ?? [];
    const fromLedger = ledger.flatMap((item) => item.image_ids ?? []);
    const fromMessages = thread?.messages.flatMap((message) =>
      (message.citations ?? []).flatMap((citation) => citation.image_ids ?? [])
    ) ?? [];
    return [...new Set([...fromResult, ...fromLedger, ...fromMessages])];
  }, [ledger, result, thread]);

  useEffect(() => {
    const list = messageListRef.current;
    if (list) list.scrollTop = list.scrollHeight;
  }, [streamState?.partialAnswer, thread?.messages.length]);

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
            disabled={streamState?.status === "running"}
          >
            {threads.map((item) => <option key={item.thread_id} value={item.thread_id}>{item.title}</option>)}
          </select>
          <button className="icon-button" type="button" title="新建调查线程" onClick={onNewThread} disabled={streamState?.status === "running"}>
            <Plus size={18} />
          </button>
        </div>
      </div>

      {thread?.status === "waiting_for_clarification" && <div className="clarification-band" data-testid="clarification-band">
        <AlertTriangle size={18} />
        <div><strong>等待补充信息 · 第 {thread.clarification_round} 轮</strong><span>{thread.pending_fields.join(" / ")}</span></div>
      </div>}

      <div className="message-list" data-testid="message-list" ref={messageListRef}>
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
                onClick={() => onOpenTrace(message.presentation?.trace_id)}
              >
                <FileText size={13} />
                {citation.document_id ?? citation.source_type ?? "证据"} {citation.revision ?? ""}
                {citation.page_or_section ? ` · ${citation.page_or_section}` : ""}
              </button>)}
            </div>}
            {message.role === "assistant"
              && message.presentation?.mode === "structured_card"
              && message.presentation.answer
              && <StructuredAnswer presentation={message.presentation} onOpenTrace={onOpenTrace} />}
          </div>
        </article>) : <div className="empty-conversation">当前线程尚无消息</div>}

        {streamState && <StreamMessage
          stream={streamState}
          onStop={onStopStream}
          onRetry={onRetryStream}
        />}
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

function StreamMessage({
  stream,
  onStop,
  onRetry
}: {
  stream: StreamUiState;
  onStop: () => void;
  onRetry: () => void;
}) {
  const running = stream.status === "running";
  return <article className={`message assistant stream-message stream-${stream.status}`} data-testid="streaming-message">
    <div className="message-avatar">{running ? <LoaderCircle className="spin" size={17} /> : <Bot size={17} />}</div>
    <div className="message-body">
      <div className="stream-heading">
        <div>
          <div className="message-role">SEMIKB · LIVE</div>
          <strong data-testid="stream-stage">{stream.stageMessage}</strong>
        </div>
        {running
          ? <button className="icon-button stream-control" type="button" title="停止生成" aria-label="停止生成" onClick={onStop}><Square size={14} /></button>
          : stream.retryable && <button className="text-command stream-retry" type="button" onClick={onRetry}><RotateCcw size={14} />重试</button>}
      </div>
      {(stream.internalEvidenceCount > 0 || stream.externalEvidenceCount > 0) && <div className="stream-evidence-count" data-testid="stream-evidence-count">
        受控证据 {stream.internalEvidenceCount} · 外部证据 {stream.externalEvidenceCount}
      </div>}
      <p className={`stream-answer ${stream.partialAnswer ? "has-content" : ""}`} data-testid="streaming-answer">
        {stream.partialAnswer || (running ? stageLabel(stream.stage) : "本次未保存不完整回答。")}
      </p>
      {stream.elapsedMs >= 1000 && <small className="stream-elapsed">已等待 {Math.floor(stream.elapsedMs / 1000)} 秒</small>}
    </div>
  </article>;
}

function stageLabel(stage?: StreamUiState["stage"]): string {
  return ({
    analyzing_request: "正在理解问题与当前会话…",
    awaiting_clarification: "正在判断还需要哪些关键信息…",
    retrieving_evidence: "正在检索受控知识库…",
    searching_external: "正在查询允许的外部资料…",
    reranking_evidence: "正在重排并选择证据…",
    generating_answer: "正在生成并校验回答…",
    verifying_answer: "正在校验引用和结论边界…",
    persisting_result: "正在保存最终结果…"
  } as Record<string, string>)[stage ?? ""] ?? "请求处理中…";
}

function StructuredAnswer({ presentation, onOpenTrace }: { presentation: MessagePresentation; onOpenTrace: (traceId?: string | null) => void }) {
  const answer = presentation.answer!;
  return <section className="structured-answer" data-testid="structured-answer">
    <div className="structured-heading">
      <div><span className="section-label">VERIFIED INVESTIGATION</span><h3>结构化调查结果</h3></div>
      <div className="answer-badges"><StatusPill value={presentation.status ?? "completed"} /><span className={`confidence confidence-${answer.confidence}`}>{answer.confidence}</span></div>
    </div>
    <AnswerSection icon={<CheckCircle2 size={17} />} title="已知事实" claims={answer.facts.map((item) => item.text)} />
    <AnswerSection icon={<FlaskConical size={17} />} title="待验证假设" claims={answer.hypotheses.map((item) => item.text)} />
    <AnswerSection icon={<AlertTriangle size={17} />} title="未知项" claims={answer.unknowns} />
    <AnswerSection icon={<ListChecks size={17} />} title="下一步" claims={answer.next_actions} />
    {(presentation.verification_warnings ?? []).length > 0 && <div className="verification-warning"><AlertTriangle size={15} />{presentation.verification_warnings.join(" / ")}</div>}
    {presentation.trace_id && <button type="button" className="text-command" onClick={() => onOpenTrace(presentation.trace_id)}><SearchCheck size={16} />查看证据选择过程</button>}
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
