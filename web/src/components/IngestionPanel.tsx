import { type FormEvent, useState } from "react";
import {
  AlertTriangle,
  FileUp,
  History,
  RefreshCw,
  RotateCcw,
  Upload,
  X
} from "lucide-react";
import type { IngestionJob, UploadMetadata } from "../types";
import { EmptyState, formatDate, labelForStage, Metric, StatusPill } from "./Common";

type Props = {
  jobs: IngestionJob[];
  selectedJob?: IngestionJob;
  loading: boolean;
  onSelect: (jobId: string) => void;
  onUpload: (file: File, metadata: UploadMetadata) => Promise<boolean>;
  onRetry: (jobId: string) => Promise<void>;
  onRefresh: () => Promise<void>;
};

const initialMetadata = (): UploadMetadata => ({
  document_id: `UI-DEMO-${Date.now().toString().slice(-8)}`,
  revision: "R1",
  title: "UI 入库验收资料",
  document_type: "training_note",
  source_kind: "user_upload",
  source_license: "internal",
  access_scope_key: "demo_engineering",
  fab: "FAB-01",
  product: "P-ALPHA",
  process_layer: "ETCH",
  tool_id: "ETCH-03",
  chamber: "B"
});

export function IngestionPanel({ jobs, selectedJob, loading, onSelect, onUpload, onRetry, onRefresh }: Props) {
  const [showUpload, setShowUpload] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [metadata, setMetadata] = useState<UploadMetadata>(initialMetadata);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!file) return;
    if (!await onUpload(file, metadata)) return;
    setFile(null);
    setMetadata(initialMetadata());
    setShowUpload(false);
  }

  return <section className="operations-layout">
    <header className="operations-heading">
      <div><span className="section-label">INGESTION OPERATIONS</span><h2>入库任务中心</h2></div>
      <div className="heading-actions">
        <button className="icon-button" type="button" title="刷新任务" onClick={() => void onRefresh()} disabled={loading}><RefreshCw size={17} className={loading ? "spin" : ""} /></button>
        <button className="command-button" type="button" onClick={() => setShowUpload((value) => !value)}><Upload size={17} />上传文档</button>
      </div>
    </header>

    {showUpload && <form className="upload-band" onSubmit={submit} data-testid="upload-form">
      <div className="band-heading"><div><span className="section-label">NEW INGESTION JOB</span><h3>文档与治理元数据</h3></div><button className="icon-button" type="button" title="关闭上传" onClick={() => setShowUpload(false)}><X size={17} /></button></div>
      <label className="file-drop"><FileUp size={24} /><span>{file?.name ?? "选择 PDF、Office、表格、图片或结构化文本"}</span><input data-testid="ingestion-file" type="file" accept=".pdf,.docx,.xlsx,.csv,.pptx,.png,.jpg,.jpeg,.md,.txt,.html,.htm" onChange={(event) => setFile(event.target.files?.[0] ?? null)} /></label>
      <div className="form-grid">
        <Field label="Document ID" value={metadata.document_id} onChange={(value) => setMetadata({ ...metadata, document_id: value })} />
        <Field label="Revision" value={metadata.revision} onChange={(value) => setMetadata({ ...metadata, revision: value })} />
        <Field label="标题" value={metadata.title} onChange={(value) => setMetadata({ ...metadata, title: value })} wide />
        <Field label="文档类型" value={metadata.document_type} onChange={(value) => setMetadata({ ...metadata, document_type: value })} />
        <Field label="Fab" value={metadata.fab} onChange={(value) => setMetadata({ ...metadata, fab: value })} />
        <Field label="Product" value={metadata.product} onChange={(value) => setMetadata({ ...metadata, product: value })} />
        <Field label="工艺层" value={metadata.process_layer ?? ""} onChange={(value) => setMetadata({ ...metadata, process_layer: value })} />
        <Field label="Tool" value={metadata.tool_id ?? ""} onChange={(value) => setMetadata({ ...metadata, tool_id: value })} />
        <Field label="Chamber" value={metadata.chamber ?? ""} onChange={(value) => setMetadata({ ...metadata, chamber: value })} />
      </div>
      <div className="form-actions"><button className="command-button" type="submit" disabled={loading || !file}><Upload size={17} />提交入库</button></div>
    </form>}

    <div className="operations-split">
      <div className="data-table-wrap"><table>
        <thead><tr><th>文档</th><th>阶段</th><th>进度</th><th>产物</th><th>尝试</th><th>更新时间</th></tr></thead>
        <tbody>{jobs.map((job) => <tr key={job.job_id} className={selectedJob?.job_id === job.job_id ? "active-row" : ""}>
          <td><button className="table-link" type="button" onClick={() => onSelect(job.job_id)}><strong>{job.document_id} {job.revision}</strong><span>{job.filename}</span></button></td>
          <td><StatusPill value={job.status} /><span>{labelForStage(job.current_stage)}</span></td>
          <td><div className="progress"><span style={{ width: `${job.progress}%` }} /></div><small>{job.progress}%</small></td>
          <td><strong>{job.chunks_count} Chunk</strong><span>{job.images_count} 图 / {job.tables_count} 表</span></td>
          <td>{job.attempt}</td>
          <td>{formatDate(job.finished_at ?? job.started_at ?? job.created_at)}</td>
        </tr>)}</tbody>
      </table>{!jobs.length && <EmptyState icon={<History size={30} />} title="暂无入库任务" />}</div>

      {selectedJob && <aside className="job-detail" data-testid="job-detail">
        <div className="detail-title"><div><span className="section-label">JOB DETAIL</span><h3>{selectedJob.document_id} {selectedJob.revision}</h3></div><StatusPill value={selectedJob.status} /></div>
        <div className="detail-metrics"><Metric label="进度" value={`${selectedJob.progress}%`} /><Metric label="尝试" value={selectedJob.attempt} /></div>
        <dl className="compact-dl">
          <div><dt>Job ID</dt><dd>{selectedJob.job_id}</dd></div>
          <div><dt>Parser</dt><dd>{selectedJob.parser_version}</dd></div>
          <div><dt>Chunker</dt><dd>{selectedJob.chunker_version}</dd></div>
          <div><dt>Embedding</dt><dd>{selectedJob.embedding_version}</dd></div>
          <div><dt>Index</dt><dd>{selectedJob.index_version}</dd></div>
          <div><dt>发起人</dt><dd>{selectedJob.created_by}</dd></div>
        </dl>
        {selectedJob.safe_error_summary && <div className="warning-band"><AlertTriangle size={17} /><div><strong>{selectedJob.error_code ?? "FAILED"}</strong><span>{selectedJob.safe_error_summary}</span></div></div>}
        <div className="event-timeline">
          <h4><History size={16} />阶段事件</h4>
          {selectedJob.events.map((event) => <div className="event-item" key={event.event_id}><span className="event-marker" /><div><strong>{labelForStage(event.stage)} · {event.progress}%</strong><p>{event.message}</p><small>{formatDate(event.created_at)} · attempt {event.attempt}</small></div></div>)}
        </div>
        {selectedJob.status === "failed" && <button className="command-button" type="button" disabled={loading} onClick={() => void onRetry(selectedJob.job_id)}><RotateCcw size={17} />重试失败任务</button>}
      </aside>}
    </div>
  </section>;
}

function Field({ label, value, onChange, wide = false }: { label: string; value: string; onChange: (value: string) => void; wide?: boolean }) {
  return <label className={wide ? "field-wide" : ""}><span>{label}</span><input required value={value} onChange={(event) => onChange(event.target.value)} /></label>;
}
