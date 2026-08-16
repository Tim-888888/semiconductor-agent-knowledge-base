import { type FormEvent, useState } from "react";
import {
  AlertTriangle,
  ArchiveX,
  BookOpenCheck,
  FileText,
  FileUp,
  FlaskConical,
  FolderArchive,
  History,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  Upload,
  X
} from "lucide-react";
import type {
  CorpusStandardizationJob,
  CorpusStandardizationMetadata,
  DocumentLifecycleOperationRecord,
  IngestionJob,
  KnowledgeDocumentRevisionSummary,
  KnowledgeDocumentSummary,
  UploadMetadata
} from "../types";
import { EmptyState, formatDate, labelForStage, Metric, StatusPill } from "./Common";

type Props = {
  jobs: IngestionJob[];
  selectedJob?: IngestionJob;
  corpusJobs: CorpusStandardizationJob[];
  selectedCorpusJob?: CorpusStandardizationJob;
  documents: KnowledgeDocumentSummary[];
  selectedDocument?: KnowledgeDocumentSummary;
  revisions: KnowledgeDocumentRevisionSummary[];
  selectedRevision?: KnowledgeDocumentRevisionSummary;
  lifecycleOperation?: DocumentLifecycleOperationRecord;
  loading: boolean;
  onSelect: (jobId: string) => void;
  onUpload: (file: File, metadata: UploadMetadata) => Promise<boolean>;
  onRetry: (jobId: string) => Promise<void>;
  onSelectCorpus: (jobId: string) => void;
  onUploadCorpus: (
    files: File[],
    metadata: CorpusStandardizationMetadata,
    sidecar: string
  ) => Promise<boolean>;
  onRetryCorpus: (jobId: string) => Promise<void>;
  onRefresh: () => Promise<void>;
  onSelectDocument: (documentId: string) => void;
  onSelectRevision: (revision: KnowledgeDocumentRevisionSummary) => void;
  onLifecycleAction: (
    action: "withdraw" | "restore",
    revision: KnowledgeDocumentRevisionSummary,
    reason: string
  ) => Promise<void>;
  onRetryLifecycle: (operationId: string) => Promise<void>;
};

const initialCorpusMetadata = (): CorpusStandardizationMetadata => ({
  corpus_id: `corpus-${Date.now().toString().slice(-8)}`,
  snapshot_version: "v1",
  display_name: "",
  source_kind: "user_upload",
  source_license: "unknown",
  corpus_kind: "auto"
});

const sidecarTemplate = JSON.stringify({
  sidecar_schema_version: "semikb-corpus-sidecar-v1",
  profile: {
    profile_schema_version: "semikb-corpus-profile-v1",
    corpus_kind: "auto",
    include_globs: ["**/*", "*"],
    exclude_globs: ["**/.DS_Store", "**/Thumbs.db", "**/__MACOSX/**"],
    role_rules: [],
    relation_rules: [],
    tabular_sample_rows: 200,
    tabular_max_columns: 256,
    generate_image_text: true
  },
  files: [],
  relations: []
}, null, 2);

const initialMetadata = (): UploadMetadata => ({
  document_id: `DOC-${Date.now().toString().slice(-8)}`,
  revision: "R1",
  title: "",
  document_type: "unknown",
  approval_status: "draft",
  lifecycle: "staged",
  source_kind: "user_upload",
  source_license: "unknown",
  retrieval_policy: "standard"
});

const demoMetadata = (): UploadMetadata => ({
  document_id: `UI-DEMO-${Date.now().toString().slice(-8)}`,
  revision: "R1",
  title: "UI 入库验收资料",
  document_type: "training_note",
  approval_status: "approved",
  lifecycle: "published",
  source_kind: "synthetic",
  source_license: "CC0-1.0",
  source_id: "semikb.demo.synthetic",
  source_manifest_version: "1.0.0",
  dataset_version: "demo-v2",
  source_license_status: "verified",
  redistribution_policy: "allowed",
  access_scope_key: "demo_engineering",
  fab: "FAB-01",
  product: "P-ALPHA",
  process_layer: "ETCH",
  tool_id: "ETCH-03",
  chamber: "B",
  retrieval_policy: "standard"
});

export function IngestionPanel(props: Props) {
  const {
    jobs,
    selectedJob,
    corpusJobs,
    selectedCorpusJob,
    documents,
    selectedDocument,
    revisions,
    selectedRevision,
    lifecycleOperation,
    loading,
    onSelect,
    onUpload,
    onRetry,
    onSelectCorpus,
    onUploadCorpus,
    onRetryCorpus,
    onRefresh,
    onSelectDocument,
    onSelectRevision,
    onLifecycleAction,
    onRetryLifecycle
  } = props;
  const [mode, setMode] = useState<"jobs" | "corpus" | "documents">("jobs");
  const [showUpload, setShowUpload] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [metadata, setMetadata] = useState<UploadMetadata>(initialMetadata);
  const [corpusFiles, setCorpusFiles] = useState<File[]>([]);
  const [corpusMetadata, setCorpusMetadata] = useState<CorpusStandardizationMetadata>(initialCorpusMetadata);
  const [sidecar, setSidecar] = useState(sidecarTemplate);
  const [lifecycleAction, setLifecycleAction] = useState<"withdraw" | "restore" | null>(null);
  const [reason, setReason] = useState("");

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!file) return;
    if (!await onUpload(file, metadata)) return;
    setFile(null);
    setMetadata(initialMetadata());
    setShowUpload(false);
  }

  async function submitLifecycle(event: FormEvent) {
    event.preventDefault();
    if (!selectedRevision || !lifecycleAction || reason.trim().length < 8) return;
    await onLifecycleAction(lifecycleAction, selectedRevision, reason.trim());
    setLifecycleAction(null);
    setReason("");
  }

  async function submitCorpus(event: FormEvent) {
    event.preventDefault();
    if (!corpusFiles.length) return;
    if (!await onUploadCorpus(corpusFiles, corpusMetadata, sidecar)) return;
    setCorpusFiles([]);
    setCorpusMetadata(initialCorpusMetadata());
    setSidecar(sidecarTemplate);
    setShowUpload(false);
  }

  function switchMode(next: "jobs" | "corpus" | "documents") {
    setMode(next);
    setShowUpload(false);
    setLifecycleAction(null);
    setReason("");
  }

  function selectFile(nextFile: File | null) {
    setFile(nextFile);
    if (!nextFile) return;
    const basename = nextFile.name.replace(/\.[^.]+$/, "");
    setMetadata((current) => ({
      ...current,
      title: current.title || basename,
      document_id: current.document_id.startsWith("DOC-")
        ? `DOC-${basename.replace(/[^A-Za-z0-9]+/g, "-").replace(/^-|-$/g, "").toUpperCase().slice(0, 72) || Date.now().toString().slice(-8)}`
        : current.document_id
    }));
  }

  return <section className="operations-layout">
    <header className="operations-heading">
      <div><span className="section-label">KNOWLEDGE OPERATIONS</span><h2>入库与知识文档</h2></div>
      <div className="heading-actions">
        <div className="segmented" aria-label="知识运营视图">
          <button type="button" className={mode === "jobs" ? "active" : ""} onClick={() => switchMode("jobs")}>入库任务</button>
          <button type="button" className={mode === "corpus" ? "active" : ""} onClick={() => switchMode("corpus")}>语料标准化</button>
          <button type="button" className={mode === "documents" ? "active" : ""} onClick={() => switchMode("documents")}>知识文档</button>
        </div>
        <button className="icon-button" type="button" title="刷新数据" onClick={() => void onRefresh()} disabled={loading}><RefreshCw size={17} className={loading ? "spin" : ""} /></button>
        {mode === "jobs" && <button className="command-button" type="button" onClick={() => setShowUpload((value) => !value)}><Upload size={17} />上传文档</button>}
        {mode === "corpus" && <button className="command-button" type="button" onClick={() => setShowUpload((value) => !value)}><FolderArchive size={17} />新建快照</button>}
      </div>
    </header>

    {mode === "jobs" && <>
      {showUpload && <form className="upload-band" onSubmit={submit} data-testid="upload-form">
        <div className="band-heading"><div><span className="section-label">NEW INGESTION JOB</span><h3>文档与治理元数据</h3></div><div className="heading-actions"><button className="text-command" type="button" onClick={() => setMetadata(demoMetadata())}><FlaskConical size={16} />加载演示配置</button><button className="icon-button" type="button" title="关闭上传" onClick={() => setShowUpload(false)}><X size={17} /></button></div></div>
        <div className="warning-band"><ShieldCheck size={16} /><div><strong>待复核</strong><span>新文件默认以 draft / staged 保存，不会进入活动检索索引。</span></div></div>
        <label className="file-drop"><FileUp size={24} /><span>{file?.name ?? "选择 PDF、Office、表格、图片或结构化文本"}</span><input data-testid="ingestion-file" type="file" accept=".pdf,.docx,.xlsx,.csv,.pptx,.png,.jpg,.jpeg,.md,.txt,.html,.htm" onChange={(event) => selectFile(event.target.files?.[0] ?? null)} /></label>
        <div className="form-grid">
          <Field label="Document ID" value={metadata.document_id} onChange={(value) => setMetadata({ ...metadata, document_id: value })} required />
          <Field label="Revision" value={metadata.revision} onChange={(value) => setMetadata({ ...metadata, revision: value })} required />
          <Field label="标题" value={metadata.title} onChange={(value) => setMetadata({ ...metadata, title: value })} wide required />
          <Field label="文档类型" value={metadata.document_type} onChange={(value) => setMetadata({ ...metadata, document_type: value })} required />
          <SelectField label="审批状态" value={metadata.approval_status} options={["draft", "approved", "rejected"]} onChange={(value) => setMetadata({ ...metadata, approval_status: value as UploadMetadata["approval_status"] })} />
          <SelectField label="目标生命周期" value={metadata.lifecycle} options={["staged", "published", "quarantined"]} onChange={(value) => setMetadata({ ...metadata, lifecycle: value as UploadMetadata["lifecycle"] })} />
          <SelectField label="检索保护策略" value={metadata.retrieval_policy} options={["standard", "protected"]} onChange={(value) => setMetadata({ ...metadata, retrieval_policy: value as UploadMetadata["retrieval_policy"] })} />
          <Field label="权限 Scope" value={metadata.access_scope_key ?? ""} onChange={(value) => setMetadata({ ...metadata, access_scope_key: value })} />
          <Field label="来源类型" value={metadata.source_kind} onChange={(value) => setMetadata({ ...metadata, source_kind: value })} />
          <Field label="来源许可" value={metadata.source_license} onChange={(value) => setMetadata({ ...metadata, source_license: value })} />
          <Field label="Source ID" value={metadata.source_id ?? ""} onChange={(value) => setMetadata({ ...metadata, source_id: value || undefined })} />
          <Field label="Manifest Version" value={metadata.source_manifest_version ?? ""} onChange={(value) => setMetadata({ ...metadata, source_manifest_version: value || undefined })} />
          <Field label="Dataset Version" value={metadata.dataset_version ?? ""} onChange={(value) => setMetadata({ ...metadata, dataset_version: value || undefined })} />
          <SelectField label="许可审计" value={metadata.source_license_status ?? ""} options={["", "verified", "declared", "unclear", "restricted"]} onChange={(value) => setMetadata({ ...metadata, source_license_status: value ? value as UploadMetadata["source_license_status"] : undefined })} />
          <SelectField label="再分发策略" value={metadata.redistribution_policy ?? ""} options={["", "allowed", "restricted", "prohibited", "unknown"]} onChange={(value) => setMetadata({ ...metadata, redistribution_policy: value ? value as UploadMetadata["redistribution_policy"] : undefined })} />
          <Field label="Fab" value={metadata.fab ?? ""} onChange={(value) => setMetadata({ ...metadata, fab: value })} />
          <Field label="Product" value={metadata.product ?? ""} onChange={(value) => setMetadata({ ...metadata, product: value })} />
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
    </>}

    {mode === "corpus" && <>
      {showUpload && <form className="upload-band" onSubmit={submitCorpus} data-testid="corpus-upload-form">
        <div className="band-heading"><div><span className="section-label">CORPUS SNAPSHOT</span><h3>通用语料标准化</h3></div><button className="icon-button" type="button" title="关闭上传" onClick={() => setShowUpload(false)}><X size={17} /></button></div>
        <div className="warning-band"><ShieldCheck size={16} /><div><strong>仅进入审核区</strong><span>原件和派生物保存到私有存储，不创建知识文档、Chunk 或 Milvus 向量。</span></div></div>
        <label className="file-drop"><FolderArchive size={24} /><span>{corpusFiles.length ? `已选择 ${corpusFiles.length} 个文件` : "选择多个文件或一个 ZIP 快照"}</span><input data-testid="corpus-files" type="file" multiple onChange={(event) => {
          const nextFiles = Array.from(event.target.files ?? []);
          setCorpusFiles(nextFiles);
          if (nextFiles.length && !corpusMetadata.display_name) {
            const displayName = nextFiles.length === 1
              ? nextFiles[0].name.replace(/\.[^.]+$/, "")
              : `语料快照 ${new Date().toLocaleDateString()}`;
            setCorpusMetadata({ ...corpusMetadata, display_name: displayName });
          }
        }} /></label>
        <div className="form-grid">
          <Field label="Corpus ID" value={corpusMetadata.corpus_id} onChange={(value) => setCorpusMetadata({ ...corpusMetadata, corpus_id: value })} required />
          <Field label="Snapshot Version" value={corpusMetadata.snapshot_version} onChange={(value) => setCorpusMetadata({ ...corpusMetadata, snapshot_version: value })} required />
          <Field label="显示名称" value={corpusMetadata.display_name} onChange={(value) => setCorpusMetadata({ ...corpusMetadata, display_name: value })} wide required />
          <SelectField label="语料类型" value={corpusMetadata.corpus_kind} options={["auto", "document_collection", "tabular_dataset", "image_corpus", "mixed"]} onChange={(value) => setCorpusMetadata({ ...corpusMetadata, corpus_kind: value as CorpusStandardizationMetadata["corpus_kind"] })} />
          <Field label="来源类型" value={corpusMetadata.source_kind} onChange={(value) => setCorpusMetadata({ ...corpusMetadata, source_kind: value })} required />
          <Field label="来源许可" value={corpusMetadata.source_license} onChange={(value) => setCorpusMetadata({ ...corpusMetadata, source_license: value })} required />
          <Field label="来源 URL" value={corpusMetadata.source_uri ?? ""} onChange={(value) => setCorpusMetadata({ ...corpusMetadata, source_uri: value })} wide />
          <Field label="权限 Scope" value={corpusMetadata.access_scope_key ?? ""} onChange={(value) => setCorpusMetadata({ ...corpusMetadata, access_scope_key: value })} />
          <Field label="使用限制" value={corpusMetadata.use_restrictions ?? ""} onChange={(value) => setCorpusMetadata({ ...corpusMetadata, use_restrictions: value })} wide />
        </div>
        <label className="sidecar-editor"><span>Sidecar / Profile JSON</span><textarea data-testid="corpus-sidecar" rows={14} spellCheck={false} value={sidecar} onChange={(event) => setSidecar(event.target.value)} /></label>
        <div className="form-actions"><button className="command-button" type="submit" disabled={loading || !corpusFiles.length || !corpusMetadata.display_name.trim()}><FolderArchive size={17} />提交标准化</button></div>
      </form>}

      <div className="operations-split">
        <div className="data-table-wrap"><table>
          <thead><tr><th>语料快照</th><th>阶段</th><th>进度</th><th>清单</th><th>警告</th><th>更新时间</th></tr></thead>
          <tbody>{corpusJobs.map((job) => <tr key={job.job_id} className={selectedCorpusJob?.job_id === job.job_id ? "active-row" : ""}>
            <td><button className="table-link" type="button" onClick={() => onSelectCorpus(job.job_id)}><strong>{job.metadata.display_name}</strong><span>{job.metadata.corpus_id} · {job.metadata.snapshot_version}</span></button></td>
            <td><StatusPill value={job.status} /></td>
            <td><div className="progress"><span style={{ width: `${job.progress}%` }} /></div><small>{job.progress}%</small></td>
            <td><strong>{job.files_count} 文件</strong><span>{job.documents_count} 文档 / {job.tables_count} 表 / {job.images_count} 图</span></td>
            <td>{job.report?.warning_codes.length ?? job.unsupported_count}</td>
            <td>{formatDate(job.finished_at ?? job.started_at ?? job.created_at)}</td>
          </tr>)}</tbody>
        </table>{!corpusJobs.length && <EmptyState icon={<FolderArchive size={30} />} title="暂无语料标准化任务" />}</div>

        {selectedCorpusJob && <aside className="job-detail" data-testid="corpus-job-detail">
          <div className="detail-title"><div><span className="section-label">STANDARDIZATION REVIEW</span><h3>{selectedCorpusJob.metadata.display_name}</h3></div><StatusPill value={selectedCorpusJob.status} /></div>
          <div className="detail-metrics"><Metric label="文件" value={selectedCorpusJob.files_count} /><Metric label="未支持" value={selectedCorpusJob.unsupported_count} /></div>
          <dl className="compact-dl">
            <div><dt>Job ID</dt><dd>{selectedCorpusJob.job_id}</dd></div>
            <div><dt>Snapshot Hash</dt><dd>{selectedCorpusJob.snapshot_hash.slice(0, 16)}...</dd></div>
            <div><dt>识别类型</dt><dd>{selectedCorpusJob.report?.inferred_corpus_kind ?? "等待识别"}</dd></div>
            <div><dt>来源许可</dt><dd>{selectedCorpusJob.metadata.source_license}</dd></div>
          </dl>
          {selectedCorpusJob.safe_error_summary && <div className="warning-band"><AlertTriangle size={17} /><div><strong>{selectedCorpusJob.error_code}</strong><span>{selectedCorpusJob.safe_error_summary}</span></div></div>}
          {selectedCorpusJob.report && <>
            <div className="warning-band"><ShieldCheck size={16} /><div><strong>等待人工审核</strong><span>{selectedCorpusJob.report.review_reasons.join(" · ")}</span></div></div>
            <div className="corpus-file-list"><h4><FileText size={16} />逐文件处理清单</h4>{selectedCorpusJob.report.files.map((file) => <div className="corpus-file-item" key={file.file_id}><div><strong>{file.relative_path}</strong><span>{file.role} · {file.parser_name ?? "未路由"}</span></div><span>{file.standardized_ref ? "已标准化" : "仅保留原件"}</span>{file.warning_codes.length > 0 && <small>{file.warning_codes.join(" · ")}</small>}</div>)}</div>
            <div className="compact-dl"><div><dt>显式关系</dt><dd>{selectedCorpusJob.report.relations.length}</dd></div><div><dt>警告代码</dt><dd>{selectedCorpusJob.report.warning_codes.join(" · ") || "无"}</dd></div></div>
          </>}
          <div className="event-timeline"><h4><History size={16} />阶段事件</h4>{selectedCorpusJob.events.map((event) => <div className="event-item" key={event.event_id}><span className="event-marker" /><div><strong>{event.status} · {event.progress}%</strong><p>{event.message}</p><small>{formatDate(event.created_at)} · attempt {event.attempt}</small></div></div>)}</div>
          {selectedCorpusJob.status === "failed" && <button className="command-button" type="button" disabled={loading} onClick={() => void onRetryCorpus(selectedCorpusJob.job_id)}><RotateCcw size={17} />重试标准化</button>}
        </aside>}
      </div>
    </>}

    {mode === "documents" && <div className="operations-split knowledge-documents-view">
      <div className="data-table-wrap"><table>
        <thead><tr><th>知识文档</th><th>当前 Revision</th><th>状态</th><th>版本数</th><th>来源</th><th>更新时间</th></tr></thead>
        <tbody>{documents.map((document) => <tr key={document.document_id} className={selectedDocument?.document_id === document.document_id ? "active-row" : ""}>
          <td><button className="table-link" type="button" onClick={() => onSelectDocument(document.document_id)}><strong>{document.document_id}</strong><span>{document.title}</span></button></td>
          <td>{document.current_revision ?? "-"}</td>
          <td>{document.current_lifecycle ? <StatusPill value={document.current_lifecycle} /> : "-"}</td>
          <td>{document.revision_count}</td>
          <td><strong>{document.source_id ?? "internal"}</strong><span>{document.dataset_version ?? "未关联数据集"}</span></td>
          <td>{formatDate(document.updated_at)}</td>
        </tr>)}</tbody>
      </table>{!documents.length && <EmptyState icon={<BookOpenCheck size={30} />} title="暂无知识文档" />}</div>

      {selectedDocument && <aside className="job-detail document-detail" data-testid="knowledge-document-detail">
        <div className="detail-title"><div><span className="section-label">KNOWLEDGE DOCUMENT</span><h3>{selectedDocument.document_id}</h3></div>{selectedDocument.current_lifecycle && <StatusPill value={selectedDocument.current_lifecycle} />}</div>
        <label className="revision-select"><span>Revision</span><select value={selectedRevision?.revision ?? ""} onChange={(event) => {
          const revision = revisions.find((item) => item.revision === event.target.value);
          if (revision) {
            onSelectRevision(revision);
            setLifecycleAction(null);
            setReason("");
          }
        }}>{revisions.map((revision) => <option key={revision.revision} value={revision.revision}>{revision.revision} · {revision.lifecycle}</option>)}</select></label>

        {selectedRevision && <>
          <div className="detail-metrics"><Metric label="Chunk" value={selectedRevision.counts.chunks} /><Metric label="活动向量" value={selectedRevision.counts.vectors} /></div>
          <dl className="compact-dl">
            <div><dt>状态</dt><dd><StatusPill value={selectedRevision.lifecycle} /></dd></div>
            <div><dt>审批</dt><dd>{selectedRevision.approval_status}</dd></div>
            <div><dt>图片 / 表格</dt><dd>{selectedRevision.counts.images} / {selectedRevision.counts.tables}</dd></div>
            <div><dt>权限范围</dt><dd>{selectedRevision.access_scope_key}</dd></div>
            <div><dt>来源许可</dt><dd>{selectedRevision.source_license}</dd></div>
            <div><dt>生效时间</dt><dd>{formatDate(selectedRevision.effective_at)}</dd></div>
          </dl>

          {!lifecycleAction && <div className="document-actions">
            {selectedRevision.lifecycle === "published" && <button className="danger-command" type="button" disabled={loading} onClick={() => setLifecycleAction("withdraw")}><ArchiveX size={17} />受控下架</button>}
            {selectedRevision.lifecycle === "withdrawn" && <button className="command-button" type="button" disabled={loading} onClick={() => setLifecycleAction("restore")}><ShieldCheck size={17} />校验并恢复</button>}
          </div>}

          {lifecycleAction && <form className="lifecycle-action-form" onSubmit={submitLifecycle}>
            <div><strong>{lifecycleAction === "withdraw" ? "确认受控下架" : "确认校验并恢复"}</strong><span>{lifecycleAction === "withdraw" ? "检索和图片访问会立即被阻断，MinIO 原件仍保留。" : "系统会重新校验原件、权限、有效期、Hash 和活动索引。"}</span></div>
            <label><span>操作原因</span><textarea autoFocus rows={4} value={reason} onChange={(event) => setReason(event.target.value)} placeholder="至少 8 个字符，原因会进入审计记录" /></label>
            <div className="form-actions"><button className="text-command" type="button" onClick={() => { setLifecycleAction(null); setReason(""); }}>取消</button><button className={lifecycleAction === "withdraw" ? "danger-command" : "command-button"} type="submit" disabled={loading || reason.trim().length < 8}>{lifecycleAction === "withdraw" ? <ArchiveX size={17} /> : <ShieldCheck size={17} />}{lifecycleAction === "withdraw" ? "确认下架" : "确认恢复"}</button></div>
          </form>}
        </>}

        {lifecycleOperation && lifecycleOperation.selector.document_id === selectedDocument.document_id && <div className="lifecycle-operation">
          <div><FileText size={16} /><strong>最近操作</strong><StatusPill value={lifecycleOperation.status} /></div>
          <span>{lifecycleOperation.action} · {lifecycleOperation.operation_id}</span>
          <span>影响 {lifecycleOperation.affected.chunks} Chunk / {lifecycleOperation.affected.images} 图 / {lifecycleOperation.affected.tables} 表</span>
          {lifecycleOperation.warning_codes.length > 0 && <span className="operation-warning">{lifecycleOperation.warning_codes.join(" · ")}</span>}
          {lifecycleOperation.status === "compensation_required" && <button className="command-button" type="button" disabled={loading} onClick={() => void onRetryLifecycle(lifecycleOperation.operation_id)}><RotateCcw size={17} />重试补偿任务</button>}
        </div>}
      </aside>}
    </div>}
  </section>;
}

function Field({ label, value, onChange, wide = false, required = false }: { label: string; value: string; onChange: (value: string) => void; wide?: boolean; required?: boolean }) {
  return <label className={wide ? "field-wide" : ""}><span>{label}</span><input required={required} value={value} onChange={(event) => onChange(event.target.value)} /></label>;
}

function SelectField({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (value: string) => void }) {
  return <label><span>{label}</span><select value={value} onChange={(event) => onChange(event.target.value)}>{options.map((option) => <option key={option || "unset"} value={option}>{option || "未设置"}</option>)}</select></label>;
}
