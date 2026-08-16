import { type FormEvent, useEffect, useMemo, useState } from "react";
import {
  BarChart3,
  ClipboardCheck,
  FlaskConical,
  RefreshCw,
  RotateCcw,
  SearchCheck,
  TriangleAlert
} from "lucide-react";
import type { EvaluationDataset, EvaluationReleaseFreeze, EvaluationRun } from "../types";
import { EmptyState, formatDate, formatMetric, Metric, StatusPill } from "./Common";

type Props = {
  datasets: EvaluationDataset[];
  freezes: EvaluationReleaseFreeze[];
  runs: EvaluationRun[];
  selectedRun?: EvaluationRun;
  loading: boolean;
  onSelect: (runId: string) => void;
  onRun: (input: { dataset_version: string; retrieval_profile: EvaluationRun["retrieval_profile"]; baseline_run_id?: string }) => Promise<void>;
  onRetry: (runId: string) => Promise<void>;
  onRefresh: () => Promise<void>;
  onOpenTrace: (runId: string, caseId: string) => Promise<void>;
};

const profiles: { value: EvaluationRun["retrieval_profile"]; label: string }[] = [
  { value: "dense", label: "Dense" },
  { value: "hybrid", label: "Dense + Sparse" },
  { value: "reranked", label: "Hybrid + Rerank" },
  { value: "full", label: "Full / 条件 HyDE" }
];

export function EvaluationPanel({ datasets, freezes, runs, selectedRun, loading, onSelect, onRun, onRetry, onRefresh, onOpenTrace }: Props) {
  const [datasetVersion, setDatasetVersion] = useState("t5-live-v1");
  const [profile, setProfile] = useState<EvaluationRun["retrieval_profile"]>("full");
  const [baselineRunId, setBaselineRunId] = useState("");
  const [caseFilter, setCaseFilter] = useState<"all" | "failed">("all");
  const [selectedCaseId, setSelectedCaseId] = useState("");

  useEffect(() => {
    if (datasets.length && !datasets.some((item) => item.dataset_version === datasetVersion)) {
      setDatasetVersion(datasets[0].dataset_version);
    }
  }, [datasetVersion, datasets]);

  const baselineOptions = runs.filter((run) => run.status === "completed" && run.dataset_version === datasetVersion);
  const visibleCases = useMemo(() => (selectedRun?.case_results ?? []).filter((item) => caseFilter === "all" || !item.passed), [caseFilter, selectedRun]);
  const selectedCase = visibleCases.find((item) => item.case_id === selectedCaseId) ?? visibleCases.find((item) => !item.passed) ?? visibleCases[0];

  async function submit(event: FormEvent) {
    event.preventDefault();
    await onRun({
      dataset_version: datasetVersion,
      retrieval_profile: profile,
      ...(baselineRunId ? { baseline_run_id: baselineRunId } : {})
    });
  }

  return <section className="operations-layout evaluation-layout">
    <header className="operations-heading">
      <div><span className="section-label">OFFLINE EVALUATION</span><h2>黄金集与基线</h2></div>
      <button className="icon-button" type="button" title="刷新评估" onClick={() => void onRefresh()} disabled={loading}><RefreshCw size={17} className={loading ? "spin" : ""} /></button>
    </header>

    <form className="evaluation-launcher" onSubmit={submit} data-testid="evaluation-form">
      <label><span>数据集</span><select value={datasetVersion} onChange={(event) => { setDatasetVersion(event.target.value); setBaselineRunId(""); }}>{datasets.map((dataset) => <option key={dataset.dataset_version} value={dataset.dataset_version}>{dataset.dataset_version} · {dataset.purpose} · {dataset.case_count} Case</option>)}{!datasets.length && <option value="t5-live-v1">t5-live-v1</option>}</select></label>
      <label><span>检索档位</span><select value={profile} onChange={(event) => setProfile(event.target.value as EvaluationRun["retrieval_profile"])}>{profiles.map((item) => <option value={item.value} key={item.value}>{item.label}</option>)}</select></label>
      <label><span>Baseline</span><select value={baselineRunId} onChange={(event) => setBaselineRunId(event.target.value)}><option value="">无</option>{baselineOptions.map((run) => <option value={run.evaluation_run_id} key={run.evaluation_run_id}>{run.retrieval_profile} · {run.evaluation_run_id.slice(-8)}</option>)}</select></label>
      <button className="command-button" type="submit" disabled={loading || !datasetVersion}><ClipboardCheck size={17} />创建运行</button>
    </form>

    <section className="run-snapshot" data-testid="evaluation-release-freezes">
      <strong>Release Freeze</strong>
      {freezes.length === 0 && <span>尚未冻结，holdout 不可运行</span>}
      {freezes.slice(0, 3).map((freeze) => <span key={freeze.freeze_id}>{freeze.release_version} · {freeze.status} · {freeze.freeze_hash.slice(0, 12)}</span>)}
    </section>

    <div className="evaluation-split">
      <aside className="run-index">
        <div className="index-heading"><span className="section-label">EVALUATION RUNS</span><strong>{runs.length}</strong></div>
        <div className="run-index-list">{runs.map((run) => <button type="button" key={run.evaluation_run_id} className={selectedRun?.evaluation_run_id === run.evaluation_run_id ? "active" : ""} onClick={() => onSelect(run.evaluation_run_id)}>
          <div><StatusPill value={run.status} /><span>{formatDate(run.created_at)}</span></div>
          <strong>{run.dataset_version} · {run.retrieval_profile}</strong>
          <small>{run.evaluation_run_id.slice(-12)}</small>
        </button>)}</div>
      </aside>

      <div className="evaluation-detail">
        {!selectedRun ? <EmptyState icon={<BarChart3 size={34} />} title="尚无评估运行" /> : <>
          <div className="run-heading">
            <div><span className="section-label">RUN DETAIL</span><h3>{selectedRun.dataset_version} · {selectedRun.retrieval_profile}</h3><p className="mono-id">{selectedRun.evaluation_run_id}</p></div>
            <div className="heading-actions"><StatusPill value={selectedRun.status} />{selectedRun.status === "failed" && <button className="icon-button" type="button" title="重试评估" onClick={() => void onRetry(selectedRun.evaluation_run_id)} disabled={loading}><RotateCcw size={17} /></button>}</div>
          </div>

          <section className="run-snapshot"><span>purpose={selectedRun.dataset_purpose}</span><span>leakage={selectedRun.dataset_leakage_status}</span><span>sealed={selectedRun.dataset_sealed_at ? formatDate(selectedRun.dataset_sealed_at) : "no"}</span><span>opened={selectedRun.dataset_opened_at ? formatDate(selectedRun.dataset_opened_at) : "no"}</span>{selectedRun.release_freeze_hash && <span>freeze={selectedRun.release_freeze_hash.slice(0, 12)}</span>}</section>

          <div className="metric-grid evaluation-metrics">
            <Metric label="Recall@5" value={formatMetric(selectedRun.aggregate_metrics.recall_at_5)} tone="good" />
            <Metric label="MRR" value={formatMetric(selectedRun.aggregate_metrics.mrr)} tone="good" />
            <Metric label="nDCG@5" value={formatMetric(selectedRun.aggregate_metrics.ndcg_at_5)} tone="good" />
            <Metric label="负例准确率" value={formatMetric(selectedRun.aggregate_metrics.no_evidence_accuracy)} />
            <Metric label="图片 Recall" value={formatMetric(selectedRun.aggregate_metrics.image_recall_at_5)} />
            <Metric label="通过率" value={formatMetric(selectedRun.aggregate_metrics.pass_rate)} />
            <Metric label="平均延迟" value={`${formatMetric(selectedRun.aggregate_metrics.average_latency_ms, 2)} ms`} />
            <Metric label="P95 延迟" value={`${formatMetric(selectedRun.aggregate_metrics.p95_latency_ms, 2)} ms`} />
          </div>

          {Object.keys(selectedRun.baseline_comparison).length > 0 && <section className="baseline-section">
            <h3><BarChart3 size={17} />基线差异</h3>
            <div className="data-table-wrap"><table><thead><tr><th>指标</th><th>Baseline</th><th>当前</th><th>Delta</th><th>结论</th></tr></thead><tbody>{Object.entries(selectedRun.baseline_comparison).map(([metric, comparison]) => <tr key={metric}><td><strong>{metric}</strong></td><td>{formatMetric(comparison.baseline)}</td><td>{formatMetric(comparison.current)}</td><td className={comparison.outcome === "improved" ? "delta-good" : comparison.outcome === "regressed" ? "delta-bad" : ""}>{comparison.delta > 0 ? "+" : ""}{formatMetric(comparison.delta, 4)}</td><td><StatusPill value={comparison.outcome} /></td></tr>)}</tbody></table></div>
          </section>}

          <section className="case-section">
            <div className="section-toolbar"><h3><FlaskConical size={17} />Case 结果</h3><div className="segmented">{(["all", "failed"] as const).map((value) => <button type="button" className={caseFilter === value ? "active" : ""} key={value} onClick={() => setCaseFilter(value)}>{value === "all" ? "全部" : "仅失败"}</button>)}</div></div>
            <div className="case-split">
              <div className="case-list">{visibleCases.map((item) => <button type="button" key={item.case_id} className={selectedCase?.case_id === item.case_id ? "active" : ""} onClick={() => setSelectedCaseId(item.case_id)}><StatusPill value={item.passed ? "completed" : "failed"} /><strong>{item.case_id}</strong><span>{item.question}</span></button>)}{!visibleCases.length && <p className="inline-empty">{caseFilter === "failed" ? "没有失败 Case" : ["queued", "running"].includes(selectedRun.status) ? "评估运行中，Case 结果稍后写入" : "暂无 Case 结果"}</p>}</div>
              {selectedCase && <div className="case-detail" data-testid="evaluation-case-detail">
                <div className="detail-title"><div><span className="section-label">CASE DETAIL</span><h4>{selectedCase.case_id}</h4></div><StatusPill value={selectedCase.passed ? "completed" : "failed"} /></div>
                <p>{selectedCase.question}</p>
                <dl className="compact-dl">
                  <div><dt>期望</dt><dd>{selectedCase.expected_chunk_ids.join(" / ") || "无证据"}</dd></div>
                  <div><dt>实际</dt><dd>{selectedCase.actual_chunk_ids.join(" / ") || "无证据"}</dd></div>
                  <div><dt>缺失</dt><dd>{selectedCase.missing_expected_chunk_ids.join(" / ") || "-"}</dd></div>
                  <div><dt>截断</dt><dd>{selectedCase.cutoff_reason}</dd></div>
                  <div><dt>耗时</dt><dd>{formatMetric(selectedCase.latency_ms, 2)} ms</dd></div>
                  <div><dt>标签</dt><dd>{[...selectedCase.tags, ...selectedCase.failure_tags].join(" / ") || "-"}</dd></div>
                </dl>
                {selectedCase.warnings.length > 0 && <div className="warning-band"><TriangleAlert size={16} /><div>{selectedCase.warnings.map((warning) => <span key={warning}>{warning}</span>)}</div></div>}
                <button className="text-command" type="button" onClick={() => void onOpenTrace(selectedRun.evaluation_run_id, selectedCase.case_id)}><SearchCheck size={16} />打开 RetrievalTrace</button>
              </div>}
            </div>
          </section>

          <section className="run-snapshot"><span>dataset {selectedRun.dataset_hash.slice(0, 12)}</span>{Object.entries(selectedRun.component_versions).map(([key, value]) => <span key={key}>{key}={value}</span>)}</section>
          {selectedRun.safe_error_summary && <div className="warning-band"><TriangleAlert size={17} /><div><strong>运行失败</strong><span>{selectedRun.safe_error_summary}</span></div></div>}
        </>}
      </div>
    </div>
  </section>;
}
