import { useMemo, useState } from "react";
import {
  Clock3,
  FileSearch,
  Filter,
  Image as ImageIcon,
  LockKeyhole,
  Route,
  ShieldCheck,
  TriangleAlert
} from "lucide-react";
import type { RetrievalTrace } from "../types";
import { EmptyState, formatDate, formatMetric, Metric, StatusPill } from "./Common";

type Props = {
  traces: RetrievalTrace[];
  trace?: RetrievalTrace;
  onSelect: (traceId: string) => void;
  onOpenImage: (imageId: string) => void;
};

export function TracePanel({ traces, trace, onSelect, onOpenImage }: Props) {
  const [routeFilter, setRouteFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState<"all" | "selected" | "excluded">("all");
  const routes = useMemo(() => [...new Set(trace?.candidates.flatMap((item) => item.routes) ?? [])], [trace]);
  const candidates = useMemo(() => (trace?.candidates ?? []).filter((candidate) => {
    const routeMatches = routeFilter === "all" || candidate.routes.includes(routeFilter);
    const statusMatches = statusFilter === "all" || (statusFilter === "selected" ? candidate.selected : !candidate.selected);
    return routeMatches && statusMatches;
  }), [routeFilter, statusFilter, trace]);

  if (!trace) return <EmptyState icon={<FileSearch size={34} />} title="尚无检索 Trace" detail="完成一次受控查询后生成" />;

  return <section className="trace-shell">
    <aside className="trace-index" aria-label="Trace 列表">
      <div className="index-heading"><span className="section-label">RECENT TRACES</span><strong>{traces.length}</strong></div>
      <div className="trace-index-list">
        {traces.map((item) => <button
          type="button"
          key={item.trace_id}
          className={item.trace_id === trace.trace_id ? "active" : ""}
          onClick={() => onSelect(item.trace_id)}
        >
          <span>{formatDate(item.created_at)}</span>
          <strong>{item.original_query}</strong>
          <small>{item.routes.join(" · ")}</small>
        </button>)}
      </div>
    </aside>

    <div className="trace-layout">
      <header className="trace-summary">
        <div><span className="section-label">RETRIEVAL TRACE</span><h2>{trace.original_query}</h2><p className="mono-id">{trace.trace_id}</p></div>
        <div className="route-list">{trace.routes.map((routeName) => <span key={routeName}>{routeName}</span>)}</div>
      </header>

      <div className="trace-kpis">
        <Metric label="候选" value={trace.candidates.length} />
        <Metric label="最终证据" value={trace.final_evidence_ids.length} tone="good" />
        <Metric label="截断" value={trace.cutoff_reason} />
        <Metric label="总耗时" value={`${formatMetric(trace.timings_ms.total, 2)} ms`} />
      </div>

      <section className="trace-context-band">
        <div><Filter size={16} /><span>约束</span><strong>{formatMap(trace.metadata_filters)}</strong></div>
        <div><Route size={16} /><span>HyDE</span><strong>{trace.hyde_query || "未启用"}</strong></div>
        <div><ShieldCheck size={16} /><span>Scope</span><strong>{trace.access_scope_keys.join(" / ")}</strong></div>
      </section>

      <div className="trace-toolbar">
        <label><span>召回通道</span><select value={routeFilter} onChange={(event) => setRouteFilter(event.target.value)}><option value="all">全部</option>{routes.map((routeName) => <option key={routeName} value={routeName}>{routeName}</option>)}</select></label>
        <div className="segmented" aria-label="候选状态筛选">
          {(["all", "selected", "excluded"] as const).map((value) => <button type="button" key={value} className={statusFilter === value ? "active" : ""} onClick={() => setStatusFilter(value)}>{value === "all" ? "全部" : value === "selected" ? "入选" : "排除"}</button>)}
        </div>
      </div>

      <div className="data-table-wrap trace-table"><table>
        <thead><tr><th>证据</th><th>通道 / 排名</th><th>Dense</th><th>Sparse</th><th>HyDE</th><th>RRF</th><th>Rerank</th><th>决策</th></tr></thead>
        <tbody>{candidates.map((candidate) => <tr key={candidate.chunk_id}>
          <td><strong>{candidate.document_id} {candidate.revision}</strong><span>{candidate.page_or_section}</span><small>{candidate.chunk_id}</small></td>
          <td>{candidate.routes.map((routeName) => <span key={routeName}>{routeName} #{candidate.route_ranks[routeName] ?? "-"}</span>)}</td>
          <td>{formatMetric(candidate.dense_score)}</td>
          <td>{formatMetric(candidate.sparse_score)}</td>
          <td>{formatMetric(candidate.hyde_score)}</td>
          <td>{formatMetric(candidate.rrf_score, 4)}</td>
          <td><strong>{formatMetric(candidate.rerank_score)}</strong></td>
          <td><StatusPill value={candidate.selected ? "selected" : "excluded"} /><span>{candidate.context_selection_reason ?? candidate.exclusion_reason ?? "-"}</span>{candidate.protected_evidence && <small className="protected"><LockKeyhole size={12} />受保护证据</small>}</td>
        </tr>)}</tbody>
      </table></div>

      <div className="trace-bottom-grid">
        <section className="detail-band">
          <h3><Clock3 size={17} />阶段耗时</h3>
          <dl>{Object.entries(trace.timings_ms).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{formatMetric(value, 2)} ms</dd></div>)}</dl>
        </section>
        <section className="detail-band">
          <h3><ShieldCheck size={17} />组件版本</h3>
          <dl>{Object.entries(trace.component_versions).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{value}</dd></div>)}</dl>
        </section>
      </div>

      {trace.warnings.length > 0 && <div className="warning-band"><TriangleAlert size={17} /><div>{trace.warnings.map((warning) => <span key={warning}>{warning}</span>)}</div></div>}
      {trace.image_asset_ids.length > 0 && <div className="trace-images">{trace.image_asset_ids.map((imageId) => <button type="button" onClick={() => onOpenImage(imageId)} key={imageId}><ImageIcon size={16} />{imageId}</button>)}</div>}
    </div>
  </section>;
}

function formatMap(value: Record<string, unknown>): string {
  const entries = Object.entries(value).filter(([, item]) => item !== null && item !== "" && (!Array.isArray(item) || item.length));
  return entries.length ? entries.map(([key, item]) => `${key}=${Array.isArray(item) ? item.join(",") : String(item)}`).join(" · ") : "无额外约束";
}
