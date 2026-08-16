import type { ReactNode } from "react";

export function Metric({ label, value, tone = "default" }: {
  label: string;
  value: string | number;
  tone?: "default" | "good" | "warn" | "bad";
}) {
  return <div className={`metric metric-${tone}`}><span>{label}</span><strong>{value}</strong></div>;
}

export function EmptyState({ icon, title, detail }: {
  icon: ReactNode;
  title: string;
  detail?: string;
}) {
  return <div className="empty-state">{icon}<h2>{title}</h2>{detail && <p>{detail}</p>}</div>;
}

export function StatusPill({ value }: { value: string }) {
  const normalized = value.toLowerCase();
  const tone = ["completed", "published", "selected", "active"].includes(normalized)
    ? "good"
    : ["failed", "refused", "regressed", "excluded", "insufficient_information", "compensation_required"].includes(normalized)
      ? "bad"
      : ["queued", "running", "parsing", "embedding", "waiting_for_clarification", "blocking", "vector_cleanup", "restore_validating", "restore_indexing"].includes(normalized)
        ? "active"
        : "neutral";
  return <span className={`status-pill status-${tone}`}>{labelForStatus(value)}</span>;
}

export function formatDate(value?: string | null): string {
  if (!value) return "-";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  }).format(new Date(value));
}

export function formatMetric(value?: number, digits = 3): string {
  if (value === undefined || Number.isNaN(value)) return "-";
  return value.toFixed(digits).replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
}

export function labelForStage(value: string): string {
  return ({
    queued: "排队",
    validating: "文件校验",
    parsing: "解析",
    quality_check: "质量检查",
    embedding: "向量化",
    staged: "暂存",
    published: "已发布",
    withdrawn: "已下架",
    superseded: "已替代",
    expired: "已过期",
    quarantined: "已隔离",
    requested: "已请求",
    blocking: "阻断中",
    vector_cleanup: "清理向量",
    restore_validating: "恢复校验",
    restore_indexing: "重建索引",
    restored: "已恢复",
    compensation_required: "需要补偿",
    failed: "失败"
  } as Record<string, string>)[value] ?? value;
}

function labelForStatus(value: string): string {
  return ({
    queued: "排队",
    running: "运行中",
    completed: "完成",
    failed: "失败",
    published: "已发布",
    withdrawn: "已下架",
    superseded: "已替代",
    expired: "已过期",
    quarantined: "已隔离",
    staged: "已暂存",
    requested: "已请求",
    blocking: "阻断中",
    vector_cleanup: "清理向量",
    restore_validating: "恢复校验",
    restore_indexing: "重建索引",
    restored: "已恢复",
    compensation_required: "需要补偿",
    selected: "入选",
    excluded: "排除",
    active: "活动",
    waiting_for_clarification: "等待补充",
    clarify: "待澄清",
    refused: "已拒绝",
    deferred: "已延后",
    insufficient_information: "信息不足",
    improved: "提升",
    regressed: "回退",
    unchanged: "持平"
  } as Record<string, string>)[value] ?? value;
}
