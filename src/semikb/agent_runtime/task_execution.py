"""Controlled multi-task completion and route-specific result validation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from semikb.contracts.models import (
    AgentRoute,
    EvidenceLedgerEntry,
    IntentTarget,
    IntentTaskAction,
    IntentTaskItem,
    RoutePlan,
    TaskExecutionDecision,
    TaskExecutionResult,
    TaskExecutionStatus,
)

ROUTE_GENERATION_CONTRACTS: dict[AgentRoute, str] = {
    AgentRoute.REUSE_EVIDENCE: (
        "Use the revalidated internal evidence ledger and preserve its citations."
    ),
    AgentRoute.INTERNAL_RAG: (
        "Use internal controlled document evidence. Every factual claim must cite internal evidence."
    ),
    AgentRoute.TOOL_ONLY: (
        "Use only labeled read-only manufacturing tool facts. Do not imply that a write action occurred."
    ),
    AgentRoute.RAG_AND_TOOL: (
        "Use both internal controlled documents and labeled read-only manufacturing tool facts. "
        "Keep document requirements separate from observed or simulated tool facts."
    ),
    AgentRoute.RAG_AND_WEB: (
        "Use internal controlled documents as the authority. External evidence is supplementary and "
        "must never override an applicable internal requirement."
    ),
}


TARGET_LABELS: dict[IntentTarget, str] = {
    IntentTarget.PREVIOUS_USER_MESSAGE: "回顾上一条用户问题",
    IntentTarget.PREVIOUS_ANSWER: "处理上一条回答",
    IntentTarget.SOP: "SOP",
    IntentTarget.RECIPE: "Recipe",
    IntentTarget.FDC: "FDC 数据",
    IntentTarget.SPC: "SPC 数据",
    IntentTarget.WAFER_MAP: "晶圆图",
    IntentTarget.LOT: "Lot 数据",
    IntentTarget.CASE: "异常 Case",
    IntentTarget.ALARM: "报警信息",
    IntentTarget.REPORT: "报告",
    IntentTarget.GENERAL: "当前请求",
}

ACTION_LABELS: dict[IntentTaskAction, str] = {
    IntentTaskAction.RECALL: "回顾",
    IntentTaskAction.SUMMARIZE: "总结",
    IntentTaskAction.SIMPLIFY: "简化",
    IntentTaskAction.TRANSLATE: "翻译",
    IntentTaskAction.LOOKUP: "查询",
    IntentTaskAction.COMPARE: "对照",
    IntentTaskAction.DIAGNOSE: "分析",
    IntentTaskAction.EXPLAIN: "解释",
    IntentTaskAction.EXECUTE: "执行",
    IntentTaskAction.GENERATE: "生成",
}


def route_generation_contract(route: AgentRoute | str | None) -> str:
    """Return the evidence contract injected into the answer prompt."""

    try:
        normalized = AgentRoute(str(route)) if route is not None else None
    except ValueError:
        normalized = None
    return ROUTE_GENERATION_CONTRACTS.get(
        normalized,
        "Use only the supplied evidence ledger and preserve exact evidence identifiers.",
    )


class TaskExecutionCoordinator:
    """Turn route planning decisions into one terminal result per task."""

    def finalize(
        self,
        *,
        route_plan: RoutePlan,
        task_items: Iterable[IntentTaskItem],
        actual_route: AgentRoute,
        answer_text: str,
        task_outputs: dict[str, str] | None = None,
        evidence_ledger: Iterable[EvidenceLedgerEntry] = (),
        cited_evidence_ids: Iterable[str] = (),
        external_evidence: Iterable[dict[str, Any]] = (),
        missing_fields: Iterable[str] = (),
        authorization_errors: Iterable[str] = (),
    ) -> list[TaskExecutionResult]:
        tasks = {item.task_id: item for item in task_items}
        outputs = task_outputs or {}
        ledger = list(evidence_ledger)
        citations = {str(item) for item in cited_evidence_ids}
        missing = [str(item) for item in missing_fields]
        authorization = [str(item) for item in authorization_errors]
        results: list[TaskExecutionResult] = []
        results_by_id: dict[str, TaskExecutionResult] = {}

        for decision in route_plan.task_decisions:
            task = tasks.get(decision.task_id)
            label = self._task_label(task)
            dependency_failure = next(
                (
                    results_by_id[dependency]
                    for dependency in (task.depends_on if task else [])
                    if dependency in results_by_id
                    and results_by_id[dependency].status is not TaskExecutionStatus.COMPLETED
                ),
                None,
            )
            if dependency_failure is not None and decision.decision is TaskExecutionDecision.EXECUTE:
                result = TaskExecutionResult(
                    task_id=decision.task_id,
                    status=TaskExecutionStatus.DEFERRED,
                    route=decision.route,
                    reason_code="dependency_not_completed",
                    message=self._status_message(label, "已延后：依赖任务尚未完成。"),
                )
            elif authorization and decision.decision is TaskExecutionDecision.EXECUTE:
                result = TaskExecutionResult(
                    task_id=decision.task_id,
                    status=TaskExecutionStatus.REFUSED,
                    route=None,
                    reason_code=authorization[0],
                    message=self._status_message(label, "未执行：请求对象超出当前授权范围。"),
                )
            elif decision.decision is TaskExecutionDecision.CLARIFY:
                detail = "、".join(missing) or "关键条件"
                result = TaskExecutionResult(
                    task_id=decision.task_id,
                    status=TaskExecutionStatus.CLARIFY,
                    route=AgentRoute.CLARIFY,
                    reason_code=decision.reason_code,
                    message=self._status_message(label, f"需要先补充：{detail}。"),
                )
            elif decision.decision is TaskExecutionDecision.REFUSE:
                result = TaskExecutionResult(
                    task_id=decision.task_id,
                    status=TaskExecutionStatus.REFUSED,
                    route=AgentRoute.REFUSE,
                    reason_code=decision.reason_code,
                    message=self._refusal_message(label, decision.reason_code),
                )
            elif decision.decision is TaskExecutionDecision.DEFER:
                result = TaskExecutionResult(
                    task_id=decision.task_id,
                    status=TaskExecutionStatus.DEFERRED,
                    route=decision.route,
                    reason_code=decision.reason_code,
                    message=self._status_message(
                        label,
                        "已延后，本轮没有静默执行该任务。",
                    ),
                )
            else:
                route = self._actual_task_route(decision.route, actual_route)
                result = self._validate_executed_task(
                    task_id=decision.task_id,
                    label=label,
                    route=route,
                    answer_text=outputs.get(decision.task_id, answer_text),
                    ledger=ledger,
                    citations=citations,
                    external_evidence=list(external_evidence),
                )
            results.append(result)
            results_by_id[result.task_id] = result

        return results

    @staticmethod
    def _actual_task_route(
        planned: AgentRoute | None,
        actual: AgentRoute,
    ) -> AgentRoute:
        if planned is AgentRoute.HISTORY_DIRECT:
            return AgentRoute.CHAT_DIRECT
        if planned is AgentRoute.REUSE_EVIDENCE and actual is AgentRoute.INTERNAL_RAG:
            return actual
        return planned or actual

    def _validate_executed_task(
        self,
        *,
        task_id: str,
        label: str,
        route: AgentRoute,
        answer_text: str,
        ledger: list[EvidenceLedgerEntry],
        citations: set[str],
        external_evidence: list[dict[str, Any]],
    ) -> TaskExecutionResult:
        internal_ids = [
            item.evidence_id
            for item in ledger
            if item.source_type == "internal_controlled" and item.evidence_id in citations
        ]
        tool_ids = [
            item.evidence_id
            for item in ledger
            if item.source_type == "simulated_live_data" and item.evidence_id in citations
        ]
        external_ids = [
            item.evidence_id
            for item in ledger
            if item.source_type == "external" and item.evidence_id in citations
        ]
        if route is AgentRoute.CHAT_DIRECT:
            internal_ids = []
            tool_ids = []
            external_ids = []
        elif route in {AgentRoute.REUSE_EVIDENCE, AgentRoute.INTERNAL_RAG}:
            tool_ids = []
            external_ids = []
        elif route is AgentRoute.TOOL_ONLY:
            internal_ids = []
            external_ids = []
        elif route is AgentRoute.RAG_AND_TOOL:
            external_ids = []
        elif route is AgentRoute.RAG_AND_WEB:
            tool_ids = []
        has_answer = bool(answer_text.strip())
        warnings: list[str] = []

        valid = has_answer
        if route in {AgentRoute.REUSE_EVIDENCE, AgentRoute.INTERNAL_RAG}:
            valid = valid and bool(internal_ids)
        elif route is AgentRoute.TOOL_ONLY:
            valid = valid and bool(tool_ids)
        elif route is AgentRoute.RAG_AND_TOOL:
            valid = valid and bool(internal_ids) and bool(tool_ids)
        elif route is AgentRoute.RAG_AND_WEB:
            valid = valid and bool(internal_ids)
            if not external_ids:
                warnings.append(
                    "external_evidence_unavailable"
                    if any(item.get("source_type") == "external_unavailable" for item in external_evidence)
                    else "external_evidence_not_cited"
                    if any(item.get("source_type") == "external" for item in external_evidence)
                    else "external_evidence_empty"
                )

        if not valid:
            return TaskExecutionResult(
                task_id=task_id,
                status=TaskExecutionStatus.FAILED,
                route=route,
                reason_code="route_contract_not_satisfied",
                message=self._status_message(
                    label,
                    f"未完成：结果没有满足 {route.value} 的证据要求。",
                ),
                evidence_ids=internal_ids,
                tool_fact_ids=tool_ids,
                external_evidence_ids=external_ids,
            )

        reason = "completed_with_external_degradation" if warnings else "route_contract_satisfied"
        message = self._status_message(
            label,
            f"已完成，并通过 {route.value} 路由校验。",
        )
        if warnings:
            message = self._status_message(
                label,
                "已完成内部证据校验；外部资料不可用或未返回结果。",
            )
        return TaskExecutionResult(
            task_id=task_id,
            status=TaskExecutionStatus.COMPLETED,
            route=route,
            reason_code=reason,
            message=message,
            evidence_ids=internal_ids,
            tool_fact_ids=tool_ids,
            external_evidence_ids=external_ids,
            validation_warnings=warnings,
        )

    @staticmethod
    def _task_label(task: IntentTaskItem | None) -> str:
        if task is None:
            return "该任务"
        action = ACTION_LABELS.get(task.action, task.action.value)
        target = TARGET_LABELS.get(task.target_type, task.target_type.value)
        if task.action is IntentTaskAction.EXECUTE and task.target_type is IntentTarget.RECIPE:
            return "Recipe 修改"
        if target.startswith(action):
            return target
        separator = " " if target[:1].isascii() and target[:1].isalpha() else ""
        return f"{action}{separator}{target}"

    @staticmethod
    def _refusal_message(label: str, reason_code: str) -> str:
        if reason_code == "controlled_write_not_allowed":
            return TaskExecutionCoordinator._status_message(
                label,
                "未执行：智库只提供受控只读分析，不能修改 Recipe。",
            )
        if reason_code in {"product_out_of_scope", "tool_out_of_scope"}:
            return TaskExecutionCoordinator._status_message(
                label,
                "未执行：请求对象超出当前授权范围。",
            )
        return TaskExecutionCoordinator._status_message(
            label,
            "未执行：该操作超出当前智库的受控能力边界。",
        )

    @staticmethod
    def _status_message(label: str, detail: str) -> str:
        separator = " " if label[-1:].isascii() and label[-1:].isalnum() else ""
        return f"{label}{separator}{detail}"
