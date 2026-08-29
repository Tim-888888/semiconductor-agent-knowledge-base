"""Deterministic route policy for validated conversation understanding."""

from __future__ import annotations

import re
from typing import Any

from semikb.contracts.models import (
    ActorScope,
    AgentRoute,
    ConversationUnderstanding,
    IntentTarget,
    IntentTaskAction,
    PrimaryIntent,
    RoutePlan,
    RouteTaskDecision,
    SlotOperationKind,
    TaskExecutionDecision,
    TaskShape,
)

LATEST_TERMS = ("最新", "当前版本", "现行", "实时", "现在", "today", "latest")
EXTERNAL_TERMS = ("外部", "互联网", "网上", "web", "公开资料", "行业最新", "官方公告")
WEB_DISABLED_PATTERN = re.compile(
    r"(?:不要|不需要|无需|不用|禁止|别)(?:使用|调用|查询|搜索|访问|走)?\s*(?:web|外部|互联网|网上)|"
    r"(?:仅|只)(?:查询|检索|使用)?(?:内部|知识库|已入库)"
)
EVIDENCE_FOLLOWUP_PATTERN = re.compile(
    r"(?=.*(?:刚才|上一轮|上面|上述|其中|这些|该条|这条))"
    r"(?=.*(?:证据|来源|引用|依据))",
    re.IGNORECASE,
)
MUTATION_ACTIONS = {IntentTaskAction.EXECUTE}
MUTATION_TARGETS = {IntentTarget.RECIPE}
ROUTE_THRESHOLDS = {
    AgentRoute.CHAT_DIRECT: 0.50,
    AgentRoute.REUSE_EVIDENCE: 0.82,
    AgentRoute.INTERNAL_RAG: 0.58,
    AgentRoute.TOOL_ONLY: 0.68,
    AgentRoute.RAG_AND_TOOL: 0.72,
    AgentRoute.RAG_AND_WEB: 0.72,
}


def is_evidence_followup(request: str) -> bool:
    """Identify a deictic question about evidence already selected in this thread."""

    return EVIDENCE_FOLLOWUP_PATTERN.search(request) is not None


class RoutePolicy:
    """Map semantic tasks to a small audited route whitelist."""

    def decide(
        self,
        understanding: ConversationUnderstanding,
        actor_scope: ActorScope,
        context: dict[str, Any],
        request: str,
    ) -> RoutePlan:
        reasons: list[str] = []
        task_decisions: list[RouteTaskDecision] = []
        candidate_routes: list[AgentRoute] = []
        combined_slots = {**understanding.inherited_slots, **understanding.explicit_slots}

        scope_error = self._scope_error(combined_slots, actor_scope)
        for task in understanding.task_items:
            decision = task.execution_policy
            reason = "task_allowed"
            route: AgentRoute | None = None
            if self._is_forbidden_mutation(task):
                decision = TaskExecutionDecision.REFUSE
                reason = "controlled_write_not_allowed"
            elif task.target_type is IntentTarget.REPORT and task.action is IntentTaskAction.GENERATE:
                decision = TaskExecutionDecision.DEFER
                reason = "report_generation_not_enabled"
            elif scope_error:
                decision = TaskExecutionDecision.REFUSE
                reason = scope_error
            elif decision is TaskExecutionDecision.DEFER:
                reason = "task_deferred_by_policy"
            elif decision is TaskExecutionDecision.REFUSE:
                reason = (
                    "outside_semikb_capability"
                    if task.primary_intent is PrimaryIntent.ACTION_REQUEST
                    and task.target_type is IntentTarget.GENERAL
                    else "model_flagged_unsafe"
                )
            else:
                route = self._task_route(task, understanding, context, request)
                candidate_routes.append(route)
            task_decisions.append(
                RouteTaskDecision(
                    task_id=task.task_id,
                    decision=decision,
                    route=route,
                    reason_code=reason,
                )
            )

        task_decisions, candidate_routes = self._enforce_predefined_combination(
            task_decisions
        )

        executable = [item for item in task_decisions if item.decision is TaskExecutionDecision.EXECUTE]
        if not executable and any(
            item.decision is TaskExecutionDecision.REFUSE for item in task_decisions
        ):
            return RoutePlan(
                route=AgentRoute.REFUSE,
                confidence=understanding.confidence,
                reason_codes=["all_tasks_refused"],
                task_decisions=task_decisions,
                retrieval_skipped_reason="request_refused_before_retrieval",
                invalidated_context_refs=self._invalidated_refs(understanding, context),
            )

        suggested_route = (
            AgentRoute.CHAT_DIRECT
            if understanding.suggested_route is AgentRoute.HISTORY_DIRECT
            else understanding.suggested_route
        )
        route = self._combine_routes(candidate_routes, suggested_route)
        missing = self._required_missing(understanding, combined_slots, route)
        if missing:
            reasons.append("required_slots_missing")
            task_decisions = [
                item.model_copy(
                    update={
                        "decision": TaskExecutionDecision.CLARIFY
                        if item.decision is TaskExecutionDecision.EXECUTE
                        else item.decision,
                        "reason_code": "required_slots_missing"
                        if item.decision is TaskExecutionDecision.EXECUTE
                        else item.reason_code,
                    }
                )
                for item in task_decisions
            ]
            return RoutePlan(
                route=AgentRoute.CLARIFY,
                confidence=understanding.confidence,
                reason_codes=reasons,
                task_decisions=task_decisions,
                missing_slots=missing,
                retrieval_skipped_reason="clarification_required_before_retrieval",
                invalidated_context_refs=self._invalidated_refs(understanding, context),
            )

        threshold = ROUTE_THRESHOLDS.get(route, 0.65)
        if understanding.confidence < threshold:
            reasons.append(f"confidence_below_{route.value}_threshold")
            return RoutePlan(
                route=AgentRoute.CLARIFY,
                confidence=understanding.confidence,
                reason_codes=reasons,
                task_decisions=[
                    item.model_copy(
                        update={
                            "decision": TaskExecutionDecision.CLARIFY
                            if item.decision is TaskExecutionDecision.EXECUTE
                            else item.decision,
                            "reason_code": "semantic_confidence_too_low"
                            if item.decision is TaskExecutionDecision.EXECUTE
                            else item.reason_code,
                        }
                    )
                    for item in task_decisions
                ],
                retrieval_skipped_reason="low_confidence_requires_clarification",
                invalidated_context_refs=self._invalidated_refs(understanding, context),
            )

        if route is not understanding.suggested_route:
            reasons.append("server_policy_overrode_suggested_route")
        reasons.append("route_whitelist_match")
        skipped = None
        if route is AgentRoute.CHAT_DIRECT:
            skipped = "answer_available_without_external_retrieval"
        elif route is AgentRoute.TOOL_ONLY:
            skipped = "manufacturing_data_only_no_vector_retrieval"
        elif route is AgentRoute.REUSE_EVIDENCE:
            skipped = "validated_previous_evidence_reused"
        return RoutePlan(
            route=route,
            confidence=understanding.confidence,
            reason_codes=reasons,
            task_decisions=task_decisions,
            retrieval_skipped_reason=skipped,
            invalidated_context_refs=self._invalidated_refs(understanding, context),
        )

    @staticmethod
    def _task_route(
        task,
        understanding: ConversationUnderstanding,
        context: dict[str, Any],
        request: str,
    ) -> AgentRoute:
        lowered = request.lower()
        if is_evidence_followup(request) and RoutePolicy._has_valid_evidence(context):
            return AgentRoute.REUSE_EVIDENCE
        if (
            task.target_type is IntentTarget.PREVIOUS_USER_MESSAGE
            and task.action is IntentTaskAction.RECALL
        ):
            return AgentRoute.CHAT_DIRECT
        if task.target_type is IntentTarget.PREVIOUS_ANSWER and task.action in {
            IntentTaskAction.RECALL,
            IntentTaskAction.SIMPLIFY,
            IntentTaskAction.SUMMARIZE,
            IntentTaskAction.TRANSLATE,
        }:
            return AgentRoute.CHAT_DIRECT
        if understanding.primary_intent is PrimaryIntent.INVESTIGATION and task.primary_intent in {
            PrimaryIntent.INVESTIGATION,
            PrimaryIntent.DATA_QUERY,
            PrimaryIntent.KNOWLEDGE_QUERY,
        }:
            return AgentRoute.RAG_AND_TOOL
        if task.primary_intent in {PrimaryIntent.CONVERSATION, PrimaryIntent.CONTENT_TASK}:
            if (
                understanding.interaction_mode.value == "feedback"
                and task.action not in {
                    IntentTaskAction.SIMPLIFY,
                    IntentTaskAction.SUMMARIZE,
                    IntentTaskAction.TRANSLATE,
                }
            ):
                return AgentRoute.CHAT_DIRECT
            if task.target_type in {
                IntentTarget.PREVIOUS_USER_MESSAGE,
                IntentTarget.PREVIOUS_ANSWER,
            }:
                return AgentRoute.CHAT_DIRECT
            return AgentRoute.CHAT_DIRECT
        if task.primary_intent is PrimaryIntent.DATA_QUERY:
            return AgentRoute.TOOL_ONLY
        if task.primary_intent is PrimaryIntent.INVESTIGATION:
            return AgentRoute.RAG_AND_TOOL
        if any(term in lowered for term in EXTERNAL_TERMS) and not WEB_DISABLED_PATTERN.search(
            lowered
        ):
            return AgentRoute.RAG_AND_WEB
        if (
            any(term in lowered for term in ("继续", "基于刚才", "根据刚才", "再分析"))
            and not any(term in lowered for term in LATEST_TERMS)
            and RoutePolicy._has_valid_evidence(context)
        ):
            return AgentRoute.REUSE_EVIDENCE
        return AgentRoute.INTERNAL_RAG

    @staticmethod
    def _combine_routes(
        routes: list[AgentRoute],
        suggested: AgentRoute,
    ) -> AgentRoute:
        unique = set(routes)
        if not unique:
            return suggested if suggested in AgentRoute else AgentRoute.CLARIFY
        if AgentRoute.RAG_AND_TOOL in unique or {
            AgentRoute.INTERNAL_RAG,
            AgentRoute.TOOL_ONLY,
        }.issubset(unique):
            return AgentRoute.RAG_AND_TOOL
        if AgentRoute.RAG_AND_WEB in unique:
            return AgentRoute.RAG_AND_WEB
        non_direct = [
            item
            for item in routes
            if item is not AgentRoute.CHAT_DIRECT
        ]
        return non_direct[0] if non_direct else routes[0]

    @staticmethod
    def _enforce_predefined_combination(
        decisions: list[RouteTaskDecision],
    ) -> tuple[list[RouteTaskDecision], list[AgentRoute]]:
        """Defer incompatible tasks instead of inventing a free-form execution graph."""

        executable = [
            item
            for item in decisions
            if item.decision is TaskExecutionDecision.EXECUTE and item.route is not None
        ]
        non_direct = [
            item for item in executable if item.route is not AgentRoute.CHAT_DIRECT
        ]
        routes = {item.route for item in non_direct}
        allowed = (
            len(routes) <= 1
            or routes.issubset(
                {
                    AgentRoute.INTERNAL_RAG,
                    AgentRoute.TOOL_ONLY,
                    AgentRoute.RAG_AND_TOOL,
                }
            )
            or routes.issubset({AgentRoute.INTERNAL_RAG, AgentRoute.RAG_AND_WEB})
        )
        if allowed:
            return decisions, [item.route for item in executable if item.route is not None]

        primary = non_direct[0].route if non_direct else AgentRoute.CHAT_DIRECT
        adjusted: list[RouteTaskDecision] = []
        for item in decisions:
            compatible = (
                item.route in {None, AgentRoute.CHAT_DIRECT, primary}
                or primary in {
                    AgentRoute.INTERNAL_RAG,
                    AgentRoute.TOOL_ONLY,
                    AgentRoute.RAG_AND_TOOL,
                }
                and item.route
                in {
                    AgentRoute.INTERNAL_RAG,
                    AgentRoute.TOOL_ONLY,
                    AgentRoute.RAG_AND_TOOL,
                }
                or primary in {AgentRoute.INTERNAL_RAG, AgentRoute.RAG_AND_WEB}
                and item.route in {AgentRoute.INTERNAL_RAG, AgentRoute.RAG_AND_WEB}
            )
            if item.decision is TaskExecutionDecision.EXECUTE and not compatible:
                adjusted.append(
                    item.model_copy(
                        update={
                            "decision": TaskExecutionDecision.DEFER,
                            "route": None,
                            "reason_code": "unsupported_route_combination_deferred",
                        }
                    )
                )
            else:
                adjusted.append(item)
        return adjusted, [
            item.route
            for item in adjusted
            if item.decision is TaskExecutionDecision.EXECUTE and item.route is not None
        ]

    @staticmethod
    def _required_missing(
        understanding: ConversationUnderstanding,
        slots: dict[str, str],
        route: AgentRoute,
    ) -> list[str]:
        missing: list[str] = []
        executable = [
            item
            for item in understanding.task_items
            if item.execution_policy is TaskExecutionDecision.EXECUTE
        ]
        for task in executable:
            if task.primary_intent is PrimaryIntent.DATA_QUERY:
                RoutePolicy._append_data_task_missing(missing, task, slots)
            elif task.primary_intent is PrimaryIntent.INVESTIGATION:
                if task.task_shape is TaskShape.UNSPECIFIED:
                    RoutePolicy._append_legacy_manufacturing_missing(missing, slots)
                else:
                    if not slots.get("time_range"):
                        missing.append("time_range")
                    if not RoutePolicy._has_affected_object(slots):
                        missing.append("affected_object")
        if route in {AgentRoute.TOOL_ONLY, AgentRoute.RAG_AND_TOOL} and not executable:
            RoutePolicy._append_legacy_manufacturing_missing(missing, slots)
        history_task = any(
            item.target_type in {
                IntentTarget.PREVIOUS_USER_MESSAGE,
                IntentTarget.PREVIOUS_ANSWER,
            }
            and item.action in {
                IntentTaskAction.RECALL,
                IntentTaskAction.SIMPLIFY,
                IntentTaskAction.SUMMARIZE,
                IntentTaskAction.TRANSLATE,
            }
            for item in understanding.task_items
        )
        if history_task and not understanding.context_message_ids:
            missing.append("history_reference")
        return list(dict.fromkeys(missing))[:3]

    @staticmethod
    def _append_data_task_missing(
        missing: list[str],
        task,
        slots: dict[str, str],
    ) -> None:
        if task.task_shape is TaskShape.UNSPECIFIED:
            RoutePolicy._append_legacy_manufacturing_missing(missing, slots)
            return
        if not slots.get("time_range"):
            missing.append("time_range")
        if task.task_shape in {
            TaskShape.AGGREGATE_RANKING,
            TaskShape.EVENT_LIST,
            TaskShape.TREND_ANALYSIS,
        }:
            return
        if task.task_shape is TaskShape.ENTITY_LOOKUP and not RoutePolicy._has_affected_object(
            slots
        ):
            missing.append("affected_object")

    @staticmethod
    def _append_legacy_manufacturing_missing(
        missing: list[str],
        slots: dict[str, str],
    ) -> None:
        if not slots.get("product"):
            missing.append("product")
        if not slots.get("time_range"):
            missing.append("time_range")
        if not (slots.get("tool_id") or slots.get("chamber")):
            missing.append("tool_or_chamber")

    @staticmethod
    def _has_affected_object(slots: dict[str, str]) -> bool:
        return any(
            slots.get(name)
            for name in ("product", "tool_id", "chamber", "lot_id", "case_id")
        )

    @staticmethod
    def _scope_error(slots: dict[str, str], actor_scope: ActorScope) -> str | None:
        if slots.get("product") and actor_scope.products and slots["product"] not in actor_scope.products:
            return "product_out_of_scope"
        if slots.get("tool_id") and actor_scope.tool_ids and slots["tool_id"] not in actor_scope.tool_ids:
            return "tool_out_of_scope"
        return None

    @staticmethod
    def _is_forbidden_mutation(task) -> bool:
        return task.action in MUTATION_ACTIONS and task.target_type in MUTATION_TARGETS

    @staticmethod
    def _has_valid_evidence(context: dict[str, Any]) -> bool:
        active = context.get("active_context", {})
        refs = active.get("evidence_refs", []) if isinstance(active, dict) else []
        return any(isinstance(item, dict) and item.get("valid") is True for item in refs)

    @staticmethod
    def _invalidated_refs(
        understanding: ConversationUnderstanding,
        context: dict[str, Any],
    ) -> list[str]:
        changed = {
            item.slot_name
            for item in understanding.slot_operations
            if item.operation in {SlotOperationKind.CORRECT, SlotOperationKind.CLEAR}
        }
        if not changed:
            return []
        active = context.get("active_context", {})
        slots = active.get("slots", {}) if isinstance(active, dict) else {}
        invalidated = []
        if isinstance(slots, dict):
            for name, item in slots.items():
                dependencies = item.get("depends_on", []) if isinstance(item, dict) else []
                if changed.intersection(str(value) for value in dependencies):
                    invalidated.append(f"slot:{name}")
        refs = active.get("evidence_refs", []) if isinstance(active, dict) else []
        invalidated.extend(
            f"evidence:{item['evidence_id']}"
            for item in refs
            if isinstance(item, dict) and item.get("valid") is True and item.get("evidence_id")
        )
        return invalidated
