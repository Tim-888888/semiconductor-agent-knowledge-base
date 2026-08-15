"""Bounded LangGraph workflow for evidence-driven semiconductor investigations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any, Literal, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from semikb.agent_runtime.direct_reply import (
    DirectReplyGenerator,
    DirectReplyKind,
    DirectReplyRequest,
)
from semikb.agent_runtime.llm_gateway import OpenAICompatibleLLMGateway
from semikb.agent_runtime.memory import MemoryService
from semikb.agent_runtime.routing import RoutePolicy
from semikb.agent_runtime.streaming_answer import StreamingAnswerAssembler, format_answer
from semikb.agent_runtime.task_execution import (
    TaskExecutionCoordinator,
    route_generation_contract,
)
from semikb.agent_runtime.tools import ManufacturingToolbox
from semikb.agent_runtime.understanding import CHAMBER_PATTERN, ConversationUnderstandingService
from semikb.agent_runtime.web_search import AliyunWebSearchGateway
from semikb.config import Settings
from semikb.contracts.models import (
    ActorScope,
    AgentAnswer,
    AgentRoute,
    AnswerClaim,
    AuditEvent,
    Chunk,
    ConversationUnderstanding,
    EvidenceLedgerEntry,
    IntentTaskItem,
    RetrievalConstraints,
    RoutePlan,
    TaskExecutionDecision,
)
from semikb.storage.conversations import ConversationRepository


class CaseState(TypedDict, total=False):
    request: str
    thread_id: str
    run_id: str
    user_scope: dict[str, Any]
    conversation_context: dict[str, Any]
    understanding: dict[str, Any]
    route_plan: dict[str, Any]
    interaction_mode: str
    route_decision: str
    route_confidence: float
    task_items: list[dict[str, Any]]
    task_results: list[dict[str, Any]]
    task_outputs: dict[str, str]
    combined_direct_text: str
    slot_operations: list[dict[str, Any]]
    inherited_slots: dict[str, str]
    context_message_ids: list[str]
    standalone_query: str
    retrieval_skipped_reason: str | None
    invalidated_context_refs: list[str]
    cancel_scope: str | None
    intent: str
    risk_level: str
    constraints: dict[str, Any]
    authorization_errors: list[str]
    missing_required_fields: list[str]
    clarification_questions: list[str]
    clarification_round: int
    clarification_response: str | None
    retrieval_query: str
    retrieval_routes: list[str]
    candidate_ids: list[str]
    reranked_evidence: list[dict[str, Any]]
    image_evidence: list[str]
    external_evidence: list[dict[str, Any]]
    live_data_refs: list[dict[str, Any]]
    evidence_ledger: list[dict[str, Any]]
    approved_preferences: list[str]
    answer: dict[str, Any]
    answer_text: str
    citations: list[dict[str, Any]]
    trace_id: str | None
    status: str
    model_metadata: dict[str, Any]
    verification_warnings: list[str]


class ConversationGraph:
    """Compile the controlled graph with replaceable persistence and service adapters."""

    def __init__(
        self,
        *,
        settings: Settings,
        repository: ConversationRepository,
        retrieval: Any,
        checkpointer: Any,
        memory_service: MemoryService,
        llm: OpenAICompatibleLLMGateway | None = None,
        web_search: AliyunWebSearchGateway | None = None,
        toolbox: ManufacturingToolbox | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.retrieval = retrieval
        self.memory_service = memory_service
        self.llm = llm or OpenAICompatibleLLMGateway(settings)
        self.web_search = web_search or AliyunWebSearchGateway(settings)
        self.toolbox = toolbox or ManufacturingToolbox()
        self.understanding = ConversationUnderstandingService(settings, self.llm)
        self.route_policy = RoutePolicy()
        self.direct_reply = DirectReplyGenerator(settings, self.llm)
        self.task_execution = TaskExecutionCoordinator()

        workflow = StateGraph(CaseState)
        workflow.add_node("ingest_request", self._ingest_request)
        workflow.add_node("authorize_scope", self._authorize_scope)
        workflow.add_node("classify_and_extract", self._classify_and_extract)
        workflow.add_node("validate_required_fields", self._validate_required_fields)
        workflow.add_node("prepare_clarification_reply", self._prepare_clarification_reply)
        workflow.add_node("clarify_missing_fields", self._clarify_missing_fields)
        workflow.add_node("direct_answer", self._direct_answer)
        workflow.add_node("refuse_request", self._refuse_request)
        workflow.add_node("execute_combination_prelude", self._execute_combination_prelude)
        workflow.add_node("retrieve_evidence", self._retrieve_evidence)
        workflow.add_node("build_evidence_ledger", self._build_evidence_ledger)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("verify_answer", self._verify_answer)
        workflow.add_node("insufficient_information", self._insufficient_information)
        workflow.add_node("finalize_task_results", self._finalize_task_results)
        workflow.add_node("audit", self._audit)

        workflow.add_edge(START, "ingest_request")
        workflow.add_edge("ingest_request", "authorize_scope")
        workflow.add_edge("authorize_scope", "classify_and_extract")
        workflow.add_edge("classify_and_extract", "validate_required_fields")
        workflow.add_conditional_edges(
            "validate_required_fields",
            self._route_after_validation,
            {
                "clarify": "prepare_clarification_reply",
                "direct": "direct_answer",
                "refuse": "refuse_request",
                "retrieve": "execute_combination_prelude",
                "insufficient": "insufficient_information",
            },
        )
        workflow.add_edge("prepare_clarification_reply", "clarify_missing_fields")
        workflow.add_edge("clarify_missing_fields", "classify_and_extract")
        workflow.add_edge("direct_answer", "finalize_task_results")
        workflow.add_edge("refuse_request", "finalize_task_results")
        workflow.add_edge("execute_combination_prelude", "retrieve_evidence")
        workflow.add_edge("retrieve_evidence", "build_evidence_ledger")
        workflow.add_edge("build_evidence_ledger", "generate_answer")
        workflow.add_edge("generate_answer", "verify_answer")
        workflow.add_edge("verify_answer", "finalize_task_results")
        workflow.add_edge("insufficient_information", "finalize_task_results")
        workflow.add_edge("finalize_task_results", "audit")
        workflow.add_edge("audit", END)
        self.compiled = workflow.compile(checkpointer=checkpointer, store=memory_service.store)

    async def _ingest_request(self, state: CaseState) -> CaseState:
        self._emit_stream(
            "stage",
            stage="analyzing_request",
            message="正在分析问题与会话上下文",
        )
        preferences = state.get("approved_preferences")
        if preferences is None:
            preferences = await asyncio.to_thread(
                self.memory_service.approved_preferences,
                state["user_scope"]["user_id"],
            )
        return {
            "status": "running",
            "constraints": {},
            "understanding": {},
            "route_plan": {},
            "interaction_mode": "task",
            "route_decision": AgentRoute.CLARIFY.value,
            "route_confidence": 0.0,
            "task_items": [],
            "task_results": [],
            "task_outputs": {},
            "combined_direct_text": "",
            "slot_operations": [],
            "inherited_slots": {},
            "context_message_ids": [],
            "standalone_query": state["request"],
            "retrieval_skipped_reason": None,
            "invalidated_context_refs": [],
            "cancel_scope": None,
            "authorization_errors": [],
            "missing_required_fields": [],
            "clarification_questions": [],
            "retrieval_query": state["request"],
            "retrieval_routes": [],
            "candidate_ids": [],
            "reranked_evidence": [],
            "image_evidence": [],
            "external_evidence": [],
            "live_data_refs": [],
            "evidence_ledger": [],
            "approved_preferences": preferences,
            "answer": {},
            "answer_text": "",
            "citations": [],
            "trace_id": None,
            "model_metadata": {},
            "verification_warnings": [],
        }

    @staticmethod
    def _authorize_scope(state: CaseState) -> CaseState:
        try:
            ActorScope.model_validate(state["user_scope"])
        except ValueError:
            return {"authorization_errors": ["invalid_actor_scope"]}
        return {"authorization_errors": []}

    async def _classify_and_extract(self, state: CaseState) -> CaseState:
        self._emit_stream(
            "stage",
            stage="routing_request",
            message="正在识别任务、上下文引用与受控路由",
        )
        actor_scope = ActorScope.model_validate(state["user_scope"])
        result = await self.understanding.understand(
            state["request"],
            state.get("conversation_context", {}),
            clarification_pending=bool(state.get("clarification_response")),
        )
        understanding = result.understanding
        plan = self.route_policy.decide(
            understanding,
            actor_scope,
            state.get("conversation_context", {}),
            state["request"],
        )
        constraints = {
            **understanding.inherited_slots,
            **understanding.explicit_slots,
        }
        if understanding.primary_intent.value == "investigation":
            intent = "anomaly_investigation"
        elif any(
            item.target_type.value == "recipe" and item.action.value == "execute"
            for item in understanding.task_items
        ):
            intent = "recipe_impact"
        elif understanding.primary_intent.value == "content_task":
            intent = "report"
        else:
            intent = "knowledge_qa"
        risk = (
            "high"
            if plan.route is AgentRoute.REFUSE
            else "medium"
            if plan.route in {AgentRoute.TOOL_ONLY, AgentRoute.RAG_AND_TOOL}
            else "low"
        )
        for decision in plan.task_decisions:
            if decision.decision is TaskExecutionDecision.EXECUTE:
                progress = "queued"
                message = "任务已进入受控执行队列"
            elif decision.decision is TaskExecutionDecision.CLARIFY:
                progress = "clarify"
                message = "任务需要补充关键信息"
            elif decision.decision is TaskExecutionDecision.REFUSE:
                progress = "refused"
                message = "任务已由服务端策略拒绝"
            else:
                progress = "deferred"
                message = "任务已明确延后，本轮不会静默执行"
            self._emit_stream(
                "task_status",
                task_id=decision.task_id,
                status=progress,
                route=decision.route.value if decision.route else None,
                message=message,
            )
        return {
            "intent": intent,
            "risk_level": risk,
            "constraints": constraints,
            "retrieval_query": understanding.standalone_query or state["request"],
            "standalone_query": understanding.standalone_query,
            "understanding": understanding.model_dump(mode="json"),
            "route_plan": plan.model_dump(mode="json"),
            "interaction_mode": understanding.interaction_mode.value,
            "route_decision": plan.route.value,
            "route_confidence": plan.confidence,
            "task_items": [item.model_dump(mode="json") for item in understanding.task_items],
            "slot_operations": [
                item.model_dump(mode="json") for item in understanding.slot_operations
            ],
            "inherited_slots": understanding.inherited_slots,
            "context_message_ids": understanding.context_message_ids,
            "retrieval_skipped_reason": plan.retrieval_skipped_reason,
            "invalidated_context_refs": plan.invalidated_context_refs,
            "cancel_scope": understanding.cancel_scope.value if understanding.cancel_scope else None,
            "model_metadata": {**state.get("model_metadata", {}), **result.metadata},
        }

    @staticmethod
    def _constraint_is_grounded(
        field: str,
        value: Any,
        request: str,
        conversation_context: dict[str, Any] | None = None,
    ) -> bool:
        """Never turn an LLM-inferred identifier into a retrieval filter."""

        normalized_value = re.sub(r"\s+", "", str(value)).lower()
        normalized_request = re.sub(r"\s+", "", request).lower()
        if not normalized_value:
            return False
        active_context = (conversation_context or {}).get("active_context", {})
        context_slots = (
            active_context.get("slots", {}) if isinstance(active_context, dict) else {}
        )
        context_slot = context_slots.get(field, {}) if isinstance(context_slots, dict) else {}
        if (
            isinstance(context_slot, dict)
            and context_slot.get("valid") is True
            and re.sub(r"\s+", "", str(context_slot.get("value", ""))).lower()
            == normalized_value
            and context_slot.get("source_message_id")
        ):
            return True
        if field == "time_range":
            return any(char.isdigit() for char in normalized_value) and any(
                token in normalized_request for token in ("小时", "天", "周", "最近", "过去", "-")
            )
        if normalized_value in normalized_request:
            return True
        return False

    @staticmethod
    def _safe_context_payload(context: dict[str, Any]) -> dict[str, Any]:
        """Keep context bounded and omit operational fields that extraction does not need."""

        if not isinstance(context, dict):
            return {}
        active = context.get("active_context", {})
        valid_slots: dict[str, dict[str, str]] = {}
        if isinstance(active, dict) and isinstance(active.get("slots"), dict):
            for name, slot in active["slots"].items():
                if isinstance(slot, dict) and slot.get("valid") is True:
                    valid_slots[str(name)] = {
                        "value": str(slot.get("value", "")),
                        "source_message_id": str(slot.get("source_message_id", "")),
                    }
        recent = []
        for item in context.get("recent_messages", [])[-24:]:
            if isinstance(item, dict):
                recent.append(
                    {
                        "role": str(item.get("role", "")),
                        "content": str(item.get("content", ""))[:600],
                        "message_id": str(item.get("message_id", "")),
                    }
                )
        return {
            "summary": str(context.get("summary", ""))[:2000],
            "summary_upto_message_id": context.get("summary_upto_message_id"),
            "recent_messages": recent,
            "valid_slots": valid_slots,
        }

    @staticmethod
    def _deterministic_extract(content: str) -> dict[str, Any]:
        normal = content.upper()
        constraints: dict[str, Any] = {}
        patterns = {
            "product": r"\bP-[A-Z0-9][A-Z0-9-]*\b",
            "tool_id": r"\b(?:ETCH|CVD|PVD|CMP|PHOTO|LITHO|IMP|DIFF)-\d+[A-Z]?\b",
            "recipe_version": r"\bV\d+(?:\.\d+)+\b",
            "lot_id": r"\bLOT[-_ ]?[A-Z0-9-]+\b",
        }
        for field, pattern in patterns.items():
            match = re.search(pattern, normal)
            if match:
                constraints[field] = match.group(0).replace(" ", "-")
        chamber = CHAMBER_PATTERN.search(normal)
        if chamber:
            constraints["chamber"] = chamber.group(1)
        time_range = re.search(
            r"(?:最近|过去)?\s*\d+\s*(?:小时|天|周)|\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\s*(?:至|到|~)\s*\d{4}[-/]\d{1,2}[-/]\d{1,2})?",
            content,
            re.IGNORECASE,
        )
        if time_range:
            constraints["time_range"] = time_range.group(0).strip()
        investigation_terms = ("良率", "yield", "下降", "异常调查", "根因", "alarm", "报警")
        recipe_terms = ("recipe影响", "recipe 影响", "配方影响")
        lowered = content.lower()
        if any(term in lowered for term in recipe_terms):
            intent = "recipe_impact"
        elif any(term in lowered for term in investigation_terms):
            intent = "anomaly_investigation"
        else:
            intent = "knowledge_qa"
        risk = "high" if any(term in lowered for term in ("修改", "下发", "停机", "执行")) else "low"
        return {"intent": intent, "risk_level": risk, "constraints": constraints}

    @staticmethod
    def _validate_required_fields(state: CaseState) -> CaseState:
        constraints = state.get("constraints", {})
        scope = ActorScope.model_validate(state["user_scope"])
        authorization_errors = list(state.get("authorization_errors", []))
        if constraints.get("product") and scope.products and constraints["product"] not in scope.products:
            authorization_errors.append("product_out_of_scope")
        if constraints.get("tool_id") and scope.tool_ids and constraints["tool_id"] not in scope.tool_ids:
            authorization_errors.append("tool_out_of_scope")

        route_plan = RoutePlan.model_validate(state.get("route_plan", {}))
        missing = list(route_plan.missing_slots)
        if route_plan.route is AgentRoute.CLARIFY and not missing:
            missing.append("request_goal")
        prompts = {
            "product": "受影响的 Product 是什么？",
            "time_range": "异常从何时开始，或需要查询哪个时间范围？",
            "tool_or_chamber": "涉及哪个 Tool 或 Chamber？若未知，请提供已知 FDC 报警或 Lot 范围。",
            "history_reference": "当前线程没有可引用的上一轮内容，请明确需要处理的文本或问题。",
            "request_goal": "请明确希望查询知识、查看制造数据，还是处理上一轮内容。",
        }
        return {
            "authorization_errors": sorted(set(authorization_errors)),
            "missing_required_fields": missing[:3],
            "clarification_questions": [prompts[field] for field in missing[:3]],
        }

    def _route_after_validation(
        self,
        state: CaseState,
    ) -> Literal["clarify", "direct", "refuse", "retrieve", "insufficient"]:
        if state.get("authorization_errors"):
            return "refuse"
        route = AgentRoute(str(state.get("route_decision", AgentRoute.CLARIFY)))
        if route is AgentRoute.REFUSE:
            return "refuse"
        if route in {AgentRoute.HISTORY_DIRECT, AgentRoute.CHAT_DIRECT}:
            return "direct"
        if not state.get("missing_required_fields") and route is not AgentRoute.CLARIFY:
            return "retrieve"
        if state.get("clarification_round", 0) >= self.settings.agent_max_clarification_rounds:
            return "insufficient"
        return "clarify"

    async def _prepare_clarification_reply(self, state: CaseState) -> CaseState:
        self._emit_stream(
            "stage",
            stage="awaiting_clarification",
            message="关键信息不足，正在生成澄清问题",
        )
        result = await self.direct_reply.generate(
            DirectReplyRequest(
                kind=DirectReplyKind.CLARIFICATION,
                user_request=state["request"],
                conversation_context=state.get("conversation_context", {}),
                missing_slots=tuple(state.get("missing_required_fields", [])),
                clarification_questions=tuple(state.get("clarification_questions", [])),
            ),
            self._emit_direct_delta,
        )
        metadata = dict(state.get("model_metadata", {}))
        metadata["direct_reply_audit"] = result.audit.model_dump(mode="json")
        task_results = self._final_task_results(state, answer_text=result.text)
        return {
            "answer_text": result.text,
            "model_metadata": metadata,
            "task_results": task_results,
        }

    @staticmethod
    def _clarify_missing_fields(state: CaseState) -> CaseState:
        payload = {
            "kind": "clarification",
            "round": state.get("clarification_round", 0) + 1,
            "missing_fields": state.get("missing_required_fields", []),
            "questions": state.get("clarification_questions", []),
            "response": state.get("answer_text", ""),
        }
        response = interrupt(payload)
        merged = f"{state['request']}\n用户补充：{response}"
        return {
            "request": merged,
            "clarification_response": str(response),
            "clarification_round": state.get("clarification_round", 0) + 1,
        }

    async def _direct_answer(self, state: CaseState) -> CaseState:
        understanding = ConversationUnderstanding.model_validate(state["understanding"])
        context = state.get("conversation_context", {})
        history_task = next(
            (
                item
                for item in understanding.task_items
                if item.target_type.value in {"previous_user_message", "previous_answer"}
            ),
            None,
        )
        action = history_task.action.value if history_task else None
        if history_task and action == "recall":
            kind = DirectReplyKind.HISTORY_RECALL
        elif history_task and action in {"simplify", "summarize", "translate"}:
            kind = DirectReplyKind.HISTORY_TRANSFORM
        elif understanding.cancel_scope or understanding.slot_operations:
            kind = DirectReplyKind.CONTROL_ACK
        elif understanding.interaction_mode.value == "feedback":
            kind = DirectReplyKind.FEEDBACK
        else:
            kind = DirectReplyKind.GENERAL_CHAT

        control_summary = None
        if understanding.cancel_scope:
            control_summary = "已记录本次取消范围；会话历史仍会保留。"
        elif understanding.slot_operations:
            changes = [
                f"{item.slot_name}={item.value}"
                for item in understanding.slot_operations
                if item.value
            ]
            control_summary = (
                f"已更新当前会话条件：{'，'.join(changes)}。依赖旧条件的上下文将失效。"
            )

        self._emit_stream(
            "stage",
            stage="generating_answer",
            message="正在生成受控的自然回复",
        )
        self._emit_running_tasks(state, routes={AgentRoute.CHAT_DIRECT})
        result = await self.direct_reply.generate(
            DirectReplyRequest(
                kind=kind,
                user_request=state["request"],
                conversation_context=context,
                context_message_ids=tuple(understanding.context_message_ids),
                action=action,
                control_summary=control_summary,
            ),
            self._emit_direct_delta,
        )
        metadata = dict(state.get("model_metadata", {}))
        metadata["direct_reply_audit"] = result.audit.model_dump(mode="json")
        return {
            "answer_text": result.text,
            "answer": {},
            "citations": [],
            "status": "completed",
            "model_metadata": metadata,
        }

    @staticmethod
    def _context_message(
        context: dict[str, Any],
        message_ids: list[str],
    ) -> dict[str, Any] | None:
        allowed = set(message_ids)
        for item in reversed(context.get("recent_messages", [])):
            if isinstance(item, dict) and item.get("message_id") in allowed:
                return item
        return None

    async def _refuse_request(self, state: CaseState) -> CaseState:
        task_reasons = [
            str(item.get("reason_code"))
            for item in state.get("route_plan", {}).get("task_decisions", [])
            if isinstance(item, dict) and item.get("reason_code")
        ]
        reason_codes = tuple(
            dict.fromkeys([*state.get("authorization_errors", []), *task_reasons])
        ) or ("outside_semikb_capability",)
        alternatives = (
            ("authorized_scope_help", "read_only_semiconductor_help")
            if state.get("authorization_errors")
            else ("read_only_semiconductor_help", "capability_guidance")
        )
        self._emit_stream(
            "stage",
            stage="generating_answer",
            message="正在说明能力边界与可用替代方案",
        )
        result = await self.direct_reply.generate(
            DirectReplyRequest(
                kind=DirectReplyKind.REFUSAL,
                user_request=state["request"],
                conversation_context=state.get("conversation_context", {}),
                reason_codes=reason_codes,
                alternative_codes=alternatives,
            ),
            self._emit_direct_delta,
        )
        metadata = dict(state.get("model_metadata", {}))
        metadata["direct_reply_audit"] = result.audit.model_dump(mode="json")
        return {
            "answer_text": result.text,
            "answer": {},
            "citations": [],
            "status": "refused",
            "model_metadata": metadata,
        }

    async def _execute_combination_prelude(self, state: CaseState) -> CaseState:
        """Execute the direct-history part of a predefined direct + business route."""

        understanding = ConversationUnderstanding.model_validate(state["understanding"])
        plan = RoutePlan.model_validate(state["route_plan"])
        tasks = {item.task_id: item for item in understanding.task_items}
        outputs: dict[str, str] = {}
        audits: list[dict[str, Any]] = []
        for decision in plan.task_decisions:
            if (
                decision.decision is not TaskExecutionDecision.EXECUTE
                or decision.route is not AgentRoute.CHAT_DIRECT
            ):
                continue
            task = tasks.get(decision.task_id)
            if task is None or task.target_type.value not in {
                "previous_user_message",
                "previous_answer",
            }:
                continue
            self._emit_stream(
                "task_status",
                task_id=decision.task_id,
                status="running",
                route=AgentRoute.CHAT_DIRECT.value,
                message="正在处理服务端选中的历史消息",
            )
            if outputs:
                self._emit_stream("answer_delta", delta="\n\n")
            kind = (
                DirectReplyKind.HISTORY_RECALL
                if task.action.value == "recall"
                else DirectReplyKind.HISTORY_TRANSFORM
            )
            generated = await self.direct_reply.generate(
                DirectReplyRequest(
                    kind=kind,
                    user_request=state["request"],
                    conversation_context=state.get("conversation_context", {}),
                    context_message_ids=tuple(understanding.context_message_ids),
                    action=task.action.value,
                ),
                self._emit_direct_delta,
            )
            outputs[decision.task_id] = generated.text
            audits.append(generated.audit.model_dump(mode="json"))

        metadata = dict(state.get("model_metadata", {}))
        if audits:
            metadata["combined_direct_reply_audits"] = audits
        return {
            "task_outputs": outputs,
            "combined_direct_text": "\n\n".join(outputs.values()),
            "model_metadata": metadata,
        }

    @staticmethod
    def _emit_direct_delta(
        delta: str,
        provider: str | None,
        model: str | None,
    ) -> None:
        ConversationGraph._emit_stream(
            "answer_delta",
            delta=delta,
            provider=provider,
            model=model,
        )

    async def _retrieve_evidence(self, state: CaseState) -> CaseState:
        actor_scope = ActorScope.model_validate(state["user_scope"])
        constraints = self._retrieval_constraints(state.get("constraints", {}))
        route = AgentRoute(str(state["route_decision"]))
        self._emit_running_tasks(
            state,
            routes={
                AgentRoute.REUSE_EVIDENCE,
                AgentRoute.INTERNAL_RAG,
                AgentRoute.TOOL_ONLY,
                AgentRoute.RAG_AND_TOOL,
                AgentRoute.RAG_AND_WEB,
            },
        )
        evidence: list[Chunk] = []
        trace = None

        if route is AgentRoute.REUSE_EVIDENCE:
            active = state.get("conversation_context", {}).get("active_context", {})
            trace_id = active.get("trace_id") if isinstance(active, dict) else None
            reused = (
                self.retrieval.reuse_trace_evidence(
                    str(trace_id),
                    actor_scope,
                    constraints=constraints,
                )
                if trace_id
                else None
            )
            if reused is not None:
                evidence, trace = reused
            else:
                route = AgentRoute.INTERNAL_RAG

        retrieval_task = None
        if route in {
            AgentRoute.INTERNAL_RAG,
            AgentRoute.RAG_AND_TOOL,
            AgentRoute.RAG_AND_WEB,
        }:
            self._emit_stream(
                "stage",
                stage="retrieving_evidence",
                message="正在执行权限过滤与混合召回",
            )
            retrieval_task = asyncio.to_thread(
                self.retrieval.search,
                state["retrieval_query"],
                actor_scope,
                top_k=5,
                thread_id=state["thread_id"],
                constraints=constraints,
            )
        web_task = None
        if route is AgentRoute.RAG_AND_WEB:
            self._emit_stream(
                "stage",
                stage="searching_external",
                message="内部证据不足，正在查询受控外部来源",
            )
            web_task = asyncio.create_task(self.web_search.search(state["retrieval_query"]))

        if retrieval_task is not None:
            evidence, trace = await retrieval_task
        external: list[dict[str, Any]] = []
        if web_task is not None:
            try:
                external = await web_task
            except Exception as exc:
                external = [
                    {
                        "source_type": "external_unavailable",
                        "content": type(exc).__name__,
                        "url": "",
                    }
                ]
        if trace is not None:
            trace.external_evidence = external
            self.retrieval.save_trace(trace)
            self._emit_stream(
                "stage",
                stage="reranking_evidence",
                message="召回完成，正在整理重排后的证据",
            )
        live_data = []
        if route in {AgentRoute.TOOL_ONLY, AgentRoute.RAG_AND_TOOL}:
            live_data = self.toolbox.query_for_case(
                state["request"],
                state.get("constraints", {}),
            )
        return {
            "route_decision": route.value,
            "retrieval_skipped_reason": (
                "validated_previous_evidence_reused"
                if route is AgentRoute.REUSE_EVIDENCE
                else "manufacturing_data_only_no_vector_retrieval"
                if route is AgentRoute.TOOL_ONLY
                else None
            ),
            "retrieval_routes": trace.routes if trace else [],
            "candidate_ids": [candidate.chunk_id for candidate in trace.candidates] if trace else [],
            "reranked_evidence": [chunk.model_dump(mode="json") for chunk in evidence],
            "image_evidence": trace.image_asset_ids if trace else [],
            "external_evidence": external,
            "live_data_refs": live_data,
            "trace_id": trace.trace_id if trace else None,
        }

    @staticmethod
    def _retrieval_constraints(values: dict[str, Any]) -> RetrievalConstraints:
        allowed = set(RetrievalConstraints.model_fields)
        return RetrievalConstraints.model_validate(
            {key: value for key, value in values.items() if key in allowed}
        )

    def _build_evidence_ledger(self, state: CaseState) -> CaseState:
        trace = (
            self.retrieval.get_trace(
                str(state["trace_id"]),
                ActorScope.model_validate(state["user_scope"]),
            )
            if state.get("trace_id")
            else None
        )
        trace_candidates = {
            candidate.chunk_id: candidate
            for candidate in (trace.candidates if trace else [])
        }
        ledger: list[EvidenceLedgerEntry] = []
        for raw_chunk in state.get("reranked_evidence", []):
            chunk = Chunk.model_validate(raw_chunk)
            candidate = trace_candidates.get(chunk.chunk_id)
            ledger.append(
                EvidenceLedgerEntry(
                    evidence_id=f"chunk:{chunk.chunk_id}",
                    source_type="internal_controlled",
                    content=chunk.chunk_text[:1600],
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    revision=chunk.revision,
                    approval_status=str(chunk.approval_status),
                    effective_at=chunk.effective_at,
                    source_uri=str(chunk.metadata.get("source_uri", "")),
                    page_or_section=chunk.page_or_section,
                    tool_id=chunk.tool_id,
                    chamber=chunk.chamber,
                    recipe_version=chunk.recipe_version,
                    retrieval_routes=candidate.routes if candidate else [],
                    retrieval_score=max(
                        candidate.dense_score,
                        candidate.sparse_score,
                        candidate.hyde_score,
                    )
                    if candidate
                    else None,
                    rerank_score=candidate.rerank_score if candidate else None,
                    context_selection_reason=candidate.context_selection_reason if candidate else None,
                    image_ids=chunk.image_ids,
                )
            )
        for fact in state.get("live_data_refs", []):
            ledger.append(
                EvidenceLedgerEntry(
                    evidence_id=str(fact["evidence_id"]),
                    source_type="simulated_live_data",
                    content=str(fact["fact"]),
                    tool_id=str(fact.get("parameters", {}).get("tool_id") or "") or None,
                    chamber=fact.get("parameters", {}).get("chamber"),
                )
            )
        for index, item in enumerate(state.get("external_evidence", [])):
            if item.get("source_type") != "external":
                continue
            digest = hashlib.sha256(str(item.get("url", index)).encode()).hexdigest()[:12]
            ledger.append(
                EvidenceLedgerEntry(
                    evidence_id=f"external:{digest}",
                    source_type="external",
                    content=str(item.get("content", ""))[:1200],
                    external_url=str(item.get("url", "")),
                )
            )
        self._emit_stream(
            "evidence",
            trace_id=state.get("trace_id"),
            evidence_ids=[item.evidence_id for item in ledger],
            image_asset_ids=state.get("image_evidence", []),
            internal_count=sum(item.source_type != "external" for item in ledger),
            external_count=sum(item.source_type == "external" for item in ledger),
        )
        return {"evidence_ledger": [item.model_dump(mode="json") for item in ledger]}

    async def _generate_answer(self, state: CaseState) -> CaseState:
        self._emit_stream(
            "stage",
            stage="generating_answer",
            message="正在依据证据生成回答",
        )
        ledger = [EvidenceLedgerEntry.model_validate(item) for item in state.get("evidence_ledger", [])]
        if state.get("combined_direct_text"):
            self._emit_stream("answer_delta", delta="\n\n")
        if not ledger:
            answer = AgentAnswer(
                unknowns=["当前权限、版本和有效期范围内没有足以支持结论的证据。"],
                next_actions=["补充受影响对象、时间范围或可核验的文档与制造数据。"],
                confidence="low",
            )
            return {"answer": answer.model_dump(mode="json"), "model_metadata": state.get("model_metadata", {})}

        answer: AgentAnswer | None = None
        metadata = dict(state.get("model_metadata", {}))
        if not self.settings.demo_mode:
            provider_details: dict[str, str | None] = {"provider": None, "model": None}

            def emit_verified_delta(delta: str) -> None:
                self._emit_stream(
                    "answer_delta",
                    delta=delta,
                    provider=provider_details["provider"],
                    model=provider_details["model"],
                )

            assembler = StreamingAnswerAssembler(ledger, emit_verified_delta)

            def consume_model_delta(delta: str, provider: str, model: str) -> None:
                provider_details["provider"] = provider
                provider_details["model"] = model
                assembler.feed(delta)

            try:
                completion = await self.llm.stream_complete(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are a semiconductor investigation assistant. Use only the supplied evidence ledger. "
                                f"Route contract: {route_generation_contract(state.get('route_decision'))} "
                                "Internal controlled evidence outranks external evidence. Simulated live data must remain "
                                "labeled simulated. Do not declare a root cause; hypotheses must be testable. Stream compact "
                                "JSON objects, one object per line, with no array, prose, or markdown fence. Emit zero or more "
                                "objects in this exact type order: fact, hypothesis, unknown, next_action, then exactly one "
                                "confidence object. Claim objects use {type,text,citation_ids}; unknown and next_action use "
                                "{type,text}; confidence uses {type,value}. Every fact must cite one or more exact evidence_id "
                                "values. Confidence value must be low, medium, or high."
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "question": state["request"],
                                    "constraints": state.get("constraints", {}),
                                    "approved_display_preferences": state.get("approved_preferences", []),
                                    "evidence_ledger": [item.model_dump(mode="json") for item in ledger],
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ],
                    on_content_delta=consume_model_delta,
                    max_output_tokens=self.settings.agent_answer_max_output_tokens,
                )
                answer = assembler.finish()
                metadata.update(
                    {
                        "answer_provider": completion.provider,
                        "answer_model": completion.reported_model,
                        "answer_fallback_used": completion.fallback_used,
                        "answer_streamed": True,
                    }
                )
                if assembler.warnings:
                    metadata["answer_stream_warnings"] = assembler.warnings
            except (ValueError, json.JSONDecodeError):
                raise
        if answer is None:
            answer = self._deterministic_answer(ledger)
            self._emit_stream("answer_delta", delta=format_answer(answer))
        return {"answer": answer.model_dump(mode="json"), "model_metadata": metadata}

    @staticmethod
    def _parse_answer(payload: Any) -> AgentAnswer:
        if not isinstance(payload, dict):
            raise ValueError("Agent answer must be a JSON object.")

        def claims(value: Any) -> list[AnswerClaim]:
            if not isinstance(value, list):
                return []
            normalized: list[AnswerClaim] = []
            for item in value:
                if isinstance(item, str):
                    normalized.append(AnswerClaim(text=item))
                    continue
                if not isinstance(item, dict):
                    continue
                text = item.get("text") or item.get("claim") or item.get("content")
                citations = item.get("citation_ids", item.get("citations", []))
                if isinstance(citations, str):
                    citations = [citations]
                if text and isinstance(citations, list):
                    normalized.append(
                        AnswerClaim(
                            text=str(text),
                            citation_ids=[str(citation) for citation in citations],
                        )
                    )
            return normalized

        def strings(value: Any) -> list[str]:
            if isinstance(value, str):
                return [value]
            if not isinstance(value, list):
                return []
            normalized: list[str] = []
            for item in value:
                if isinstance(item, str):
                    normalized.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or item.get("claim")
                    if text:
                        normalized.append(str(text))
            return normalized

        confidence = str(payload.get("confidence", "low")).lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "low"
        return AgentAnswer(
            facts=claims(payload.get("facts")),
            hypotheses=claims(payload.get("hypotheses")),
            unknowns=strings(payload.get("unknowns")),
            next_actions=strings(payload.get("next_actions")),
            confidence=confidence,
        )

    @staticmethod
    def _deterministic_answer(ledger: list[EvidenceLedgerEntry]) -> AgentAnswer:
        controlled = [item for item in ledger if item.source_type == "internal_controlled"]
        simulated = [item for item in ledger if item.source_type == "simulated_live_data"]
        facts = [
            AnswerClaim(text=item.content[:360], citation_ids=[item.evidence_id])
            for item in [*controlled[:2], *simulated[:2]]
        ]
        hypotheses: list[AnswerClaim] = []
        if controlled and simulated:
            hypotheses.append(
                AnswerClaim(
                    text="当前受控案例与模拟制造数据存在相似信号，但只能作为待验证假设，不能直接认定根因。",
                    citation_ids=[controlled[0].evidence_id, simulated[0].evidence_id],
                )
            )
        return AgentAnswer(
            facts=facts,
            hypotheses=hypotheses,
            unknowns=["真实 Fab 数据尚未接入，本次制造数据来自明确标记的模拟工具。"] if simulated else [],
            next_actions=["按引用的现行 SOP 核对前置条件，并复核对应时间范围内的 FDC 与 Recipe 变更。"],
            confidence="medium" if controlled else "low",
        )

    @staticmethod
    def _verify_answer(state: CaseState) -> CaseState:
        ConversationGraph._emit_stream(
            "stage",
            stage="verifying_answer",
            message="正在校验引用、权限和结论边界",
        )
        ledger = [EvidenceLedgerEntry.model_validate(item) for item in state.get("evidence_ledger", [])]
        valid_ids = {item.evidence_id for item in ledger}
        citation_aliases = {
            alias: item.evidence_id
            for item in ledger
            for alias in (item.chunk_id, item.document_id)
            if alias
        }
        source_by_id = {item.evidence_id: item.source_type for item in ledger}
        has_internal = any(item.source_type == "internal_controlled" for item in ledger)
        answer = AgentAnswer.model_validate(state.get("answer", {}))
        warnings: list[str] = []

        verified_facts: list[AnswerClaim] = []
        for claim in answer.facts:
            citations = [
                citation_aliases.get(citation, citation)
                for citation in claim.citation_ids
                if citation_aliases.get(citation, citation) in valid_ids
            ]
            if not citations:
                warnings.append("fact_removed_without_valid_citation")
                continue
            if has_internal and all(source_by_id[citation] == "external" for citation in citations):
                warnings.append("external_only_fact_removed_when_internal_evidence_exists")
                continue
            verified_facts.append(AnswerClaim(text=claim.text, citation_ids=citations))
        verified_hypotheses = [
            AnswerClaim(
                text=claim.text.replace("根因是", "待验证假设是"),
                citation_ids=[
                    citation_aliases.get(citation, citation)
                    for citation in claim.citation_ids
                    if citation_aliases.get(citation, citation) in valid_ids
                ],
            )
            for claim in answer.hypotheses
        ]
        verified = answer.model_copy(
            update={"facts": verified_facts, "hypotheses": verified_hypotheses}
        )
        citations = [
            {
                "evidence_id": item.evidence_id,
                "source_type": item.source_type,
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "revision": item.revision,
                "page_or_section": item.page_or_section,
                "image_ids": item.image_ids,
                "external_url": item.external_url,
            }
            for item in ledger
            if item.evidence_id
            in {
                citation
                for claim in [*verified.facts, *verified.hypotheses]
                for citation in claim.citation_ids
            }
        ]
        formatted = ConversationGraph._format_answer(verified)
        if state.get("combined_direct_text"):
            formatted = f"{state['combined_direct_text']}\n\n{formatted}"
        return {
            "answer": verified.model_dump(mode="json"),
            "answer_text": formatted,
            "citations": citations,
            "status": "completed",
            "verification_warnings": warnings,
        }

    @staticmethod
    def _format_answer(answer: AgentAnswer) -> str:
        return format_answer(answer)

    @staticmethod
    def _insufficient_information(state: CaseState) -> CaseState:
        if state.get("authorization_errors"):
            response = "请求中的 Product 或 Tool 超出当前用户权限范围，系统未执行检索或工具调用。"
        else:
            missing = "、".join(state.get("missing_required_fields", []))
            response = f"经过两轮追问后信息仍不足，系统停止调查以避免猜测。仍缺少：{missing}。"
        return {
            "answer_text": response,
            "answer": AgentAnswer(
                unknowns=[response],
                confidence="low",
            ).model_dump(mode="json"),
            "citations": [],
            "status": "insufficient_information",
        }

    def _audit(self, state: CaseState) -> CaseState:
        for fact in state.get("live_data_refs", []):
            self.repository.append_audit(
                AuditEvent(
                    event_id=f"audit_{state['run_id']}_tool_{fact['tool']}",
                    event_type="read_only_tool_called",
                    actor_user_id=state["user_scope"]["user_id"],
                    thread_id=state["thread_id"],
                    trace_id=state.get("trace_id"),
                    details={
                        "tool": fact["tool"],
                        "read_only": True,
                        "parameters": fact.get("parameters", {}),
                        "source_type": fact.get("source_type"),
                    },
                )
            )
        event = AuditEvent(
            event_id=f"audit_{state['run_id']}",
            event_type="agent_run_completed",
            actor_user_id=state["user_scope"]["user_id"],
            thread_id=state["thread_id"],
            trace_id=state.get("trace_id"),
            details={
                "status": state.get("status"),
                "intent": state.get("intent"),
                "interaction_mode": state.get("interaction_mode"),
                "route_decision": state.get("route_decision"),
                "route_confidence": state.get("route_confidence"),
                "task_decisions": state.get("route_plan", {}).get("task_decisions", []),
                "task_results": state.get("task_results", []),
                "retrieval_skipped_reason": state.get("retrieval_skipped_reason"),
                "invalidated_context_refs": state.get("invalidated_context_refs", []),
                "cancel_scope": state.get("cancel_scope"),
                "risk_level": state.get("risk_level"),
                "missing_fields": state.get("missing_required_fields", []),
                "evidence_ids": [item["evidence_id"] for item in state.get("evidence_ledger", [])],
                "tool_names": [item.get("tool") for item in state.get("live_data_refs", [])],
                "model_metadata": state.get("model_metadata", {}),
                "verification_warnings": state.get("verification_warnings", []),
            },
        )
        self.repository.append_audit(event)
        return {}

    def _finalize_task_results(self, state: CaseState) -> CaseState:
        task_results = self._final_task_results(
            state,
            answer_text=str(state.get("answer_text", "")),
        )
        for item in task_results:
            self._emit_stream(
                "task_status",
                task_id=str(item["task_id"]),
                status=str(item["status"]),
                route=item.get("route"),
                message=str(item["message"]),
            )
        return {"task_results": task_results}

    def _final_task_results(
        self,
        state: CaseState,
        *,
        answer_text: str,
    ) -> list[dict[str, Any]]:
        plan = RoutePlan.model_validate(state.get("route_plan", {}))
        tasks = [IntentTaskItem.model_validate(item) for item in state.get("task_items", [])]
        ledger = [
            EvidenceLedgerEntry.model_validate(item)
            for item in state.get("evidence_ledger", [])
        ]
        route = AgentRoute(str(state.get("route_decision", plan.route.value)))
        results = self.task_execution.finalize(
            route_plan=plan,
            task_items=tasks,
            actual_route=route,
            answer_text=answer_text,
            task_outputs={
                str(key): str(value)
                for key, value in state.get("task_outputs", {}).items()
            },
            evidence_ledger=ledger,
            cited_evidence_ids=[
                str(item.get("evidence_id"))
                for item in state.get("citations", [])
                if isinstance(item, dict) and item.get("evidence_id")
            ],
            external_evidence=state.get("external_evidence", []),
            missing_fields=state.get("missing_required_fields", []),
            authorization_errors=state.get("authorization_errors", []),
        )
        return [item.model_dump(mode="json") for item in results]

    @staticmethod
    def _emit_running_tasks(
        state: CaseState,
        *,
        routes: set[AgentRoute],
    ) -> None:
        for item in state.get("route_plan", {}).get("task_decisions", []):
            if not isinstance(item, dict):
                continue
            if item.get("decision") != TaskExecutionDecision.EXECUTE.value:
                continue
            route_value = item.get("route")
            if route_value not in {route.value for route in routes}:
                continue
            ConversationGraph._emit_stream(
                "task_status",
                task_id=str(item["task_id"]),
                status="running",
                route=str(route_value),
                message="正在执行该任务的受控下游步骤",
            )

    @staticmethod
    def _emit_stream(kind: str, **payload: Any) -> None:
        """Publish safe UI events when the graph is running in custom stream mode."""

        try:
            writer = get_stream_writer()
        except RuntimeError:
            return
        writer({"kind": kind, **payload})
