"""Bounded LangGraph workflow for evidence-driven semiconductor investigations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from semikb.agent_runtime.llm_gateway import OpenAICompatibleLLMGateway
from semikb.agent_runtime.memory import MemoryService
from semikb.agent_runtime.tools import ManufacturingToolbox
from semikb.agent_runtime.web_search import AliyunWebSearchGateway
from semikb.config import Settings
from semikb.contracts.models import (
    ActorScope,
    AgentAnswer,
    AnswerClaim,
    AuditEvent,
    Chunk,
    EvidenceLedgerEntry,
    RetrievalConstraints,
)
from semikb.storage.conversations import ConversationRepository


class CaseState(TypedDict, total=False):
    request: str
    thread_id: str
    run_id: str
    user_scope: dict[str, Any]
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

        workflow = StateGraph(CaseState)
        workflow.add_node("ingest_request", self._ingest_request)
        workflow.add_node("authorize_scope", self._authorize_scope)
        workflow.add_node("classify_and_extract", self._classify_and_extract)
        workflow.add_node("validate_required_fields", self._validate_required_fields)
        workflow.add_node("clarify_missing_fields", self._clarify_missing_fields)
        workflow.add_node("retrieve_evidence", self._retrieve_evidence)
        workflow.add_node("build_evidence_ledger", self._build_evidence_ledger)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("verify_answer", self._verify_answer)
        workflow.add_node("insufficient_information", self._insufficient_information)
        workflow.add_node("audit", self._audit)

        workflow.add_edge(START, "ingest_request")
        workflow.add_edge("ingest_request", "authorize_scope")
        workflow.add_edge("authorize_scope", "classify_and_extract")
        workflow.add_edge("classify_and_extract", "validate_required_fields")
        workflow.add_conditional_edges(
            "validate_required_fields",
            self._route_after_validation,
            {
                "clarify": "clarify_missing_fields",
                "retrieve": "retrieve_evidence",
                "insufficient": "insufficient_information",
            },
        )
        workflow.add_edge("clarify_missing_fields", "classify_and_extract")
        workflow.add_edge("retrieve_evidence", "build_evidence_ledger")
        workflow.add_edge("build_evidence_ledger", "generate_answer")
        workflow.add_edge("generate_answer", "verify_answer")
        workflow.add_edge("verify_answer", "audit")
        workflow.add_edge("insufficient_information", "audit")
        workflow.add_edge("audit", END)
        self.compiled = workflow.compile(checkpointer=checkpointer, store=memory_service.store)

    async def _ingest_request(self, state: CaseState) -> CaseState:
        preferences = await asyncio.to_thread(
            self.memory_service.approved_preferences,
            state["user_scope"]["user_id"],
        )
        return {
            "status": "running",
            "constraints": {},
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
        deterministic = self._deterministic_extract(state["request"])
        llm_result: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        if not self.settings.demo_mode:
            try:
                completion = await self.llm.complete(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Extract semiconductor request constraints. Return JSON only with keys "
                                "intent, risk_level, product, process_layer, tool_id, chamber, recipe_id, "
                                "recipe_version, time_range, lot_id. Use null when absent. Do not infer "
                                "identifiers that the user did not provide."
                            ),
                        },
                        {"role": "user", "content": state["request"]},
                    ],
                    response_json=True,
                    max_output_tokens=320,
                )
                parsed = json.loads(completion.content)
                if isinstance(parsed, dict):
                    llm_result = parsed
                metadata = {
                    "extract_provider": completion.provider,
                    "extract_model": completion.reported_model,
                    "extract_fallback_used": completion.fallback_used,
                }
            except (ValueError, json.JSONDecodeError, RuntimeError) as exc:
                metadata = {"extract_warning": type(exc).__name__}

        constraints = {
            key: value
            for key in (
                "product",
                "process_layer",
                "tool_id",
                "chamber",
                "recipe_id",
                "recipe_version",
                "time_range",
                "lot_id",
            )
            if (value := llm_result.get(key)) not in (None, "")
            and self._constraint_is_grounded(key, value, state["request"])
        }
        constraints.update(deterministic["constraints"])
        intent = str(llm_result.get("intent") or deterministic["intent"])
        if intent not in {"knowledge_qa", "anomaly_investigation", "recipe_impact", "report"}:
            intent = deterministic["intent"]
        risk = str(llm_result.get("risk_level") or deterministic["risk_level"])
        if risk not in {"low", "medium", "high"}:
            risk = deterministic["risk_level"]
        return {
            "intent": intent,
            "risk_level": risk,
            "constraints": constraints,
            "retrieval_query": state["request"],
            "model_metadata": {**state.get("model_metadata", {}), **metadata},
        }

    @staticmethod
    def _constraint_is_grounded(field: str, value: Any, request: str) -> bool:
        """Never turn an LLM-inferred identifier into a retrieval filter."""

        normalized_value = re.sub(r"\s+", "", str(value)).lower()
        normalized_request = re.sub(r"\s+", "", request).lower()
        if not normalized_value:
            return False
        if field == "time_range":
            return any(char.isdigit() for char in normalized_value) and any(
                token in normalized_request for token in ("小时", "天", "周", "最近", "过去", "-")
            )
        return normalized_value in normalized_request

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
        chamber = re.search(r"(?:CHAMBER|腔体)\s*[-:]?\s*([A-Z0-9]+)", normal)
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

        missing: list[str] = []
        if state.get("intent") in {"anomaly_investigation", "recipe_impact"}:
            if not constraints.get("product"):
                missing.append("product")
            if not constraints.get("time_range"):
                missing.append("time_range")
            if not (constraints.get("tool_id") or constraints.get("chamber")):
                missing.append("tool_or_chamber")
        prompts = {
            "product": "受影响的 Product 是什么？",
            "time_range": "异常从何时开始，或需要查询哪个时间范围？",
            "tool_or_chamber": "涉及哪个 Tool 或 Chamber？若未知，请提供已知 FDC 报警或 Lot 范围。",
        }
        return {
            "authorization_errors": sorted(set(authorization_errors)),
            "missing_required_fields": missing[:3],
            "clarification_questions": [prompts[field] for field in missing[:3]],
        }

    def _route_after_validation(
        self,
        state: CaseState,
    ) -> Literal["clarify", "retrieve", "insufficient"]:
        if state.get("authorization_errors"):
            return "insufficient"
        if not state.get("missing_required_fields"):
            return "retrieve"
        if state.get("clarification_round", 0) >= self.settings.agent_max_clarification_rounds:
            return "insufficient"
        return "clarify"

    @staticmethod
    def _clarify_missing_fields(state: CaseState) -> CaseState:
        payload = {
            "kind": "clarification",
            "round": state.get("clarification_round", 0) + 1,
            "missing_fields": state.get("missing_required_fields", []),
            "questions": state.get("clarification_questions", []),
        }
        response = interrupt(payload)
        merged = f"{state['request']}\n用户补充：{response}"
        return {
            "request": merged,
            "clarification_response": str(response),
            "clarification_round": state.get("clarification_round", 0) + 1,
        }

    async def _retrieve_evidence(self, state: CaseState) -> CaseState:
        actor_scope = ActorScope.model_validate(state["user_scope"])
        constraints = self._retrieval_constraints(state.get("constraints", {}))
        retrieval_task = asyncio.to_thread(
            self.retrieval.search,
            state["retrieval_query"],
            actor_scope,
            top_k=5,
            thread_id=state["thread_id"],
            constraints=constraints,
        )
        web_task = None
        if self.web_search.should_search(state["retrieval_query"]):
            web_task = asyncio.create_task(self.web_search.search(state["retrieval_query"]))

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
        trace.external_evidence = external
        self.retrieval.save_trace(trace)
        live_data = self.toolbox.query_for_case(state["request"], state.get("constraints", {}))
        return {
            "retrieval_routes": trace.routes,
            "candidate_ids": [candidate.chunk_id for candidate in trace.candidates],
            "reranked_evidence": [chunk.model_dump(mode="json") for chunk in evidence],
            "image_evidence": trace.image_asset_ids,
            "external_evidence": external,
            "live_data_refs": live_data,
            "trace_id": trace.trace_id,
        }

    @staticmethod
    def _retrieval_constraints(values: dict[str, Any]) -> RetrievalConstraints:
        allowed = set(RetrievalConstraints.model_fields)
        return RetrievalConstraints.model_validate(
            {key: value for key, value in values.items() if key in allowed}
        )

    def _build_evidence_ledger(self, state: CaseState) -> CaseState:
        trace = self.retrieval.get_trace(
            str(state["trace_id"]),
            ActorScope.model_validate(state["user_scope"]),
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
        return {"evidence_ledger": [item.model_dump(mode="json") for item in ledger]}

    async def _generate_answer(self, state: CaseState) -> CaseState:
        ledger = [EvidenceLedgerEntry.model_validate(item) for item in state.get("evidence_ledger", [])]
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
            try:
                completion = await self.llm.complete(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are a semiconductor investigation assistant. Use only the supplied evidence ledger. "
                                "Internal controlled evidence outranks external evidence. Simulated live data must remain "
                                "labeled simulated. Do not declare a root cause; hypotheses must be testable. Return JSON "
                                "with facts, hypotheses (each has text and citation_ids), unknowns, next_actions, confidence. "
                                "Every fact must cite one or more exact evidence_id values."
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
                    response_json=True,
                    max_output_tokens=self.settings.agent_answer_max_output_tokens,
                )
                answer = self._parse_answer(json.loads(completion.content))
                metadata.update(
                    {
                        "answer_provider": completion.provider,
                        "answer_model": completion.reported_model,
                        "answer_fallback_used": completion.fallback_used,
                    }
                )
            except (ValueError, json.JSONDecodeError, RuntimeError) as exc:
                metadata["answer_warning"] = type(exc).__name__
        answer = answer or self._deterministic_answer(ledger)
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
            return [str(item) for item in value] if isinstance(value, list) else []

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
        return {
            "answer": verified.model_dump(mode="json"),
            "answer_text": ConversationGraph._format_answer(verified),
            "citations": citations,
            "status": "completed",
            "verification_warnings": warnings,
        }

    @staticmethod
    def _format_answer(answer: AgentAnswer) -> str:
        lines = ["基于当前有效且有权限访问的受控证据："]
        if answer.facts:
            lines.append("\n已知事实")
            lines.extend(
                f"- {claim.text} [{', '.join(claim.citation_ids)}]" for claim in answer.facts
            )
        if answer.hypotheses:
            lines.append("\n待验证假设")
            lines.extend(
                f"- {claim.text} [{', '.join(claim.citation_ids)}]"
                for claim in answer.hypotheses
            )
        if answer.unknowns:
            lines.append("\n仍不确定")
            lines.extend(f"- {item}" for item in answer.unknowns)
        if answer.next_actions:
            lines.append("\n建议下一步")
            lines.extend(f"- {item}" for item in answer.next_actions)
        lines.append(f"\n置信度：{answer.confidence}")
        return "\n".join(lines)

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
