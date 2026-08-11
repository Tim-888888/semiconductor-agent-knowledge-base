"""Continuous-thread orchestration for evidence-bound semiconductor assistance."""

from __future__ import annotations

import re
from typing import Any

from semikb.agent_runtime.graph import ConversationGraph
from semikb.agent_runtime.tools import query_demo_manufacturing_data
from semikb.agent_runtime.web_search import AliyunWebSearchGateway
from semikb.config import Settings
from semikb.contracts.models import ActorScope, ChatMessage, ThreadRecord
from semikb.rag_retrieval.service import RetrievalService
from semikb.storage.memory import DemoStore


class ConversationService:
    """Coordinates thread memory, clarification, retrieval, and evidence-constrained output."""

    def __init__(self, store: DemoStore, retrieval: RetrievalService, settings: Settings) -> None:
        self.store = store
        self.retrieval = retrieval
        self.settings = settings
        self.web_search = AliyunWebSearchGateway(settings)
        self.graph = ConversationGraph()

    def create_thread(self, title: str, actor_scope: ActorScope) -> ThreadRecord:
        return self.store.create_thread(ThreadRecord(title=title, actor_scope=actor_scope))

    async def send_message(self, thread_id: str, content: str) -> dict[str, Any]:
        thread = self.store.get_thread(thread_id)
        if thread is None:
            raise KeyError(thread_id)
        thread.messages.append(ChatMessage(role="user", content=content))
        missing = self._missing_required_fields(content, thread)
        decision = self.graph.decide(thread.thread_id, content, missing)
        if decision["action"] == "clarify":
            return self._clarify(thread, missing)

        evidence, trace = self.retrieval.search(
            content,
            thread.actor_scope,
            top_k=5,
            thread_id=thread.thread_id,
        )
        tool_facts = query_demo_manufacturing_data(content)
        external_evidence: list[dict[str, str]] = []
        if self.web_search.should_search(content):
            try:
                external_evidence = await self.web_search.search(content)
            except Exception as exc:  # External search is supplementary and must not block internal evidence.
                external_evidence = [{"source_type": "external_unavailable", "content": type(exc).__name__, "url": ""}]
        trace.external_evidence = external_evidence
        self.store.save_trace(trace)

        response = self._compose_response(evidence, tool_facts, external_evidence)
        citations = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "revision": chunk.revision,
                "page_or_section": chunk.page_or_section,
                "image_ids": chunk.image_ids,
            }
            for chunk in evidence
        ]
        thread.messages.append(ChatMessage(role="assistant", content=response, citations=citations))
        thread.summary = self._summarize(thread)
        thread.pending_fields = []
        thread.clarification_round = 0
        self.store.save_thread(thread)
        return {
            "thread": thread,
            "response": response,
            "citations": citations,
            "trace_id": trace.trace_id,
            "image_asset_ids": trace.image_asset_ids,
            "tool_facts": tool_facts,
            "external_evidence": external_evidence,
        }

    def _missing_required_fields(self, content: str, thread: ThreadRecord) -> list[str]:
        normal = content.lower()
        previous_pending = set(thread.pending_fields)
        if previous_pending:
            unresolved = set(previous_pending)
            if re.search(r"p-[a-z0-9]+", normal):
                unresolved.discard("product")
            if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d+\s*(小时|天)", content):
                unresolved.discard("time_range")
            if "etch-" in normal or "chamber" in normal:
                unresolved.discard("tool_or_chamber")
            return sorted(unresolved)

        investigation_terms = ("良率", "yield", "下降", "异常调查", "根因")
        if not any(term in normal for term in investigation_terms):
            return []
        missing: list[str] = []
        if not re.search(r"p-[a-z0-9]+", normal):
            missing.append("product")
        if not re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d+\s*(小时|天)", content):
            missing.append("time_range")
        if "etch-" not in normal and "chamber" not in normal:
            missing.append("tool_or_chamber")
        return missing[:3]

    def _clarify(self, thread: ThreadRecord, missing: list[str]) -> dict[str, Any]:
        if thread.clarification_round >= 2:
            response = "当前信息不足以可靠调查。请提供：" + "、".join(missing) + "。"
            thread.messages.append(ChatMessage(role="assistant", content=response))
            thread.pending_fields = missing
            self.store.save_thread(thread)
            return {"thread": thread, "response": response, "clarification_required": False, "missing_fields": missing}

        prompts = {
            "product": "受影响的 Product 是什么？",
            "time_range": "异常从什么时间开始，或需要查询哪个时间范围？",
            "tool_or_chamber": "涉及哪个 Tool 和 Chamber？如未知，请说明已知的 FDC 报警或 Lot 范围。",
        }
        response = "为避免把模拟数据或经验当成根因，请补充：" + " ".join(prompts[key] for key in missing)
        thread.clarification_round += 1
        thread.pending_fields = missing
        thread.messages.append(ChatMessage(role="assistant", content=response))
        self.store.save_thread(thread)
        return {"thread": thread, "response": response, "clarification_required": True, "missing_fields": missing}

    @staticmethod
    def _compose_response(
        evidence: list[Any], tool_facts: list[dict[str, str]], external_evidence: list[dict[str, str]]) -> str:
        if not evidence:
            return "在当前权限、版本和有效期范围内，没有找到足以支持结论的受控证据。"
        evidence_lines = []
        for chunk in evidence[:3]:
            excerpt = chunk.chunk_text.replace("\n", " ")[:220]
            evidence_lines.append(f"- [{chunk.document_id} {chunk.revision} | {chunk.page_or_section}] {excerpt}")
        sections = ["基于当前有效且有权限访问的受控证据：", *evidence_lines]
        if tool_facts:
            sections.append("模拟只读制造数据：")
            sections.extend(f"- {item['fact']}" for item in tool_facts)
        if external_evidence:
            sections.append("外部资料：仅作补充，不覆盖内部 SOP/Recipe。")
        sections.append("建议：先复核上述 SOP 前置条件和证据范围；未确认前不要修改 Recipe 或设备参数。")
        return "\n".join(sections)

    @staticmethod
    def _summarize(thread: ThreadRecord) -> str:
        user_messages = [message.content for message in thread.messages if message.role == "user"]
        return user_messages[-1][:240] if user_messages else ""
