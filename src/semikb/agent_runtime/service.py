"""Continuous-thread orchestration for evidence-bound semiconductor assistance."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from semikb.agent_runtime.graph import ConversationGraph
from semikb.agent_runtime.llm_gateway import OpenAICompatibleLLMGateway
from semikb.agent_runtime.memory import MemoryService
from semikb.agent_runtime.tools import ManufacturingToolbox
from semikb.agent_runtime.web_search import AliyunWebSearchGateway
from semikb.config import Settings
from semikb.contracts.models import ActorScope, ChatMessage, ThreadRecord, new_id
from semikb.storage.conversations import ConversationRepository


class ConversationService:
    """Own thread metadata while LangGraph owns checkpointed execution state."""

    def __init__(
        self,
        repository: ConversationRepository,
        retrieval: Any,
        settings: Settings,
        *,
        checkpointer: Any | None = None,
        long_term_store: Any | None = None,
        llm: OpenAICompatibleLLMGateway | None = None,
        web_search: AliyunWebSearchGateway | None = None,
        toolbox: ManufacturingToolbox | None = None,
    ) -> None:
        self.repository = repository
        self.retrieval = retrieval
        self.settings = settings
        self.checkpointer = checkpointer or InMemorySaver()
        self.long_term_store = long_term_store or InMemoryStore()
        self.memory = MemoryService(self.long_term_store)
        self.graph = ConversationGraph(
            settings=settings,
            repository=repository,
            retrieval=retrieval,
            checkpointer=self.checkpointer,
            memory_service=self.memory,
            llm=llm,
            web_search=web_search,
            toolbox=toolbox,
        )

    def create_thread(self, title: str, actor_scope: ActorScope) -> ThreadRecord:
        return self.repository.create_thread(ThreadRecord(title=title, actor_scope=actor_scope))

    def get_thread(self, thread_id: str, actor_scope: ActorScope | None = None) -> ThreadRecord | None:
        thread = self.repository.get_thread(thread_id)
        if thread is None:
            return None
        if actor_scope and "admin" not in actor_scope.roles and thread.actor_scope.user_id != actor_scope.user_id:
            return None
        return thread

    def list_threads(self, actor_scope: ActorScope) -> list[ThreadRecord]:
        return self.repository.list_threads(actor_scope.user_id)

    async def send_message(
        self,
        thread_id: str,
        content: str,
        actor_scope: ActorScope | None = None,
    ) -> dict[str, Any]:
        thread = self.get_thread(thread_id, actor_scope)
        if thread is None:
            raise KeyError(thread_id)
        thread.messages.append(ChatMessage(role="user", content=content))
        self.repository.save_thread(thread)

        config = {"configurable": {"thread_id": thread.thread_id}}
        if thread.status == "waiting_for_clarification":
            result = await self.graph.compiled.ainvoke(Command(resume=content), config=config)
        else:
            result = await self.graph.compiled.ainvoke(
                {
                    "request": content,
                    "thread_id": thread.thread_id,
                    "run_id": new_id("run"),
                    "user_scope": thread.actor_scope.model_dump(mode="json"),
                    "clarification_round": 0,
                },
                config=config,
            )

        interrupt_payload = self._interrupt_payload(result)
        if interrupt_payload is not None:
            questions = [str(question) for question in interrupt_payload.get("questions", [])]
            response = "为避免猜测 Tool、Product 或时间范围，请补充：\n" + "\n".join(
                f"- {question}" for question in questions
            )
            thread.messages.append(ChatMessage(role="assistant", content=response))
            thread.status = "waiting_for_clarification"
            thread.pending_fields = [
                str(field) for field in interrupt_payload.get("missing_fields", [])
            ]
            thread.clarification_round = int(interrupt_payload.get("round", 1))
            thread.summary = self._summarize(thread)
            self.repository.save_thread(thread)
            return {
                "thread": thread,
                "response": response,
                "clarification_required": True,
                "missing_fields": thread.pending_fields,
                "clarification_round": thread.clarification_round,
            }

        response = str(result.get("answer_text") or "系统未生成可验证答复。")
        citations = list(result.get("citations", []))
        thread.messages.append(ChatMessage(role="assistant", content=response, citations=citations))
        thread.summary = self._summarize(thread)
        thread.pending_fields = []
        thread.clarification_round = 0
        thread.status = "active"
        self.repository.save_thread(thread)
        return {
            "thread": thread,
            "response": response,
            "clarification_required": False,
            "status": result.get("status", "completed"),
            "answer": result.get("answer", {}),
            "citations": citations,
            "trace_id": result.get("trace_id"),
            "image_asset_ids": result.get("image_evidence", []),
            "tool_facts": result.get("live_data_refs", []),
            "external_evidence": result.get("external_evidence", []),
            "evidence_ledger": result.get("evidence_ledger", []),
            "model_metadata": result.get("model_metadata", {}),
            "verification_warnings": result.get("verification_warnings", []),
        }

    @staticmethod
    def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
        values = result.get("__interrupt__", ())
        if not values:
            return None
        payload = getattr(values[0], "value", values[0])
        return payload if isinstance(payload, dict) else {"questions": [str(payload)]}

    @staticmethod
    def _summarize(thread: ThreadRecord) -> str:
        user_messages = [message.content for message in thread.messages if message.role == "user"]
        return user_messages[-1][:240] if user_messages else ""
