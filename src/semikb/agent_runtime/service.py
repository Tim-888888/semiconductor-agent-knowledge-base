"""Continuous-thread orchestration for evidence-bound semiconductor assistance."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from semikb.agent_runtime.context import ContextAssembler
from semikb.agent_runtime.graph import ConversationGraph
from semikb.agent_runtime.llm_gateway import LLMProviderError, OpenAICompatibleLLMGateway
from semikb.agent_runtime.memory import MemoryService
from semikb.agent_runtime.presentation import build_message_presentation
from semikb.agent_runtime.tools import ManufacturingToolbox
from semikb.agent_runtime.web_search import AliyunWebSearchGateway
from semikb.config import Settings
from semikb.contracts.models import (
    ActorScope,
    AffectSignals,
    AgentAnswer,
    AgentRoute,
    ChatMessage,
    IntentTaskItem,
    InteractionMode,
    RouteTaskDecision,
    SendMessageResponse,
    SlotOperation,
    TaskExecutionResult,
    ThreadRecord,
    new_id,
)
from semikb.contracts.streaming import (
    AgentMessageRequestRecord,
    AgentMessageRequestStatus,
    AgentStreamErrorCode,
    AgentStreamEvent,
    AgentStreamStage,
    DirectReplyAudit,
    StreamAcceptedData,
    StreamAcceptedEvent,
    StreamAnswerDeltaData,
    StreamAnswerDeltaEvent,
    StreamCompletedData,
    StreamCompletedEvent,
    StreamErrorData,
    StreamErrorEvent,
    StreamEvidenceData,
    StreamEvidenceEvent,
    StreamHeartbeatData,
    StreamHeartbeatEvent,
    StreamStageData,
    StreamStageEvent,
    StreamTaskStatusData,
    StreamTaskStatusEvent,
    UnderstandingAudit,
)
from semikb.storage.conversations import ConversationRepository


@dataclass(frozen=True, slots=True)
class PreparedStreamMessage:
    content: str
    record: AgentMessageRequestRecord
    replayed: bool
    resume_checkpoint: bool


@dataclass(slots=True)
class ActiveStreamControl:
    cancelled: asyncio.Event
    graph_task: asyncio.Task[None] | None = None


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
        self.context = ContextAssembler(settings)
        self._active_streams: dict[tuple[str, str, str], ActiveStreamControl] = {}
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
        return self._hydrate_message_presentations(thread)

    def list_threads(self, actor_scope: ActorScope) -> list[ThreadRecord]:
        return self.repository.list_threads(actor_scope.user_id)

    async def send_message(
        self,
        thread_id: str,
        content: str,
        actor_scope: ActorScope | None = None,
    ) -> dict[str, Any]:
        """Retain the non-streaming route while sharing ordering and persistence semantics."""

        thread = self.get_thread(thread_id, actor_scope)
        if thread is None:
            raise KeyError(thread_id)
        scope = actor_scope or thread.actor_scope
        prepared = await self.prepare_stream_message(
            thread_id,
            content,
            new_id("req"),
            scope,
        )
        stream = self.stream_message(prepared)
        completed: dict[str, Any] | None = None
        try:
            async for event in stream:
                if isinstance(event, StreamCompletedEvent):
                    completed = event.data.result.model_dump(mode="python")
                elif isinstance(event, StreamErrorEvent):
                    raise RuntimeError(event.data.message)
        finally:
            await stream.aclose()
        if completed is not None:
            return completed
        raise RuntimeError("non-streaming message ended without a terminal event")

    async def prepare_stream_message(
        self,
        thread_id: str,
        content: str,
        request_id: str,
        actor_scope: ActorScope,
    ) -> PreparedStreamMessage:
        """Reserve idempotency and persist the user message before SSE headers are sent."""

        thread = self.get_thread(thread_id, actor_scope)
        if thread is None:
            raise KeyError(thread_id)
        record = AgentMessageRequestRecord(
            request_id=request_id,
            thread_id=thread_id,
            actor_user_id=actor_scope.user_id,
            content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            user_message_id=new_id("msg"),
            run_id=new_id("run"),
        )
        prepared, replayed = await asyncio.to_thread(
            self.repository.prepare_message_request,
            record,
        )
        if not replayed:
            user_message = ChatMessage(
                message_id=prepared.user_message_id,
                request_id=prepared.request_id,
                run_id=prepared.run_id,
                turn_seq=prepared.user_turn_seq,
                role="user",
                content=content,
            )
            try:
                await asyncio.to_thread(
                    self.repository.append_message_once,
                    thread_id,
                    user_message,
                )
            except Exception:
                await self._mark_failed(
                    prepared,
                    AgentMessageRequestStatus.FAILED,
                    AgentStreamErrorCode.INTERNAL_ERROR,
                )
                raise
        return PreparedStreamMessage(
            content=content,
            record=prepared,
            replayed=replayed,
            resume_checkpoint=thread.status == "waiting_for_clarification",
        )

    async def stream_message(
        self,
        prepared: PreparedStreamMessage,
    ) -> AsyncIterator[AgentStreamEvent]:
        """Run one prepared message and emit ordered, replayable domain events."""

        record = prepared.record
        request_key = (record.actor_user_id, record.thread_id, record.request_id)
        control = ActiveStreamControl(cancelled=asyncio.Event())
        self._active_streams[request_key] = control
        sequence = 0
        started = time.perf_counter()

        def envelope(event_type: type[AgentStreamEvent], data: Any) -> AgentStreamEvent:
            nonlocal sequence
            sequence += 1
            return event_type(
                request_id=record.request_id,
                thread_id=record.thread_id,
                sequence=sequence,
                data=data,
            )

        yield envelope(
            StreamAcceptedEvent,
            StreamAcceptedData(
                message_id=record.user_message_id,
                run_id=record.run_id,
                attempt=record.attempt,
                replayed=prepared.replayed,
            ),
        )

        if prepared.replayed:
            if self._active_streams.get(request_key) is control:
                self._active_streams.pop(request_key, None)
            result = self._replayed_response(record)
            yield envelope(
                StreamCompletedEvent,
                StreamCompletedData(run_id=record.run_id, result=result),
            )
            return

        try:
            record = await asyncio.to_thread(self.repository.mark_message_request_running, record)
            final_state: dict[str, Any] = {}
            streamed_answer = ""
            graph_input: dict[str, Any] | Command
            thread = self.repository.get_thread(record.thread_id)
            if thread is None:
                raise KeyError(record.thread_id)
            preferences = await asyncio.to_thread(
                self.memory.approved_preferences,
                record.actor_user_id,
            )
            conversation_context = self.context.assemble(
                thread,
                current_message_id=record.user_message_id,
                approved_preferences=preferences,
            ).model_dump(mode="json")
            if prepared.resume_checkpoint:
                graph_input = Command(
                    resume=prepared.content,
                    update={
                        "conversation_context": conversation_context,
                        "approved_preferences": preferences,
                    },
                )
            else:
                graph_input = {
                    "request": prepared.content,
                    "thread_id": record.thread_id,
                    "run_id": record.run_id,
                    "user_scope": thread.actor_scope.model_dump(mode="json"),
                    "clarification_round": 0,
                    "conversation_context": conversation_context,
                    "approved_preferences": preferences,
                }

            config = {"configurable": {"thread_id": record.thread_id}}
            graph_stream = self.graph.compiled.astream(
                graph_input,
                config=config,
                stream_mode=["custom", "values"],
            )
            graph_events: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

            async def pump_graph_events() -> None:
                try:
                    async for mode, payload in graph_stream:
                        await graph_events.put((mode, payload))
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    await graph_events.put(("error", exc))
                finally:
                    await graph_events.put(("done", None))

            graph_task = asyncio.create_task(pump_graph_events())
            control.graph_task = graph_task
            try:
                while True:
                    if control.cancelled.is_set():
                        raise asyncio.CancelledError
                    try:
                        mode, payload = await asyncio.wait_for(
                            graph_events.get(),
                            timeout=self.settings.agent_stream_heartbeat_seconds,
                        )
                    except TimeoutError:
                        elapsed_ms = int((time.perf_counter() - started) * 1000)
                        yield envelope(
                            StreamHeartbeatEvent,
                            StreamHeartbeatData(elapsed_ms=elapsed_ms),
                        )
                        continue
                    if mode == "done":
                        break
                    if mode == "error":
                        raise payload
                    if mode == "values" and isinstance(payload, dict):
                        final_state = payload
                    elif mode == "custom" and isinstance(payload, dict):
                        event = self._custom_stream_event(record, sequence + 1, payload)
                        if event is not None:
                            sequence += 1
                            if isinstance(event, StreamAnswerDeltaEvent):
                                streamed_answer += event.data.delta
                            yield event
            finally:
                if not graph_task.done():
                    graph_task.cancel()
                await asyncio.gather(graph_task, return_exceptions=True)

            if control.cancelled.is_set():
                raise asyncio.CancelledError
            thread = self.repository.get_thread(record.thread_id)
            if thread is None:
                raise KeyError(record.thread_id)
            self._apply_route_metadata(record, final_state)
            response_model, assistant, thread_state = self._build_response(
                thread,
                final_state,
                request_id=record.request_id,
                run_id=record.run_id,
            )
            if streamed_answer != response_model.response:
                if response_model.response.startswith(streamed_answer):
                    suffix = response_model.response[len(streamed_answer) :]
                    if suffix:
                        yield envelope(
                            StreamAnswerDeltaEvent,
                            StreamAnswerDeltaData(delta=suffix),
                        )
                else:
                    raise ValueError("verified answer differs from streamed answer")

            yield envelope(
                StreamStageEvent,
                StreamStageData(
                    stage=AgentStreamStage.PERSISTING_RESULT,
                    message="正在保存最终回答与会话状态",
                ),
            )
            summary_thread = thread.model_copy(deep=True)
            summary_thread.messages.append(assistant)
            compaction = self.context.compact_thread(summary_thread)
            active_context = self.context.update_active_context(
                summary_thread,
                final_state,
                source_message_id=record.user_message_id,
            )
            persisted_thread = await asyncio.to_thread(
                self.repository.finalize_stream_response,
                record.thread_id,
                assistant,
                status=thread_state["status"],
                summary=compaction.summary,
                summary_upto_message_id=compaction.summary_upto_message_id,
                active_context=active_context,
                context_version=max(thread.context_version, 1),
                pending_fields=thread_state["pending_fields"],
                clarification_round=thread_state["clarification_round"],
            )
            response_model.thread = self._hydrate_message_presentations(persisted_thread)
            record.assistant_turn_seq = assistant.turn_seq
            result_payload = response_model.model_dump(mode="python", exclude={"thread"})
            record = await asyncio.to_thread(
                self.repository.mark_message_request_terminal,
                record,
                AgentMessageRequestStatus.COMPLETED,
                result_payload=result_payload,
                assistant_message_id=assistant.message_id,
                trace_id=response_model.trace_id,
            )
            yield envelope(
                StreamCompletedEvent,
                StreamCompletedData(run_id=record.run_id, result=response_model),
            )
        except asyncio.CancelledError:
            await self._mark_failed(record, AgentMessageRequestStatus.CANCELLED, AgentStreamErrorCode.CANCELLED)
            if control.cancelled.is_set():
                return
            raise
        except GeneratorExit:
            await self._mark_failed(record, AgentMessageRequestStatus.CANCELLED, AgentStreamErrorCode.CANCELLED)
            raise
        except Exception as exc:
            code, message, retryable = self._safe_stream_error(exc)
            await self._mark_failed(record, AgentMessageRequestStatus.FAILED, code)
            yield envelope(
                StreamErrorEvent,
                StreamErrorData(
                    code=code,
                    message=message,
                    retryable=retryable,
                    trace_id=record.trace_id,
                ),
            )
        finally:
            if self._active_streams.get(request_key) is control:
                self._active_streams.pop(request_key, None)

    async def cancel_stream_message(
        self,
        thread_id: str,
        request_id: str,
        actor_scope: ActorScope,
    ) -> AgentMessageRequestRecord:
        """Cancel a live run before the browser closes its SSE connection."""

        thread = self.get_thread(thread_id, actor_scope)
        if thread is None:
            raise KeyError(thread_id)
        record = await asyncio.to_thread(
            self.repository.get_message_request,
            thread_id,
            actor_scope.user_id,
            request_id,
        )
        if record is None:
            raise KeyError(request_id)
        if record.status is AgentMessageRequestStatus.COMPLETED:
            return record
        control = self._active_streams.get(
            (actor_scope.user_id, thread_id, request_id),
            None,
        )
        if control is not None:
            control.cancelled.set()
            if control.graph_task is not None and not control.graph_task.done():
                control.graph_task.cancel()
                await asyncio.gather(control.graph_task, return_exceptions=True)
        if record.status in {
            AgentMessageRequestStatus.ACCEPTED,
            AgentMessageRequestStatus.RUNNING,
        }:
            record = await asyncio.to_thread(
                self.repository.mark_message_request_terminal,
                record,
                AgentMessageRequestStatus.CANCELLED,
                error_code=AgentStreamErrorCode.CANCELLED,
            )
        return record

    async def _invoke_graph(self, thread: ThreadRecord, content: str) -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread.thread_id}}
        preferences = await asyncio.to_thread(
            self.memory.approved_preferences,
            thread.actor_scope.user_id,
        )
        context = self.context.assemble(
            thread,
            approved_preferences=preferences,
        ).model_dump(mode="json")
        if thread.status == "waiting_for_clarification":
            return await self.graph.compiled.ainvoke(
                Command(
                    resume=content,
                    update={"conversation_context": context, "approved_preferences": preferences},
                ),
                config=config,
            )
        return await self.graph.compiled.ainvoke(
            {
                "request": content,
                "thread_id": thread.thread_id,
                "run_id": new_id("run"),
                "user_scope": thread.actor_scope.model_dump(mode="json"),
                "clarification_round": 0,
                "conversation_context": context,
                "approved_preferences": preferences,
            },
            config=config,
        )

    def _build_response(
        self,
        thread: ThreadRecord,
        result: dict[str, Any],
        *,
        request_id: str | None = None,
        run_id: str | None = None,
    ) -> tuple[SendMessageResponse, ChatMessage, dict[str, Any]]:
        interrupt_payload = self._interrupt_payload(result)
        if interrupt_payload is not None:
            questions = [str(question) for question in interrupt_payload.get("questions", [])]
            response = str(interrupt_payload.get("response") or "").strip()
            if not response:
                response = "为了准确处理这个问题，我还需要确认：\n" + "\n".join(
                    f"- {question}" for question in questions
                )
            pending_fields = [str(field) for field in interrupt_payload.get("missing_fields", [])]
            clarification_round = int(interrupt_payload.get("round", 1))
            assistant = ChatMessage(
                request_id=request_id,
                run_id=run_id,
                role="assistant",
                content=response,
                presentation=build_message_presentation(
                    route=result.get("route_decision", AgentRoute.CLARIFY),
                    answer=None,
                    status="waiting_for_clarification",
                    trace_id=None,
                    verification_warnings=list(result.get("verification_warnings", [])),
                    task_results=list(result.get("task_results", [])),
                ),
            )
            model = SendMessageResponse(
                thread=thread,
                response=response,
                clarification_required=True,
                missing_fields=pending_fields,
                clarification_round=clarification_round,
                model_metadata=dict(result.get("model_metadata", {})),
                verification_warnings=list(result.get("verification_warnings", [])),
                **self._response_route_metadata(result),
            )
            return model, assistant, {
                "status": "waiting_for_clarification",
                "pending_fields": pending_fields,
                "clarification_round": clarification_round,
            }

        response = str(result.get("answer_text") or "系统未生成可验证答复。")
        citations = list(result.get("citations", []))
        answer_payload = result.get("answer")
        answer = AgentAnswer.model_validate(answer_payload) if answer_payload else None
        assistant = ChatMessage(
            request_id=request_id,
            run_id=run_id,
            role="assistant",
            content=response,
            citations=citations,
            presentation=build_message_presentation(
                route=result.get("route_decision"),
                answer=answer,
                status=str(result.get("status", "completed")),
                trace_id=result.get("trace_id"),
                verification_warnings=list(result.get("verification_warnings", [])),
                task_results=list(result.get("task_results", [])),
            ),
        )
        model = SendMessageResponse(
            thread=thread,
            response=response,
            clarification_required=False,
            status=str(result.get("status", "completed")),
            answer=answer,
            citations=citations,
            trace_id=result.get("trace_id"),
            image_asset_ids=list(result.get("image_evidence", [])),
            tool_facts=list(result.get("live_data_refs", [])),
            external_evidence=list(result.get("external_evidence", [])),
            evidence_ledger=list(result.get("evidence_ledger", [])),
            model_metadata=dict(result.get("model_metadata", {})),
            verification_warnings=list(result.get("verification_warnings", [])),
            **self._response_route_metadata(result),
        )
        return model, assistant, {
            "status": "active",
            "pending_fields": [],
            "clarification_round": 0,
        }

    @staticmethod
    def _response_route_metadata(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "interaction_mode": result.get("interaction_mode"),
            "route_decision": result.get("route_decision"),
            "route_confidence": result.get("route_confidence"),
            "task_items": result.get("task_items", []),
            "task_decisions": result.get("route_plan", {}).get("task_decisions", []),
            "task_results": result.get("task_results", []),
            "retrieval_skipped_reason": result.get("retrieval_skipped_reason"),
        }

    @staticmethod
    def _apply_route_metadata(
        record: AgentMessageRequestRecord,
        result: dict[str, Any],
    ) -> None:
        if result.get("interaction_mode"):
            record.interaction_mode = InteractionMode(str(result["interaction_mode"]))
        if result.get("route_decision"):
            record.route_decision = AgentRoute(str(result["route_decision"]))
        if result.get("route_confidence") is not None:
            record.route_confidence = float(result["route_confidence"])
        record.task_items = [
            IntentTaskItem.model_validate(item) for item in result.get("task_items", [])
        ]
        route_plan = result.get("route_plan", {})
        record.task_decisions = [
            RouteTaskDecision.model_validate(item)
            for item in route_plan.get("task_decisions", [])
        ]
        record.task_results = [
            TaskExecutionResult.model_validate(item)
            for item in result.get("task_results", [])
        ]
        record.context_message_ids = [str(item) for item in result.get("context_message_ids", [])]
        record.standalone_query = str(result.get("standalone_query", ""))
        record.retrieval_skipped_reason = result.get("retrieval_skipped_reason")
        record.slot_operations = [
            SlotOperation.model_validate(item) for item in result.get("slot_operations", [])
        ]
        record.inherited_slots = {
            str(key): str(value) for key, value in result.get("inherited_slots", {}).items()
        }
        record.invalidated_context_refs = [
            str(item) for item in result.get("invalidated_context_refs", [])
        ]
        record.cancel_scope = result.get("cancel_scope")
        understanding = result.get("understanding", {})
        if isinstance(understanding, dict) and understanding.get("affect"):
            record.affect = AffectSignals.model_validate(understanding["affect"])
        metadata = result.get("model_metadata", {})
        if isinstance(metadata, dict):
            allowed = UnderstandingAudit.model_fields.keys()
            record.understanding_audit = UnderstandingAudit.model_validate(
                {key: metadata[key] for key in allowed if key in metadata}
            )
            direct_audit = metadata.get("direct_reply_audit")
            if isinstance(direct_audit, dict):
                record.direct_reply_audit = DirectReplyAudit.model_validate(direct_audit)

    def _replayed_response(self, record: AgentMessageRequestRecord) -> SendMessageResponse:
        thread = self.get_thread(record.thread_id)
        if thread is None:
            raise KeyError(record.thread_id)
        return SendMessageResponse.model_validate({"thread": thread, **record.result_payload})

    def _hydrate_message_presentations(self, thread: ThreadRecord) -> ThreadRecord:
        missing_request_ids = {
            message.request_id
            for message in thread.messages
            if message.role == "assistant"
            and message.presentation is None
            and message.request_id
        }
        if not missing_request_ids:
            return thread

        records = self.repository.list_message_requests(
            thread.thread_id,
            thread.actor_scope.user_id,
        )
        records_by_id = {
            record.request_id: record
            for record in records
            if record.request_id in missing_request_ids
            and record.status is AgentMessageRequestStatus.COMPLETED
        }
        if not records_by_id:
            return thread

        hydrated = thread.model_copy(deep=True)
        for message in hydrated.messages:
            if message.role != "assistant" or message.presentation is not None:
                continue
            record = records_by_id.get(message.request_id or "")
            if record is None:
                continue
            payload = record.result_payload
            message.presentation = build_message_presentation(
                route=payload.get("route_decision") or record.route_decision,
                answer=payload.get("answer"),
                status=str(payload["status"]) if payload.get("status") is not None else None,
                trace_id=str(payload.get("trace_id") or record.trace_id or "") or None,
                verification_warnings=[
                    str(item) for item in payload.get("verification_warnings", [])
                ],
                task_results=list(payload.get("task_results", record.task_results)),
            )
        return hydrated

    @staticmethod
    def _custom_stream_event(
        record: AgentMessageRequestRecord,
        sequence: int,
        payload: dict[str, Any],
    ) -> AgentStreamEvent | None:
        common = {
            "request_id": record.request_id,
            "thread_id": record.thread_id,
            "sequence": sequence,
        }
        kind = payload.get("kind")
        if kind == "stage":
            return StreamStageEvent(
                **common,
                data=StreamStageData(
                    stage=AgentStreamStage(str(payload["stage"])),
                    message=str(payload["message"]),
                ),
            )
        if kind == "task_status":
            return StreamTaskStatusEvent(
                **common,
                data=StreamTaskStatusData(
                    task_id=str(payload["task_id"]),
                    status=str(payload["status"]),
                    route=payload.get("route"),
                    message=str(payload["message"]),
                ),
            )
        if kind == "evidence":
            return StreamEvidenceEvent(
                **common,
                data=StreamEvidenceData(
                    trace_id=payload.get("trace_id"),
                    evidence_ids=[str(item) for item in payload.get("evidence_ids", [])],
                    image_asset_ids=[str(item) for item in payload.get("image_asset_ids", [])],
                    internal_count=int(payload.get("internal_count", 0)),
                    external_count=int(payload.get("external_count", 0)),
                ),
            )
        if kind == "answer_delta" and payload.get("delta"):
            return StreamAnswerDeltaEvent(
                **common,
                data=StreamAnswerDeltaData(
                    delta=str(payload["delta"]),
                    provider=payload.get("provider"),
                    model=payload.get("model"),
                ),
            )
        return None

    async def _mark_failed(
        self,
        record: AgentMessageRequestRecord,
        status: AgentMessageRequestStatus,
        code: AgentStreamErrorCode,
    ) -> None:
        try:
            await asyncio.to_thread(
                self.repository.mark_message_request_terminal,
                record,
                status,
                error_code=code,
                trace_id=record.trace_id,
            )
        except Exception:
            return

    @staticmethod
    def _safe_stream_error(exc: Exception) -> tuple[AgentStreamErrorCode, str, bool]:
        if isinstance(exc, LLMProviderError):
            if exc.status_code == 429:
                return AgentStreamErrorCode.PROVIDER_RATE_LIMITED, "模型服务繁忙，请稍后重试。", True
            if "timed out" in str(exc):
                return AgentStreamErrorCode.PROVIDER_TIMEOUT, "模型响应超时，请重试。", True
            return AgentStreamErrorCode.PROVIDER_UNAVAILABLE, "模型服务暂时不可用，请稍后重试。", True
        if isinstance(exc, ValueError):
            return AgentStreamErrorCode.VERIFICATION_FAILED, "回答校验失败，未保存不完整结果。", True
        return AgentStreamErrorCode.INTERNAL_ERROR, "请求处理失败，未保存不完整结果。", False

    @staticmethod
    def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
        values = result.get("__interrupt__", ())
        if not values:
            return None
        payload = getattr(values[0], "value", values[0])
        return payload if isinstance(payload, dict) else {"questions": [str(payload)]}

    def _summarize(self, thread: ThreadRecord) -> str:
        return self.context.compact_thread(thread).summary
