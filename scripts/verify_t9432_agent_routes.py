"""Verify T9-4.3.2 route decisions and downstream call boundaries on real services."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from semikb.agent_runtime.tools import ManufacturingToolbox
from semikb.bootstrap import ApplicationContainer
from semikb.config import Settings
from semikb.contracts.models import ActorScope, AgentRoute, TaskExecutionDecision


class CountingRetrieval:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.search_calls = 0
        self.reuse_calls = 0

    def search(self, *args: Any, **kwargs: Any) -> Any:
        self.search_calls += 1
        return self.delegate.search(*args, **kwargs)

    def reuse_trace_evidence(self, *args: Any, **kwargs: Any) -> Any:
        self.reuse_calls += 1
        return self.delegate.reuse_trace_evidence(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


class CountingToolbox(ManufacturingToolbox):
    def __init__(self) -> None:
        self.query_calls = 0

    def query_for_case(self, query: str, constraints: dict[str, Any]) -> list[dict[str, Any]]:
        self.query_calls += 1
        return super().query_for_case(query, constraints)


class CountingWeb:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.search_calls = 0

    async def search(self, query: str) -> list[dict[str, str]]:
        self.search_calls += 1
        return await self.delegate.search(query)


def _counts(retrieval: CountingRetrieval, toolbox: CountingToolbox, web: CountingWeb) -> tuple[int, int, int, int]:
    return retrieval.search_calls, retrieval.reuse_calls, toolbox.query_calls, web.search_calls


async def verify() -> dict[str, Any]:
    settings = Settings(demo_mode=False)
    container = ApplicationContainer(settings)
    retrieval = CountingRetrieval(container.retrieval)
    toolbox = CountingToolbox()
    web = CountingWeb(container.conversations.graph.web_search)
    container.conversations.graph.retrieval = retrieval
    container.conversations.graph.toolbox = toolbox
    container.conversations.graph.web_search = web

    suffix = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    actor = ActorScope(
        user_id=f"t9432_acceptance_{suffix}",
        tool_ids=["ETCH-03", "ETCH-04"],
    )
    restricted_actor = ActorScope(
        user_id=f"t9432_restricted_{suffix}",
        tool_ids=["ETCH-03"],
    )
    thread_ids: list[str] = []
    checks: dict[str, bool] = {}

    def create(title: str, scope: ActorScope = actor):
        thread = container.conversations.create_thread(title, scope)
        thread_ids.append(thread.thread_id)
        return thread

    try:
        history_thread = create("T9-4.3.2 history")
        first_question = "当前 ETCH-03 Chamber B 清腔后首片异常 SOP 怎么要求？"
        await container.conversations.send_message(history_thread.thread_id, first_question, actor)
        before = _counts(retrieval, toolbox, web)
        history = await container.conversations.send_message(
            history_thread.thread_id,
            "我刚才问什么了？",
            actor,
        )
        checks["history_direct_exact"] = (
            history["route_decision"] == AgentRoute.HISTORY_DIRECT
            and first_question in history["response"]
            and not history["citations"]
        )
        checks["history_skips_downstream"] = _counts(retrieval, toolbox, web) == before

        before = _counts(retrieval, toolbox, web)
        feedback = await container.conversations.send_message(
            history_thread.thread_id,
            "回答太复杂了",
            actor,
        )
        checks["feedback_chat_direct"] = feedback["route_decision"] == AgentRoute.CHAT_DIRECT
        checks["feedback_skips_downstream"] = _counts(retrieval, toolbox, web) == before

        tool_thread = create("T9-4.3.2 tool")
        before = _counts(retrieval, toolbox, web)
        tool_result = await container.conversations.send_message(
            tool_thread.thread_id,
            "查 P-ALPHA ETCH-03 Chamber B 最近24小时 FDC 报警",
            actor,
        )
        after = _counts(retrieval, toolbox, web)
        checks["tool_only_route"] = tool_result["route_decision"] == AgentRoute.TOOL_ONLY
        checks["tool_only_boundary"] = (
            after[0] == before[0]
            and after[1] == before[1]
            and after[2] == before[2] + 1
            and after[3] == before[3]
        )

        mixed_thread = create("T9-4.3.2 mixed")
        mixed = await container.conversations.send_message(
            mixed_thread.thread_id,
            "查 P-ALPHA ETCH-03 最近24小时 FDC 报警、修改 Recipe、生成报告",
            actor,
        )
        decisions = [item["decision"] for item in mixed["task_decisions"]]
        checks["mixed_three_tasks"] = len(mixed["task_items"]) == 3
        checks["mixed_partial_policy"] = Counter(decisions) == Counter(
            {
                TaskExecutionDecision.EXECUTE: 1,
                TaskExecutionDecision.REFUSE: 1,
                TaskExecutionDecision.DEFER: 1,
            }
        )

        correction_thread = create("T9-4.3.2 correction")
        await container.conversations.send_message(
            correction_thread.thread_id,
            "查 P-ALPHA ETCH-03 Chamber B 最近24小时 FDC 报警",
            actor,
        )
        before = _counts(retrieval, toolbox, web)
        correction = await container.conversations.send_message(
            correction_thread.thread_id,
            "不是 ETCH-03，是 ETCH-04",
            actor,
        )
        updated = container.conversations.get_thread(correction_thread.thread_id, actor)
        slots = updated.active_context.slots if updated else {}
        checks["correction_chat_direct"] = correction["route_decision"] == AgentRoute.CHAT_DIRECT
        checks["correction_skips_downstream"] = _counts(retrieval, toolbox, web) == before
        checks["correction_invalidates_dependencies"] = (
            slots.get("product") is not None
            and slots["product"].valid
            and slots.get("tool_id") is not None
            and slots["tool_id"].value == "ETCH-04"
            and slots.get("chamber") is not None
            and not slots["chamber"].valid
        )

        restricted_thread = create("T9-4.3.2 scope", restricted_actor)
        before = _counts(retrieval, toolbox, web)
        refused = await container.conversations.send_message(
            restricted_thread.thread_id,
            "查 P-ALPHA ETCH-04 Chamber B 最近24小时 FDC 报警",
            restricted_actor,
        )
        checks["scope_refused"] = refused["route_decision"] == AgentRoute.REFUSE
        checks["scope_refusal_skips_downstream"] = _counts(retrieval, toolbox, web) == before

        request_docs = container.conversation_store.database["agent_message_requests"].count_documents(
            {"thread_id": {"$in": thread_ids}, "route_decision": {"$ne": None}}
        )
        checks["route_audit_persisted"] = request_docs >= 8
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise AssertionError(f"T9-4.3.2 route verification failed: {failed}")
        return {
            "checks": checks,
            "downstream_calls": {
                "retrieval_search": retrieval.search_calls,
                "retrieval_reuse": retrieval.reuse_calls,
                "tool": toolbox.query_calls,
                "web": web.search_calls,
            },
            "persisted_route_records": request_docs,
        }
    finally:
        database = container.conversation_store.database
        for thread_id in thread_ids:
            container.conversations.checkpointer.delete_thread(thread_id)
        for collection in (
            "agent_threads",
            "agent_message_requests",
            "audit_events",
            "retrieval_traces",
        ):
            database[collection].delete_many({"thread_id": {"$in": thread_ids}})
        container.conversation_store.client.close()


def main() -> None:
    print(json.dumps(asyncio.run(verify()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
