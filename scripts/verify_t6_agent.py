"""Run and clean up the real T6 interrupt/resume acceptance scenario."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.store.mongodb import MongoDBStore

from semikb.agent_runtime.service import ConversationService
from semikb.bootstrap import ApplicationContainer
from semikb.config import Settings
from semikb.contracts.models import ActorScope, CreateMemoryRequest
from semikb.storage.conversations import MongoConversationRepository


async def verify() -> dict[str, object]:
    settings = Settings(demo_mode=False)
    container = ApplicationContainer(settings)
    suffix = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    actor = ActorScope(user_id=f"t6_acceptance_{suffix}")
    thread = container.conversations.create_thread("T6 restart acceptance", actor)
    memory_id: str | None = None
    trace_id: str | None = None
    restarted_repo: MongoConversationRepository | None = None
    restarted_checkpointer: MongoDBSaver | None = None
    try:
        first = await container.conversations.send_message(
            thread.thread_id,
            "最近蚀刻良率下降，帮我调查根因。",
            actor,
        )
        if not first.get("clarification_required"):
            raise AssertionError("The incomplete request did not interrupt for clarification.")

        database = container.conversation_store.database
        checkpoints_before_resume = database["checkpoints"].count_documents(
            {"thread_id": thread.thread_id}
        )
        if checkpoints_before_resume < 1:
            raise AssertionError("No MongoDB checkpoint was persisted before resume.")

        restarted_repo = MongoConversationRepository(settings)
        restarted_checkpointer = MongoDBSaver(
            restarted_repo.client,
            db_name=settings.mongodb_database,
            checkpoint_collection_name="checkpoints",
            writes_collection_name="checkpoint_writes",
        )
        restarted = ConversationService(
            restarted_repo,
            container.retrieval,
            settings,
            checkpointer=restarted_checkpointer,
            long_term_store=MongoDBStore(restarted_repo.database["long_term_memories"]),
        )
        second = await restarted.send_message(
            thread.thread_id,
            "P-ALPHA 最近24小时 ETCH-03 Chamber B 出现 pressure alarm 和首片异常。",
            actor,
        )
        trace_id = str(second.get("trace_id") or "")
        if second.get("status") != "completed" or not trace_id:
            raise AssertionError("The resumed graph did not complete with a retrieval trace.")
        if not second.get("evidence_ledger") or not second.get("citations"):
            raise AssertionError("The answer is missing its evidence ledger or citations.")
        ledger_ids = {item["evidence_id"] for item in second["evidence_ledger"]}
        citation_ids = {item["evidence_id"] for item in second["citations"]}
        if not any(evidence_id.startswith("chunk:") for evidence_id in ledger_ids):
            raise AssertionError("The completed investigation has no internal controlled evidence.")
        if not citation_ids.issubset(ledger_ids):
            raise AssertionError("Answer citations are not closed over the evidence ledger.")
        if "根因是" in second["response"]:
            raise AssertionError("The answer declared an unverified root cause.")

        memory = restarted.memory.create(
            CreateMemoryRequest(
                memory_type="preference",
                content="回答时先展示受控证据，再列出待验证假设。",
                source_refs=[f"thread:{thread.thread_id}"],
            ),
            actor,
        )
        memory_id = memory.memory_id
        if [item.memory_id for item in restarted.memory.list(actor)] != [memory_id]:
            raise AssertionError("Approved long-term memory was not persisted for the user.")

        audit_count = database["audit_events"].count_documents({"thread_id": thread.thread_id})
        if audit_count != 4:
            raise AssertionError("Agent completion and three read-only tool calls were not fully audited.")
        checkpoint_count = database["checkpoints"].count_documents({"thread_id": thread.thread_id})
        return {
            "thread_id": thread.thread_id,
            "interrupt_round": first.get("clarification_round"),
            "checkpoint_count": checkpoint_count,
            "trace_id": trace_id,
            "evidence_ids": sorted(ledger_ids),
            "citation_ids": sorted(citation_ids),
            "image_asset_ids": second.get("image_asset_ids", []),
            "tool_names": [item["tool"] for item in second.get("tool_facts", [])],
            "model_metadata": second.get("model_metadata", {}),
            "audit_count": audit_count,
            "memory_persisted": True,
        }
    finally:
        service = (
            None
            if restarted_repo is None or restarted_checkpointer is None
            else ConversationService(
                restarted_repo,
                container.retrieval,
                settings,
                checkpointer=restarted_checkpointer,
                long_term_store=MongoDBStore(restarted_repo.database["long_term_memories"]),
            )
        )
        if service is not None and memory_id is not None:
            try:
                service.memory.delete(memory_id, actor)
            except KeyError:
                pass
        saver = restarted_checkpointer or container.conversations.checkpointer
        saver.delete_thread(thread.thread_id)
        database = container.conversation_store.database
        database["agent_threads"].delete_many({"thread_id": thread.thread_id})
        database["audit_events"].delete_many({"thread_id": thread.thread_id})
        database["retrieval_traces"].delete_many({"thread_id": thread.thread_id})
        if restarted_repo is not None:
            restarted_repo.client.close()


def main() -> None:
    print(json.dumps(asyncio.run(verify()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
