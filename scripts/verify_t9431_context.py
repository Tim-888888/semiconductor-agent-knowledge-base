"""Verify T9-4.3.1 ordering, restart recovery, isolation, and long context on MongoDB."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from semikb.agent_runtime.context import ContextAssembler
from semikb.config import get_settings
from semikb.contracts.models import ActorScope, ChatMessage, ThreadRecord, new_id
from semikb.contracts.streaming import (
    AgentMessageRequestRecord,
    AgentMessageRequestStatus,
    AgentStreamErrorCode,
)
from semikb.storage.conversations import MongoConversationRepository, ThreadBusyError
from semikb.storage.t9431_context_migration import migrate, rollback


@dataclass(frozen=True, slots=True)
class VerificationResult:
    database: str
    exact_recent_messages: bool
    older_summary_boundary: bool
    restart_recovery: bool
    thread_isolation: bool
    same_thread_ordering: bool
    same_request_stale_recovery: bool
    monotonic_sequences: bool
    migration_rollback: bool
    cleaned_up: bool


def _request(thread: ThreadRecord, request_id: str, content: str) -> AgentMessageRequestRecord:
    return AgentMessageRequestRecord(
        request_id=request_id,
        thread_id=thread.thread_id,
        actor_user_id=thread.actor_scope.user_id,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        user_message_id=new_id("msg"),
        run_id=new_id("run"),
    )


def main() -> None:
    settings = get_settings()
    if settings.demo_mode:
        raise RuntimeError("Set DEMO_MODE=false before running the real MongoDB verifier.")
    owner = f"t9431_verify_{new_id('user')}"
    repo = MongoConversationRepository(settings)
    thread_ids: list[str] = []
    request_ids: list[str] = []
    cleaned_up = False
    verification: VerificationResult | None = None
    try:
        legacy_thread_id = new_id("thread_t9431_legacy_verify")
        thread_ids.append(legacy_thread_id)
        repo.threads.insert_one(
            {
                "thread_id": legacy_thread_id,
                "title": "T9-4.3.1 rollback verifier",
                "actor_scope": ActorScope(user_id=owner).model_dump(mode="python"),
                "status": "active",
                "summary": "",
                "clarification_round": 0,
                "pending_fields": [],
                "messages": [
                    ChatMessage(role="user", content="legacy rollback probe").model_dump(
                        mode="python",
                        exclude={"turn_seq"},
                    )
                ],
            }
        )
        migration_snapshot = Path(f"/tmp/{legacy_thread_id}.json")
        migration_result = migrate(
            settings,
            apply=True,
            snapshot_path=migration_snapshot,
        )
        migrated_legacy = repo.threads.find_one({"thread_id": legacy_thread_id})
        rollback(settings, migration_snapshot)
        rolled_back_legacy = repo.threads.find_one({"thread_id": legacy_thread_id})
        migration_snapshot.unlink(missing_ok=True)
        migration_rollback = bool(
            migration_result["changed_threads"] == 1
            and migrated_legacy
            and migrated_legacy["messages"][0].get("turn_seq") == 1
            and rolled_back_legacy
            and "context_version" not in rolled_back_legacy
            and "turn_seq" not in rolled_back_legacy["messages"][0]
        )
        repo.threads.delete_one({"thread_id": legacy_thread_id})

        thread = repo.create_thread(
            ThreadRecord(title="T9-4.3.1 verifier", actor_scope=ActorScope(user_id=owner))
        )
        other = repo.create_thread(
            ThreadRecord(title="T9-4.3.1 isolation", actor_scope=ActorScope(user_id=owner))
        )
        thread_ids.extend((thread.thread_id, other.thread_id))

        for index in range(14):
            content = f"T9-4.3.1 persistent question {index + 1}"
            request_id = f"req_t9431_verify_{index:03d}"
            request_ids.append(request_id)
            record, replayed = repo.prepare_message_request(_request(thread, request_id, content))
            assert not replayed
            repo.append_message_once(
                thread.thread_id,
                ChatMessage(
                    message_id=record.user_message_id,
                    request_id=record.request_id,
                    run_id=record.run_id,
                    turn_seq=record.user_turn_seq,
                    role="user",
                    content=content,
                ),
            )
            answer = ChatMessage(
                request_id=record.request_id,
                run_id=record.run_id,
                role="assistant",
                content=f"T9-4.3.1 persistent answer {index + 1}",
            )
            current = repo.get_thread(thread.thread_id)
            assert current is not None
            compaction = ContextAssembler(settings).compact_thread(
                ThreadRecord.model_validate(
                    {
                        **current.model_dump(mode="python"),
                        "messages": [
                            *[item.model_dump(mode="python") for item in current.messages],
                            answer.model_dump(mode="python"),
                        ],
                    }
                )
            )
            persisted = repo.finalize_stream_response(
                thread.thread_id,
                answer,
                status="completed",
                summary=compaction.summary,
                summary_upto_message_id=compaction.summary_upto_message_id,
                active_context=current.active_context,
                context_version=current.context_version,
                pending_fields=[],
                clarification_round=0,
            )
            record.assistant_turn_seq = persisted.last_turn_seq
            repo.mark_message_request_terminal(
                record,
                AgentMessageRequestStatus.COMPLETED,
                assistant_message_id=answer.message_id,
            )

        final_thread = repo.get_thread(thread.thread_id)
        assert final_thread is not None
        context = ContextAssembler(settings).assemble(final_thread)
        exact_recent = len(context.recent_messages) == settings.agent_context_recent_turns * 2
        summary_boundary = bool(context.summary and context.summary_upto_message_id)
        sequences = [int(message.turn_seq or 0) for message in final_thread.messages]
        monotonic = sequences == list(range(1, len(sequences) + 1))

        restarted = MongoConversationRepository(settings)
        recovered = restarted.get_thread(thread.thread_id)
        restart_recovery = bool(
            recovered
            and len(recovered.messages) == len(final_thread.messages)
            and recovered.messages[-2].content == "T9-4.3.1 persistent question 14"
        )
        recovered_other = restarted.get_thread(other.thread_id)
        isolation = bool(recovered_other and recovered_other.messages == [])

        busy_request = _request(thread, "req_t9431_busy_001", "first active request")
        request_ids.append(busy_request.request_id)
        prepared, _ = restarted.prepare_message_request(busy_request)
        try:
            restarted.prepare_message_request(
                _request(thread, "req_t9431_busy_002", "second active request")
            )
        except ThreadBusyError:
            same_thread_ordering = True
        else:
            same_thread_ordering = False
        restarted.mark_message_request_terminal(
            prepared,
            AgentMessageRequestStatus.CANCELLED,
            error_code=AgentStreamErrorCode.CANCELLED,
        )

        stale_request = _request(thread, "req_t9431_stale_001", "stale retry request")
        request_ids.append(stale_request.request_id)
        stale_prepared, _ = restarted.prepare_message_request(stale_request)
        stale_time = datetime.now(UTC) - timedelta(seconds=settings.agent_thread_lease_seconds + 60)
        restarted.message_requests.update_one(
            {
                "thread_id": thread.thread_id,
                "request_id": stale_prepared.request_id,
            },
            {"$set": {"updated_at": stale_time}},
        )
        restarted.threads.update_one(
            {"thread_id": thread.thread_id},
            {"$set": {"active_request_started_at": stale_time}},
        )
        recovered_request, recovered_replayed = restarted.prepare_message_request(
            _request(thread, stale_prepared.request_id, "stale retry request")
        )
        same_request_stale_recovery = bool(
            not recovered_replayed
            and recovered_request.attempt == 2
            and recovered_request.user_turn_seq == stale_prepared.user_turn_seq
        )
        restarted.mark_message_request_terminal(
            recovered_request,
            AgentMessageRequestStatus.CANCELLED,
            error_code=AgentStreamErrorCode.CANCELLED,
        )
        verification = VerificationResult(
            database=settings.mongodb_database,
            exact_recent_messages=exact_recent,
            older_summary_boundary=summary_boundary,
            restart_recovery=restart_recovery,
            thread_isolation=isolation,
            same_thread_ordering=same_thread_ordering,
            same_request_stale_recovery=same_request_stale_recovery,
            monotonic_sequences=monotonic,
            migration_rollback=migration_rollback,
            cleaned_up=False,
        )
        if not all(
            value
            for key, value in asdict(verification).items()
            if key not in {"database", "cleaned_up"}
        ):
            raise RuntimeError(f"T9-4.3.1 verification failed: {asdict(verification)}")
    finally:
        repo.message_requests.delete_many({"thread_id": {"$in": thread_ids}})
        repo.threads.delete_many({"thread_id": {"$in": thread_ids}})
        cleaned_up = (
            repo.message_requests.count_documents({"thread_id": {"$in": thread_ids}}) == 0
            and repo.threads.count_documents({"thread_id": {"$in": thread_ids}}) == 0
        )
        repo.client.close()
    if verification is None:
        raise RuntimeError("T9-4.3.1 verification did not reach its assertions.")
    print(
        json.dumps(
            {**asdict(verification), "cleaned_up": cleaned_up},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
