from __future__ import annotations

from copy import deepcopy

import pytest

from semikb.storage.t9431_context_migration import (
    ContextMigrationSafetyError,
    _backfilled_document,
    _thread_snapshot,
)


def test_legacy_thread_backfill_is_deterministic_and_idempotent() -> None:
    legacy = {
        "thread_id": "thread_legacy",
        "messages": [
            {"message_id": "m1", "role": "user", "content": "one"},
            {"message_id": "m2", "role": "assistant", "content": "two"},
        ],
    }

    migrated, assigned = _backfilled_document(legacy)
    second, second_assigned = _backfilled_document(migrated)

    assert assigned == 2
    assert second_assigned == 0
    assert migrated == second
    assert [message["turn_seq"] for message in migrated["messages"]] == [1, 2]
    assert migrated["last_turn_seq"] == 2
    assert migrated["next_turn_seq"] == 3
    assert migrated["context_version"] == 1


def test_migration_snapshot_contains_only_context_fields_and_message_ids() -> None:
    document = {
        "thread_id": "thread_private",
        "title": "sensitive title",
        "messages": [
            {"message_id": "m1", "role": "user", "content": "sensitive body"},
        ],
    }

    snapshot = _thread_snapshot(document)

    assert "sensitive" not in repr(snapshot)
    assert snapshot["message_turn_seq"] == [
        {"message_id": "m1", "exists": False, "value": None}
    ]


def test_migration_accepts_aligned_partial_sequence_and_refuses_invalid_threads() -> None:
    partial = {
        "thread_id": "thread_partial",
        "messages": [
            {"message_id": "m1", "turn_seq": 1},
            {"message_id": "m2"},
        ],
    }
    migrated, assigned = _backfilled_document(partial)
    assert assigned == 1
    assert [message["turn_seq"] for message in migrated["messages"]] == [1, 2]

    duplicate = {
        "thread_id": "thread_duplicate",
        "messages": [
            {"message_id": "m1", "turn_seq": 1},
            {"message_id": "m2", "turn_seq": 1},
        ],
    }

    with pytest.raises(ContextMigrationSafetyError):
        _backfilled_document(duplicate)
    assert partial == deepcopy(partial)
