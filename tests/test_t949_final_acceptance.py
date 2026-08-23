from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_t949_acceptance import summarize
from scripts.verify_t949_final_demo import StreamResult, safe_stream_result, stream_message


def _snapshot() -> dict[str, object]:
    return {
        "mongodb": {"documents": {"count": 1, "content_sha256": "a"}},
        "milvus": {"row_count": 1, "metadata_sha256": "b"},
        "minio": {"semikb-raw": {"object_count": 1}},
        "redis": {"ping": True, "celery_queue_depth": 0},
    }


def test_safe_stream_result_excludes_answer_and_internal_result() -> None:
    result = StreamResult(
        case_id="chat",
        route="chat_direct",
        interaction_mode="direct",
        status_code=200,
        accepted_ms=10,
        first_delta_ms=20,
        total_ms=30,
        event_names=["accepted", "answer_delta", "completed"],
        answer_delta_count=1,
        evidence_count=0,
        image_count=0,
        task_statuses=[],
        trace_id=None,
        result={"response": "must not be persisted in the report"},
    )

    safe = safe_stream_result(result)

    assert "result" not in safe
    assert "must not be persisted" not in json.dumps(safe)


def test_t949_summary_requires_every_component_and_storage_invariant(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for name in ("final-demo.json", "security.json", "offline.json"):
        (evidence / name).write_text('{"passed": true}\n', encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_snapshot()), encoding="utf-8")
    current = {**_snapshot(), "retrieval_smoke": {"passed": True}}
    (evidence / "storage-state.json").write_text(json.dumps(current), encoding="utf-8")
    (evidence / "worker-ping.txt").write_text("pong\n", encoding="utf-8")
    (evidence / "health.json").write_text('{"status": "ok"}\n', encoding="utf-8")

    report = summarize(evidence, baseline)

    assert report["passed"] is True
    assert report["storage_comparison"] == {"matched": True, "differences": {}}


def test_t949_summary_fails_when_storage_changes(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for name in ("final-demo.json", "security.json", "offline.json"):
        (evidence / name).write_text('{"passed": true}\n', encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_snapshot()), encoding="utf-8")
    current = {**_snapshot(), "retrieval_smoke": {"passed": True}}
    current["milvus"] = {"row_count": 2, "metadata_sha256": "changed"}
    (evidence / "storage-state.json").write_text(json.dumps(current), encoding="utf-8")
    (evidence / "worker-ping.txt").write_text("pong\n", encoding="utf-8")
    (evidence / "health.json").write_text('{"status": "ok"}\n', encoding="utf-8")

    report = summarize(evidence, baseline)

    assert report["passed"] is False
    assert report["failed_checks"] == ["storage_invariants"]


def test_t949_shell_entrypoint_is_guarded_and_non_destructive() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/deployment/run_t949_final.sh").read_text(encoding="utf-8")

    assert "T949_ACCEPTANCE_CONFIRM" in script
    assert "--apply" in script
    assert "Refusing to overwrite non-empty evidence directory" in script
    assert "python -m scripts.verify_t949_final_demo" in script
    assert "--base-url http://web:8080" in script
    assert "python3 -m scripts.summarize_t949_acceptance" in script
    assert "verify_t948_security.py" in script
    assert "verify_t948_offline_bundle.py" in script
    assert "verify_t947_restore.py" in script
    assert "docker compose down" not in script
    assert "docker system prune" not in script
    assert "docker volume rm" not in script


def test_t949_summary_has_no_project_runtime_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/summarize_t949_acceptance.py").read_text(encoding="utf-8")

    assert "verify_t947_restore import" not in source
    assert "from semikb" not in source


def test_stream_message_accepts_completed_event_without_trailing_blank_line() -> None:
    class FakeResponse:
        status_code = 200

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_text(self) -> list[str]:
            return [
                'data: {"event":"accepted","data":{"request_id":"request-1"}}\n\n',
                'data: {"event":"answer_delta","data":{"delta":"ok"}}\n\n',
                'data: {"event":"completed","data":{"result":{"route_decision":"chat_direct"}}}',
            ]

    class FakeClient:
        def stream(self, *args: object, **kwargs: object) -> FakeResponse:
            return FakeResponse()

    result = stream_message(
        FakeClient(),  # type: ignore[arg-type]
        {},
        "thread-1",
        "tail-event",
        "hello",
        10,
    )

    assert result.route == "chat_direct"
    assert result.event_names[-1] == "completed"
