"""Run the credential-safe T9-4.9 final Demo flow against the production API."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from scripts.verify_t946_capacity import SSEDecoder
from semikb.bootstrap import ApplicationContainer
from semikb.config import Settings


@dataclass(slots=True)
class StreamResult:
    case_id: str
    route: str | None
    interaction_mode: str | None
    status_code: int
    accepted_ms: float | None
    first_delta_ms: float | None
    total_ms: float
    event_names: list[str]
    answer_delta_count: int
    evidence_count: int
    image_count: int
    task_statuses: list[str]
    trace_id: str | None
    result: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://web:8080")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def access_key_headers() -> dict[str, str]:
    access_key = os.environ.get("DEMO_ACCESS_KEY", "")
    if not access_key:
        raise RuntimeError("DEMO_ACCESS_KEY must be available through the process environment")
    return {"X-Demo-Access-Key": access_key}


def create_token(client: httpx.Client) -> str:
    response: httpx.Response | None = None
    for attempt in range(6):
        response = client.post("/api/v1/auth/demo-token", headers=access_key_headers())
        if response.status_code != 429:
            response.raise_for_status()
            return str(response.json()["access_token"])
        if attempt < 5:
            time.sleep(min(12 * (attempt + 1), 30))
    assert response is not None
    response.raise_for_status()
    raise AssertionError("unreachable")


def create_thread(client: httpx.Client, headers: dict[str, str], title: str) -> str:
    response = client.post("/api/v1/threads", headers=headers, json={"title": title})
    response.raise_for_status()
    return str(response.json()["thread_id"])


def stream_message(
    client: httpx.Client,
    headers: dict[str, str],
    thread_id: str,
    case_id: str,
    content: str,
    timeout: int,
) -> StreamResult:
    started = time.perf_counter()
    accepted_ms: float | None = None
    first_delta_ms: float | None = None
    decoder = SSEDecoder()
    events: list[dict[str, Any]] = []
    completed: dict[str, Any] | None = None

    def consume(event: dict[str, Any]) -> None:
        nonlocal accepted_ms, first_delta_ms, completed
        events.append(event)
        elapsed_ms = (time.perf_counter() - started) * 1000
        event_name = event.get("event")
        if event_name == "accepted" and accepted_ms is None:
            accepted_ms = elapsed_ms
        elif event_name == "answer_delta" and first_delta_ms is None:
            first_delta_ms = elapsed_ms
        elif event_name == "completed":
            completed = event.get("data", {}).get("result")
        elif event_name == "error":
            code = event.get("data", {}).get("code", "stream_error")
            raise AssertionError(f"{case_id} returned SSE error {code}")

    request_id = f"req_t949_{case_id}_{uuid.uuid4().hex}"
    with client.stream(
        "POST",
        f"/api/v1/threads/{thread_id}/messages/stream",
        headers={**headers, "Accept": "text/event-stream"},
        json={"content": content, "request_id": request_id},
        timeout=timeout,
    ) as response:
        status_code = response.status_code
        response.raise_for_status()
        for chunk in response.iter_text():
            for event in decoder.feed(chunk):
                consume(event)
        for event in decoder.feed("", final=True):
            consume(event)
    if completed is None:
        raise AssertionError(f"{case_id} did not emit a completed result")
    evidence = completed.get("evidence") or completed.get("evidence_ledger") or []
    return StreamResult(
        case_id=case_id,
        route=completed.get("route_decision"),
        interaction_mode=completed.get("interaction_mode"),
        status_code=status_code,
        accepted_ms=accepted_ms,
        first_delta_ms=first_delta_ms,
        total_ms=(time.perf_counter() - started) * 1000,
        event_names=[str(event.get("event")) for event in events],
        answer_delta_count=sum(event.get("event") == "answer_delta" for event in events),
        evidence_count=len(evidence),
        image_count=len(completed.get("image_asset_ids") or []),
        task_statuses=[str(item.get("status")) for item in completed.get("task_results", [])],
        trace_id=completed.get("trace_id"),
        result=completed,
    )


def latest_assistant_presentation(result: dict[str, Any]) -> dict[str, Any]:
    messages = result.get("thread", {}).get("messages", [])
    assistant = next(
        (message for message in reversed(messages) if message.get("role") == "assistant"),
        {},
    )
    return dict(assistant.get("presentation") or {})


def safe_stream_result(result: StreamResult) -> dict[str, Any]:
    return {
        key: value
        for key, value in asdict(result).items()
        if key != "result"
    }


def cleanup_threads(thread_ids: list[str]) -> bool:
    container = ApplicationContainer(Settings(demo_mode=False))
    database = container.conversation_store.database
    for thread_id in thread_ids:
        container.conversations.checkpointer.delete_thread(thread_id)
        for collection in (
            "agent_threads",
            "agent_message_requests",
            "audit_events",
            "retrieval_traces",
        ):
            database[collection].delete_many({"thread_id": thread_id})
    return all(
        database["agent_threads"].count_documents({"thread_id": thread_id}) == 0
        and database["agent_message_requests"].count_documents({"thread_id": thread_id}) == 0
        for thread_id in thread_ids
    )


def verify(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    results: list[StreamResult] = []
    thread_ids: list[str] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    cleanup_verified = False
    try:
        with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
            health = client.get("/api/v1/health")
            check("public_health", health.status_code == 200, health.status_code)
            token = create_token(client)
            headers = {"Authorization": f"Bearer {token}"}

            conversation = create_thread(client, headers, "T9-4.9 final conversation")
            thread_ids.append(conversation)
            capability = stream_message(
                client,
                headers,
                conversation,
                "capability",
                "你好，你能做什么？",
                args.timeout,
            )
            results.append(capability)
            check(
                "natural_chat_without_retrieval",
                capability.route == "chat_direct"
                and capability.trace_id is None
                and capability.evidence_count == 0,
                safe_stream_result(capability),
            )

            sop = stream_message(
                client,
                headers,
                conversation,
                "current_sop",
                "ETCH-03 Chamber B 清腔后首片异常，当前 SOP 怎么要求？",
                args.timeout,
            )
            results.append(sop)
            check(
                "controlled_rag_with_trace",
                sop.route == "internal_rag"
                and sop.trace_id is not None
                and sop.evidence_count > 0
                and latest_assistant_presentation(sop.result).get("mode") == "structured",
                safe_stream_result(sop),
            )
            trace = client.get(f"/api/v1/retrieval-traces/{sop.trace_id}", headers=headers)
            trace_payload = trace.json() if trace.status_code == 200 else {}
            check(
                "trace_reproduces_final_evidence",
                trace.status_code == 200
                and bool(trace_payload.get("candidates"))
                and bool(trace_payload.get("final_evidence_ids")),
                {
                    "status": trace.status_code,
                    "candidate_count": len(trace_payload.get("candidates") or []),
                    "final_evidence_count": len(trace_payload.get("final_evidence_ids") or []),
                    "routes": trace_payload.get("routes") or [],
                },
            )

            reuse = stream_message(
                client,
                headers,
                conversation,
                "reuse_evidence",
                "其中哪条证据提到了 RF match？",
                args.timeout,
            )
            results.append(reuse)
            check(
                "evidence_reuse",
                reuse.route == "reuse_evidence" and reuse.evidence_count > 0,
                safe_stream_result(reuse),
            )

            history = stream_message(
                client,
                headers,
                conversation,
                "history_recall",
                "我刚才问了什么？",
                args.timeout,
            )
            results.append(history)
            check(
                "history_direct_without_retrieval",
                history.route == "chat_direct"
                and history.trace_id is None
                and history.evidence_count == 0
                and latest_assistant_presentation(history.result).get("mode") == "bubble",
                safe_stream_result(history),
            )

            clarification = create_thread(client, headers, "T9-4.9 clarification")
            thread_ids.append(clarification)
            clarify = stream_message(
                client,
                headers,
                clarification,
                "clarify",
                "帮我查一下最近的良率异常。",
                args.timeout,
            )
            results.append(clarify)
            check(
                "missing_information_clarifies_without_retrieval",
                clarify.route == "clarify"
                and clarify.trace_id is None
                and clarify.evidence_count == 0,
                safe_stream_result(clarify),
            )
            resume = stream_message(
                client,
                headers,
                clarification,
                "clarify_resume",
                "P-ALPHA，ETCH-03 Chamber B，最近24小时。",
                args.timeout,
            )
            results.append(resume)
            check(
                "clarification_resumes_original_task",
                resume.route in {"tool_only", "internal_rag", "rag_and_tool"}
                and resume.route != "clarify"
                and bool(resume.task_statuses)
                and all(status == "completed" for status in resume.task_statuses),
                safe_stream_result(resume),
            )

            image_thread = create_thread(client, headers, "T9-4.9 image")
            thread_ids.append(image_thread)
            image = stream_message(
                client,
                headers,
                image_thread,
                "image_retrieval",
                "找一张边缘环状缺陷的晶圆图，并说明应该先核对哪些证据。",
                args.timeout,
            )
            results.append(image)
            image_ids = image.result.get("image_asset_ids") or []
            asset_status = None
            if image_ids:
                asset = client.get(f"/api/v1/assets/{image_ids[0]}/access", headers=headers)
                asset_status = asset.status_code
            check(
                "authorized_image_auto_preview_contract",
                image.route == "internal_rag"
                and image.image_count > 0
                and latest_assistant_presentation(image.result).get("image_asset_ids") == image_ids
                and asset_status == 200,
                {**safe_stream_result(image), "asset_access_status": asset_status},
            )

            safety = create_thread(client, headers, "T9-4.9 controlled tasks")
            thread_ids.append(safety)
            mixed = stream_message(
                client,
                headers,
                safety,
                "partial_success",
                "查 P-ALPHA ETCH-03 最近24小时 FDC 报警、修改 Recipe、生成报告",
                args.timeout,
            )
            results.append(mixed)
            check(
                "controlled_multi_task_does_not_lose_or_execute_unsafe_work",
                mixed.task_statuses == ["completed", "refused", "deferred"],
                safe_stream_result(mixed),
            )

            public_thread = create_thread(client, headers, "T9-4.9 public corpus")
            thread_ids.append(public_thread)
            public = stream_message(
                client,
                headers,
                public_thread,
                "governed_public_corpus",
                "请查询内部知识库：UCI SECOM 数据集包含什么类型的样本、特征和标签？请引用已入库的数据集说明，不要使用 Web。",
                args.timeout,
            )
            results.append(public)
            public_trace = {}
            if public.trace_id:
                response = client.get(
                    f"/api/v1/retrieval-traces/{public.trace_id}",
                    headers=headers,
                )
                if response.status_code == 200:
                    public_trace = response.json()
            routes = public_trace.get("routes") or []
            check(
                "governed_public_corpus_stays_internal",
                public.route == "internal_rag"
                and public.evidence_count > 0
                and not any("web" in str(route).lower() for route in routes),
                {**safe_stream_result(public), "trace_routes": routes},
            )

            for item in results:
                check(
                    f"sse_contract_{item.case_id}",
                    item.status_code == 200
                    and item.accepted_ms is not None
                    and item.accepted_ms <= 1000
                    and item.answer_delta_count > 0
                    and item.event_names[-1] == "completed",
                    {
                        "accepted_ms": item.accepted_ms,
                        "first_delta_ms": item.first_delta_ms,
                        "total_ms": item.total_ms,
                        "event_names": item.event_names,
                        "answer_delta_count": item.answer_delta_count,
                    },
                )

            operations: dict[str, Any] = {}
            for name, path in (
                ("ingestion_jobs", "/api/v1/ingestion-jobs"),
                ("knowledge_documents", "/api/v1/knowledge-documents?limit=10"),
                ("evaluation_runs", "/api/v1/evaluation-runs"),
            ):
                response = client.get(path, headers=headers)
                payload = response.json() if response.status_code == 200 else []
                count = len(payload.get("items", [])) if isinstance(payload, dict) else len(payload)
                operations[name] = {"status": response.status_code, "count": count}
            check(
                "operator_surfaces_have_persisted_data",
                all(item["status"] == 200 and item["count"] > 0 for item in operations.values()),
                operations,
            )
    finally:
        if thread_ids:
            cleanup_verified = cleanup_threads(thread_ids)

    check(
        "temporary_acceptance_threads_cleaned",
        cleanup_verified,
        {"thread_count": len(thread_ids)},
    )
    failures = [item["name"] for item in checks if not item["passed"]]
    return {
        "schema": "semikb-t949-final-demo-v1",
        "captured_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "case_count": len(results),
        "checks": checks,
        "passed": not failures,
        "failed_checks": failures,
    }


def main() -> None:
    args = parse_args()
    report = verify(args)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"wrote credential-safe final Demo report to {args.output}")
    else:
        print(serialized, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
