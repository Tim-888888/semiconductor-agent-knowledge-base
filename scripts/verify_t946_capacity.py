"""Run the bounded T9-4.6 Demo workload against the production HTTP API."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

MAX_CONCURRENCY = 3
CAPACITY_DOCUMENT_ID = "T946-CAPACITY-BASELINE"
QUESTIONS = (
    ("sop", "ETCH-03 Chamber B 清腔后首片异常，当前 SOP 怎么要求？"),
    ("rag_tool", "查询 P-ALPHA 最近24小时 ETCH-03 Chamber B FDC 报警，并对照 SOP 给排查建议。"),
    ("image", "有没有 ETCH-03 Chamber B 的边缘环状缺陷晶圆图？"),
)


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


class SSEDecoder:
    """Incrementally decode complete SSE data blocks across arbitrary network splits."""

    def __init__(self) -> None:
        self.buffer = ""

    def feed(self, chunk: str, *, final: bool = False) -> list[dict[str, Any]]:
        self.buffer += chunk.replace("\r\n", "\n")
        events: list[dict[str, Any]] = []
        while "\n\n" in self.buffer:
            block, self.buffer = self.buffer.split("\n\n", 1)
            event = self._decode_block(block)
            if event is not None:
                events.append(event)
        if final and self.buffer.strip():
            event = self._decode_block(self.buffer)
            self.buffer = ""
            if event is not None:
                events.append(event)
        return events

    @staticmethod
    def _decode_block(block: str) -> dict[str, Any] | None:
        data = "\n".join(
            line[5:].lstrip()
            for line in block.splitlines()
            if line.startswith("data:")
        )
        return json.loads(data) if data else None


@dataclass(slots=True)
class StreamProbe:
    label: str
    level: int
    wave: int
    request_id: str
    thread_id: str
    status_code: int
    accepted_ms: float | None = None
    first_delta_ms: float | None = None
    total_ms: float | None = None
    route: str | None = None
    trace_id: str | None = None
    retrieval_ms: float | None = None
    provider_attempts: int = 0
    event_count: int = 0
    answer_delta_count: int = 0
    error_code: str | None = None


def parse_levels(raw: str) -> list[int]:
    try:
        levels = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("levels must be comma-separated integers") from exc
    if not levels or levels[0] < 1 or levels[-1] > MAX_CONCURRENCY:
        raise argparse.ArgumentTypeError("levels must stay within the bounded range 1..3")
    return levels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://web")
    parser.add_argument("--levels", type=parse_levels, default=parse_levels("1,2,3"))
    parser.add_argument("--waves", type=int, default=1, choices=range(1, 4))
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--background-timeout", type=int, default=1800)
    parser.add_argument("--dataset-version", default="demo-v2")
    parser.add_argument("--skip-background", action="store_true")
    parser.add_argument("--skip-disconnect", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def access_key_headers() -> dict[str, str]:
    access_key = os.environ.get("DEMO_ACCESS_KEY", "")
    if not access_key:
        raise RuntimeError("DEMO_ACCESS_KEY must be available through the process environment")
    return {"X-Demo-Access-Key": access_key}


async def create_token(client: httpx.AsyncClient, user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "roles": ["engineer", "knowledge_admin"],
        "access_scope_keys": ["demo_engineering"],
        "fabs": ["FAB-01"],
        "products": ["P-ALPHA"],
        "tool_ids": ["ETCH-03"],
    }
    response: httpx.Response | None = None
    for attempt in range(6):
        response = await client.post(
            "/api/v1/auth/demo-token",
            headers=access_key_headers(),
            json=payload,
        )
        if response.status_code != 429:
            response.raise_for_status()
            return str(response.json()["access_token"])
        if attempt < 5:
            await asyncio.sleep(min(12 * (attempt + 1), 30))
    assert response is not None
    response.raise_for_status()
    raise AssertionError("unreachable")


async def create_thread(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    title: str,
) -> str:
    response = await client.post("/api/v1/threads", headers=headers, json={"title": title})
    response.raise_for_status()
    return str(response.json()["thread_id"])


def provider_attempt_count(result: dict[str, Any], trace: dict[str, Any] | None) -> int:
    metadata = result.get("model_metadata", {})
    return (
        len(metadata.get("understanding_provider_attempts", []))
        + len(metadata.get("answer_provider_attempts", []))
        + len((trace or {}).get("provider_attempts", []))
    )


async def run_stream_probe(
    client: httpx.AsyncClient,
    *,
    token: str,
    level: int,
    wave: int,
    index: int,
    timeout: int,
) -> StreamProbe:
    label, question = QUESTIONS[index]
    headers = {"Authorization": f"Bearer {token}", "Accept": "text/event-stream"}
    request_id = f"req_t946_l{level}_w{wave}_{index}_{int(time.time() * 1000)}"
    started = time.perf_counter()
    probe = StreamProbe(
        label=label,
        level=level,
        wave=wave,
        request_id=request_id,
        thread_id="",
        status_code=0,
    )
    decoder = SSEDecoder()
    completed_result: dict[str, Any] | None = None
    try:
        thread_id = await create_thread(client, headers, f"T9-4.6 L{level} W{wave} {label}")
        probe.thread_id = thread_id
        async with client.stream(
            "POST",
            f"/api/v1/threads/{thread_id}/messages/stream",
            headers=headers,
            json={"content": question, "request_id": request_id},
            timeout=timeout,
        ) as response:
            probe.status_code = response.status_code
            response.raise_for_status()
            async for chunk in response.aiter_text():
                for event in decoder.feed(chunk):
                    probe.event_count += 1
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    event_name = event.get("event")
                    if event_name == "accepted" and probe.accepted_ms is None:
                        probe.accepted_ms = elapsed_ms
                    elif event_name == "answer_delta":
                        probe.answer_delta_count += 1
                        if probe.first_delta_ms is None:
                            probe.first_delta_ms = elapsed_ms
                    elif event_name == "completed":
                        completed_result = event.get("data", {}).get("result", {})
                    elif event_name == "error":
                        probe.error_code = str(event.get("data", {}).get("code", "stream_error"))
            for event in decoder.feed("", final=True):
                probe.event_count += 1
                if event.get("event") == "completed":
                    completed_result = event.get("data", {}).get("result", {})
        probe.total_ms = (time.perf_counter() - started) * 1000
        if completed_result is None and probe.error_code is None:
            probe.error_code = "missing_completed_event"
        if completed_result:
            probe.route = completed_result.get("route_decision")
            probe.trace_id = completed_result.get("trace_id")
            trace = None
            if probe.trace_id:
                trace_response = await client.get(
                    f"/api/v1/retrieval-traces/{probe.trace_id}",
                    headers=headers,
                )
                trace_response.raise_for_status()
                trace = trace_response.json()
                probe.retrieval_ms = sum(float(value) for value in trace.get("timings_ms", {}).values())
            probe.provider_attempts = provider_attempt_count(completed_result, trace)
    except httpx.TimeoutException:
        probe.total_ms = (time.perf_counter() - started) * 1000
        probe.error_code = "timeout"
    except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
        probe.total_ms = (time.perf_counter() - started) * 1000
        probe.error_code = type(exc).__name__
    return probe


def ingestion_metadata() -> dict[str, Any]:
    return {
        "document_id": CAPACITY_DOCUMENT_ID,
        "revision": "R1",
        "title": "T9-4.6 bounded capacity acceptance note",
        "document_type": "test_report",
        "approval_status": "approved",
        "lifecycle": "published",
        "source_kind": "synthetic_acceptance",
        "source_uri": "synthetic://t9-4.6/bounded-capacity",
        "source_license": "CC0-1.0",
        "source_id": "semikb.demo.synthetic",
        "source_manifest_version": "1.0.0",
        "dataset_version": "demo-v2",
        "source_license_status": "verified",
        "redistribution_policy": "allowed",
        "access_scope_key": "demo_engineering",
        "fab": "FAB-01",
        "product": "P-ALPHA",
        "process_layer": "ETCH",
        "tool_id": "ETCH-03",
        "chamber": "B",
        "retrieval_policy": "protected",
    }


async def wait_terminal(
    client: httpx.AsyncClient,
    path: str,
    headers: dict[str, str],
    terminal: set[str],
    timeout: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = await client.get(path, headers=headers)
        response.raise_for_status()
        latest = response.json()
        if latest.get("status") in terminal:
            return latest
        await asyncio.sleep(2)
    raise TimeoutError(f"resource did not reach a terminal state: {path} ({latest.get('status')})")


async def run_background_work(
    client: httpx.AsyncClient,
    token: str,
    timeout: int,
    dataset_version: str,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"}
    document = (
        b"# Bounded capacity acceptance\n\n"
        b"This synthetic note exists only to exercise the general ingestion queue.\n\n"
        b"For ETCH-03 Chamber B, verify pressure and RF match before releasing a held lot.\n"
    )

    upload_task = client.post(
        "/api/v1/ingestion-jobs/upload",
        headers=headers,
        files={"file": ("t946-capacity.md", document, "text/markdown")},
        data={"metadata": json.dumps(ingestion_metadata())},
    )
    evaluation_task = client.post(
        "/api/v1/evaluation-runs",
        headers=headers,
        json={"dataset_version": dataset_version, "retrieval_profile": "full"},
    )
    upload_response, evaluation_response = await asyncio.gather(upload_task, evaluation_task)
    upload_response.raise_for_status()
    evaluation_response.raise_for_status()
    ingestion = upload_response.json()
    evaluation = evaluation_response.json()
    started = time.perf_counter()
    ingestion_wait = wait_terminal(
        client,
        f"/api/v1/ingestion-jobs/{ingestion['job_id']}",
        headers,
        {"published", "failed"},
        timeout,
    )
    evaluation_wait = wait_terminal(
        client,
        f"/api/v1/evaluation-runs/{evaluation['evaluation_run_id']}",
        headers,
        {"completed", "failed"},
        timeout,
    )
    final_ingestion, final_evaluation = await asyncio.gather(ingestion_wait, evaluation_wait)
    elapsed_ms = (time.perf_counter() - started) * 1000
    evaluation_provider_attempts = 0
    for case in final_evaluation.get("case_results", []):
        if not case.get("trace_id"):
            continue
        trace_response = await client.get(
            f"/api/v1/evaluation-runs/{final_evaluation['evaluation_run_id']}"
            f"/cases/{case['case_id']}/trace",
            headers=headers,
        )
        if trace_response.status_code == 200:
            evaluation_provider_attempts += len(trace_response.json().get("provider_attempts", []))
    return {
        "elapsed_ms": elapsed_ms,
        "ingestion": {
            "job_id": final_ingestion.get("job_id"),
            "status": final_ingestion.get("status"),
            "attempt": final_ingestion.get("attempt"),
            "chunks_count": final_ingestion.get("chunks_count"),
            "provider_attempts": len(final_ingestion.get("provider_attempts", [])),
        },
        "evaluation": {
            "evaluation_run_id": final_evaluation.get("evaluation_run_id"),
            "status": final_evaluation.get("status"),
            "case_count": final_evaluation.get("case_count"),
            "provider_attempts": evaluation_provider_attempts,
        },
    }


async def run_disconnect_probe(
    client: httpx.AsyncClient,
    token: str,
    timeout: int,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}", "Accept": "text/event-stream"}
    thread_id = await create_thread(client, headers, "T9-4.6 disconnect recovery")
    request_id = f"req_t946_disconnect_{int(time.time() * 1000)}"
    decoder = SSEDecoder()
    seen_events: list[str] = []
    async with client.stream(
        "POST",
        f"/api/v1/threads/{thread_id}/messages/stream",
        headers=headers,
        json={"content": QUESTIONS[1][1], "request_id": request_id},
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_text():
            for event in decoder.feed(chunk):
                seen_events.append(str(event.get("event")))
            if "accepted" in seen_events and "stage" in seen_events:
                break

    deadline = time.monotonic() + min(max(timeout, 30), 180)
    request_status = "unknown"
    while time.monotonic() < deadline:
        status_response = await client.get(
            f"/api/v1/threads/{thread_id}/message-requests/{request_id}",
            headers=headers,
        )
        status_response.raise_for_status()
        request_status = str(status_response.json().get("status"))
        if request_status in {"cancelled", "failed", "completed"}:
            break
        await asyncio.sleep(0.5)

    if request_status not in {"cancelled", "failed", "completed"}:
        return {
            "thread_id": thread_id,
            "disconnected_request_status": request_status,
            "events_before_disconnect": seen_events,
            "recovery_status": "not_attempted",
            "recovery_route": None,
            "passed": False,
        }

    recovery = await run_existing_thread_probe(
        client,
        headers=headers,
        thread_id=thread_id,
        content="你好，请简要说明你能提供哪些半导体知识协助。",
        timeout=timeout,
    )
    return {
        "thread_id": thread_id,
        "disconnected_request_status": request_status,
        "events_before_disconnect": seen_events,
        "recovery_status": recovery["status"],
        "recovery_route": recovery.get("route"),
        "passed": request_status in {"cancelled", "failed", "completed"}
        and recovery["status"] == "completed",
    }


async def run_existing_thread_probe(
    client: httpx.AsyncClient,
    *,
    headers: dict[str, str],
    thread_id: str,
    content: str,
    timeout: int,
) -> dict[str, Any]:
    request_id = f"req_t946_recovery_{int(time.time() * 1000)}"
    decoder = SSEDecoder()
    result: dict[str, Any] | None = None
    for attempt in range(10):
        async with client.stream(
            "POST",
            f"/api/v1/threads/{thread_id}/messages/stream",
            headers=headers,
            json={"content": content, "request_id": request_id},
            timeout=timeout,
        ) as response:
            if response.status_code == 409 and attempt < 9:
                await response.aread()
                await asyncio.sleep(0.5)
                continue
            response.raise_for_status()
            async for chunk in response.aiter_text():
                for event in decoder.feed(chunk):
                    if event.get("event") == "completed":
                        result = event.get("data", {}).get("result", {})
                    elif event.get("event") == "error":
                        return {"status": "failed", "route": None}
            break
    return {
        "status": "completed" if result is not None else "failed",
        "route": (result or {}).get("route_decision"),
    }


def build_summary(probes: list[StreamProbe]) -> dict[str, Any]:
    accepted = [item.accepted_ms for item in probes if item.accepted_ms is not None]
    first_delta = [item.first_delta_ms for item in probes if item.first_delta_ms is not None]
    totals = [item.total_ms for item in probes if item.total_ms is not None]
    retrieval = [item.retrieval_ms for item in probes if item.retrieval_ms is not None]
    failures = [item for item in probes if item.error_code or item.status_code >= 500]
    return {
        "request_count": len(probes),
        "completed_count": len(probes) - len(failures),
        "error_rate": len(failures) / max(len(probes), 1),
        "timeout_count": sum(item.error_code == "timeout" for item in probes),
        "http_5xx_count": sum(item.status_code >= 500 for item in probes),
        "accepted_p50_ms": percentile(accepted, 0.50),
        "accepted_p95_ms": percentile(accepted, 0.95),
        "first_delta_p50_ms": percentile(first_delta, 0.50),
        "first_delta_p95_ms": percentile(first_delta, 0.95),
        "end_to_end_p50_ms": percentile(totals, 0.50),
        "end_to_end_p95_ms": percentile(totals, 0.95),
        "retrieval_p50_ms": percentile(retrieval, 0.50),
        "retrieval_p95_ms": percentile(retrieval, 0.95),
        "observed_provider_attempts": sum(item.provider_attempts for item in probes),
    }


async def verify(args: argparse.Namespace) -> dict[str, Any]:
    timeout = httpx.Timeout(args.request_timeout, connect=15)
    probes: list[StreamProbe] = []
    background_task: asyncio.Task[dict[str, Any]] | None = None
    async with httpx.AsyncClient(base_url=args.base_url, timeout=timeout) as client:
        live = await client.get("/api/v1/live")
        live.raise_for_status()
        token = await create_token(client, f"t946_capacity_{int(time.time() * 1000)}")
        for level in args.levels:
            for wave in range(1, args.waves + 1):
                if level == MAX_CONCURRENCY and wave == 1 and not args.skip_background:
                    background_task = asyncio.create_task(
                        run_background_work(
                            client,
                            token,
                            args.background_timeout,
                            args.dataset_version,
                        )
                    )
                wave_results = await asyncio.gather(
                    *(
                        run_stream_probe(
                            client,
                            token=token,
                            level=level,
                            wave=wave,
                            index=index,
                            timeout=args.request_timeout,
                        )
                        for index in range(level)
                    )
                )
                probes.extend(wave_results)
        background: dict[str, Any] | None = None
        if background_task:
            try:
                background = await background_task
            except Exception as exc:  # noqa: BLE001 - retain an acceptance artifact on failure
                background = {
                    "error_code": type(exc).__name__,
                    "detail": str(exc)[:500],
                }
        disconnect: dict[str, Any] | None = None
        if not args.skip_disconnect:
            try:
                disconnect = await run_disconnect_probe(client, token, args.request_timeout)
            except Exception as exc:  # noqa: BLE001 - retain an acceptance artifact on failure
                disconnect = {
                    "error_code": type(exc).__name__,
                    "detail": str(exc)[:500],
                    "passed": False,
                }

    summary = build_summary(probes)
    background_passed = background is None or (
        "error_code" not in background
        and background["ingestion"]["status"] == "published"
        and background["evaluation"]["status"] == "completed"
    )
    disconnect_passed = disconnect is None or bool(disconnect["passed"])
    passed = (
        summary["error_rate"] == 0
        and summary["timeout_count"] == 0
        and summary["http_5xx_count"] == 0
        and all(item.accepted_ms is not None and item.first_delta_ms is not None for item in probes)
        and background_passed
        and disconnect_passed
    )
    return {
        "verification": "T9-4.6-bounded-capacity-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "levels": args.levels,
        "waves": args.waves,
        "max_concurrency": MAX_CONCURRENCY,
        "worker_concurrency_expected": 1,
        "summary": summary,
        "probes": [asdict(item) for item in probes],
        "background": background,
        "disconnect_recovery": disconnect,
        "passed": passed,
    }


def main() -> None:
    args = parse_args()
    report = asyncio.run(verify(args))
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
        print(f"wrote credential-safe report to {output}")
    else:
        print(serialized, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
