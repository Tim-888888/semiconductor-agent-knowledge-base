from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from scripts import verify_t946_capacity as capacity
from scripts.summarize_t946_runtime import parse_size, summarize
from scripts.verify_t946_capacity import (
    SSEDecoder,
    StreamProbe,
    build_summary,
    ingestion_metadata,
    parse_levels,
    percentile,
)
from scripts.verify_t946_state import compare_snapshots


def test_percentile_interpolates_small_bounded_samples() -> None:
    assert percentile([], 0.95) is None
    assert percentile([10], 0.95) == 10
    assert percentile([10, 20, 30], 0.50) == 20
    assert percentile([10, 20, 30], 0.95) == pytest.approx(29)


def test_sse_decoder_handles_arbitrary_network_splits() -> None:
    decoder = SSEDecoder()
    first = 'data: {"event":"accepted","data":{"run_id":"run_1"}}\n\n'
    second = 'data: {"event":"answer_delta","data":{"delta":"ok"}}\n\n'

    assert decoder.feed(first[:13]) == []
    events = decoder.feed(first[13:] + second[:9])
    assert events == [{"event": "accepted", "data": {"run_id": "run_1"}}]
    assert decoder.feed(second[9:], final=True) == [
        {"event": "answer_delta", "data": {"delta": "ok"}}
    ]


def test_capacity_levels_are_hard_bounded() -> None:
    assert parse_levels("3,1,2,2") == [1, 2, 3]
    with pytest.raises(Exception, match="1..3"):
        parse_levels("1,4")


def test_capacity_ingestion_uses_complete_published_governance() -> None:
    metadata = ingestion_metadata()

    assert metadata["approval_status"] == "approved"
    assert metadata["lifecycle"] == "published"
    assert metadata["source_id"] == "semikb.demo.synthetic"
    assert metadata["source_manifest_version"] == "1.0.0"
    assert metadata["dataset_version"] == "demo-v2"
    assert metadata["source_license_status"] == "verified"
    assert metadata["redistribution_policy"] == "allowed"


@pytest.mark.asyncio
async def test_capacity_token_creation_retries_bounded_auth_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"detail": "rate limited"})
        return httpx.Response(200, json={"access_token": "test-token"})

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setenv("DEMO_ACCESS_KEY", "test-access-key")
    monkeypatch.setattr(capacity.asyncio, "sleep", fake_sleep)
    async with httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    ) as client:
        token = await capacity.create_token(client, "capacity-user")

    assert token == "test-token"
    assert attempts == 2
    assert delays == [12]


def test_capacity_summary_counts_failures_and_provider_attempts() -> None:
    probes = [
        StreamProbe(
            label="one",
            level=1,
            wave=1,
            request_id="req_0001",
            thread_id="thread_1",
            status_code=200,
            accepted_ms=10,
            first_delta_ms=100,
            total_ms=500,
            retrieval_ms=80,
            provider_attempts=3,
        ),
        StreamProbe(
            label="two",
            level=2,
            wave=1,
            request_id="req_0002",
            thread_id="thread_2",
            status_code=503,
            total_ms=50,
            error_code="HTTPStatusError",
        ),
    ]

    summary = build_summary(probes)

    assert summary["request_count"] == 2
    assert summary["completed_count"] == 1
    assert summary["error_rate"] == 0.5
    assert summary["http_5xx_count"] == 1
    assert summary["observed_provider_attempts"] == 3


def test_runtime_summary_parses_docker_units(tmp_path: Path) -> None:
    assert parse_size("1.5GiB") == 1.5 * 1024**3
    assert parse_size("640MiB") == 640 * 1024**2
    (tmp_path / "host.jsonl").write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "mem_available_kib": 1_000_000,
                        "swap_total_kib": 0,
                        "root_available_bytes": 50_000,
                        "load_1m": 0.5,
                    }
                ),
                json.dumps(
                    {
                        "mem_available_kib": 900_000,
                        "swap_total_kib": 0,
                        "root_available_bytes": 49_000,
                        "load_1m": 1.5,
                    }
                ),
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "containers.jsonl").write_text(
        json.dumps(
            {
                "Name": "semikb-api-1",
                "CPUPerc": "12.5%",
                "MemUsage": "120MiB / 640MiB",
                "MemPerc": "18.75%",
                "PIDs": "15",
            }
        ),
        encoding="utf-8",
    )

    report = summarize(tmp_path)

    assert report["host"]["min_available_memory_bytes"] == 900_000 * 1024
    assert report["host"]["root_available_delta_bytes"] == -1000
    assert report["containers"]["semikb-api-1"]["max_cpu_percent"] == 12.5


def test_state_comparison_ignores_capture_time_and_redis_queue_depth() -> None:
    base = {
        "captured_at": "before",
        "mongodb_counts": {"documents": 3},
        "milvus": {"row_count": 4},
        "minio_object_counts": {"raw": 2},
        "redis": {"celery_queue_depth": 1},
    }
    after = {**base, "captured_at": "after", "redis": {"celery_queue_depth": 0}}

    assert compare_snapshots(base, after) == {"matched": True, "differences": {}}
