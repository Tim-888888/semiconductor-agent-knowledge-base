"""Run the deterministic T9-4.5 Provider fault matrix and write a safe report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

CASES = {
    "embedding": [
        "tests/test_embedding_provider.py::test_qianwen_encoder_rejects_wrong_dimension_and_safe_http_errors",
    ],
    "reranker": [
        "tests/test_production_retrieval.py::test_qianwen_reranker_exhaustion_degrades_with_attempt_audit",
    ],
    "llm": [
        "tests/test_llm_gateway.py::test_primary_failure_falls_back_with_qwen_parameter_shape",
        "tests/test_llm_gateway.py::test_stream_failure_after_visible_content_never_retries_or_falls_back",
    ],
    "mineru": [
        "tests/test_semikb_ingest_provider_adapters.py::test_mineru_retries_idempotent_transfer_but_not_batch_creation",
        "tests/test_semikb_ingest_provider_adapters.py::test_mineru_does_not_replay_ambiguous_batch_creation_failure",
    ],
    "ocr_vlm": [
        "tests/test_semikb_ingest_provider_adapters.py::test_qwen_vision_retries_transient_failure_then_records_success",
        "tests/test_semikb_ingest_provider_adapters.py::test_qwen_vision_client_rejects_unstructured_provider_output",
    ],
    "web_mcp": [
        "tests/test_web_search_resilience.py::test_web_search_retries_timeout_and_keeps_only_allowed_urls",
        "tests/test_web_search_resilience.py::test_web_search_invalid_response_fails_without_retry",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evidence/t9-4-5/local-fault-matrix.json"),
    )
    args = parser.parse_args()

    tests = [test for provider_tests in CASES.values() for test in provider_tests]
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *tests],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    report = {
        "schema_version": "semikb-t945-provider-resilience-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "result": "passed" if completed.returncode == 0 else "failed",
        "return_code": completed.returncode,
        "provider_matrix": {
            "embedding": {
                "faults": ["429", "invalid_vector"],
                "policy": "bounded_retry_then_fail_closed_with_trace",
            },
            "reranker": {
                "faults": ["5xx"],
                "policy": "bounded_retry_then_rrf_fallback_with_trace",
            },
            "llm": {
                "faults": ["5xx", "stream_interrupted"],
                "policy": "retry_or_fallback_only_before_visible_content",
            },
            "mineru": {
                "faults": ["5xx", "ambiguous_create_batch"],
                "policy": "retry_idempotent_transfer_only_and_keep_unpublished",
            },
            "ocr_vlm": {
                "faults": ["5xx", "invalid_structured_response"],
                "policy": "bounded_retry_transient_only_and_keep_unpublished",
            },
            "web_mcp": {
                "faults": ["timeout", "invalid_response"],
                "policy": "bounded_retry_then_internal_evidence_only",
            },
        },
        "tests": CASES,
        "pytest_summary": _last_nonempty_line(completed.stdout),
        "credential_safe": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote credential-safe report to {args.output}")
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
    return completed.returncode


def _last_nonempty_line(value: str) -> str:
    return next((line.strip() for line in reversed(value.splitlines()) if line.strip()), "")


if __name__ == "__main__":
    raise SystemExit(main())
