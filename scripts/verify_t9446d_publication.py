"""Publish reviewed corpus snapshots and run the T9-4.4.6d governed evaluation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx


class Api:
    def __init__(self, base_url: str, access_key: str) -> None:
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=120)
        response = self.client.post(
            "/api/v1/auth/demo-token",
            headers={"X-Demo-Access-Key": access_key},
            json={
                "user_id": "t9446d_admin",
                "roles": ["knowledge_admin"],
                "access_scope_keys": ["demo_engineering"],
            },
        )
        response.raise_for_status()
        self.headers = {
            "Authorization": f"Bearer {response.json()['access_token']}"
        }

    def get(self, path: str) -> Any:
        response = self.client.get(path, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def post(self, path: str, payload: dict[str, Any]) -> Any:
        response = self.client.post(path, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()


def _wait(api: Api, path: str, terminal: set[str], timeout: int = 2400) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = api.get(path)
        if value["status"] in terminal:
            return value
        time.sleep(3)
    raise TimeoutError(path)


def _snapshot_hash(batches: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        [
            {
                "batch_id": batch["batch_id"],
                "request_fingerprint": batch["request_fingerprint"],
                "source_hash": batch["source_manifest"]["source_hash"],
            }
            for batch in batches
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _publish(api: Api, spec: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = {job["metadata"]["corpus_id"]: job for job in api.get(
        "/api/v1/corpus-standardization-jobs"
    )}
    batches: list[dict[str, Any]] = []
    for review in spec["reviews"]:
        job = jobs[review["corpus_id"]]
        report = job["report"]
        selected_paths = set(review.get("selected_paths", []))
        selected = [
            item["file_id"]
            for item in report["files"]
            if item.get("standardized_ref")
            and item["role"] not in {"archive", "unsupported"}
            and (not selected_paths or item["relative_path"] in selected_paths)
        ]
        payload = {
            "request_id": review["request_id"],
            "standardization_job_id": job["job_id"],
            "expected_snapshot_hash": job["snapshot_hash"],
            "selected_file_ids": selected,
            "acknowledged_warning_codes": report["warning_codes"],
            "source_type": review["source_type"],
            "content_origin": review.get("content_origin", "real"),
            "source_url": review["source_url"],
            "license_name": review["license_name"],
            "license_status": review.get("license_status", "declared"),
            "redistribution_policy": review.get("redistribution_policy", "restricted"),
            "license_notes": review.get("license_notes", ""),
            "access_scope_key": review.get("access_scope_key", "demo_engineering"),
            "review_note": review["review_note"],
            "retrieval_policy": review.get("retrieval_policy", "standard"),
        }
        batch = api.post("/api/v1/corpus-publication-batches", payload)
        batch = _wait(
            api,
            f"/api/v1/corpus-publication-batches/{batch['batch_id']}",
            {"completed", "failed"},
        )
        if batch["status"] != "completed":
            raise RuntimeError(
                f"Publication {batch['batch_id']} failed: {batch.get('error_code')}"
            )
        batches.append(batch)
    return batches


def _artifact_chunks(
    batches: list[dict[str, Any]],
) -> dict[tuple[str, str], list[str]]:
    result: dict[tuple[str, str], list[str]] = {}
    for batch in batches:
        corpus_id = batch["source_manifest"]["source_id"]
        for item in batch["items"]:
            result[(corpus_id, item["relative_path"])] = item["reconciliation"][
                "published_chunk_ids"
            ]
    return result


def _register_dataset(
    api: Api,
    dataset: dict[str, Any],
    artifact_chunks: dict[tuple[str, str], list[str]],
    source_snapshot_hash: str,
) -> dict[str, Any]:
    cases = []
    for case in dataset["cases"]:
        expected: list[str] = list(case.get("expected_chunk_ids", []))
        for artifact in case.get("expected_artifacts", []):
            expected.extend(
                artifact_chunks[(artifact["corpus_id"], artifact["relative_path"])]
            )
        actor_scope = case.get(
            "actor_scope",
            {
                "user_id": "t9446d_evaluation",
                "roles": ["engineer"],
                "access_scope_keys": ["demo_engineering"],
                "fabs": [],
                "products": [],
                "tool_ids": [],
            },
        )
        cases.append(
            {
                "case_id": case["case_id"],
                "question": case["question"],
                "expected_chunk_ids": sorted(set(expected)),
                "expected_outcome": case.get(
                    "expected_outcome", "evidence" if expected else "no_evidence"
                ),
                "actor_scope": actor_scope,
                "tags": case.get("tags", []),
                "failure_labels": case.get("failure_labels", []),
            }
        )
    return api.post(
        "/api/v1/evaluation-datasets",
        {
            "dataset_version": dataset["dataset_version"],
            "source_kind": dataset.get("source_kind", "process-separated-test"),
            "description": dataset.get("description", ""),
            "purpose": dataset["purpose"],
            "source_snapshot_hash": source_snapshot_hash,
            "leakage_status": dataset.get("leakage_status", "cleared"),
            "seal": dataset["purpose"] == "holdout",
            "cases": cases,
        },
    )


def _run_evaluation(api: Api, version: str) -> dict[str, Any]:
    run = api.post(
        "/api/v1/evaluation-runs",
        {"dataset_version": version, "retrieval_profile": "full"},
    )
    return _wait(
        api,
        f"/api/v1/evaluation-runs/{run['evaluation_run_id']}",
        {"completed", "failed"},
    )


def verify(args: argparse.Namespace) -> dict[str, Any]:
    api = Api(args.base_url, args.access_key)
    publication_spec = json.loads(args.publication_spec.read_text(encoding="utf-8"))
    public_spec = json.loads(args.public_splits.read_text(encoding="utf-8"))
    holdout_spec = json.loads(args.private_holdout.read_text(encoding="utf-8"))
    batches = _publish(api, publication_spec)
    artifact_chunks = _artifact_chunks(batches)
    source_snapshot_hash = _snapshot_hash(batches)
    public_datasets = [
        _register_dataset(api, item, artifact_chunks, source_snapshot_hash)
        for item in public_spec["datasets"]
    ]
    holdout = _register_dataset(
        api,
        holdout_spec["dataset"],
        artifact_chunks,
        source_snapshot_hash,
    )
    public_runs = [
        _run_evaluation(api, dataset["dataset_version"])
        for dataset in public_datasets
    ]
    by_purpose = {dataset["purpose"]: dataset for dataset in public_datasets}
    freeze = api.post(
        "/api/v1/evaluation-release-freezes",
        {
            "release_version": args.release_version,
            "source_commit": args.source_commit,
            "publication_batch_ids": [batch["batch_id"] for batch in batches],
            "development_dataset_version": by_purpose["development"]["dataset_version"],
            "calibration_dataset_version": by_purpose["calibration"]["dataset_version"],
            "regression_dataset_version": by_purpose["regression"]["dataset_version"],
            "holdout_dataset_version": holdout["dataset_version"],
            "notes": "Implementation and visible evaluation inputs frozen before holdout opening.",
        },
    )
    holdout_run = _run_evaluation(api, holdout["dataset_version"])
    return {
        "schema_version": "semikb-t9446d-verification-v1",
        "publication_batches": batches,
        "public_datasets": public_datasets,
        "public_runs": public_runs,
        "release_freeze": freeze,
        "holdout_dataset": {
            key: value for key, value in holdout.items() if key != "cases"
        },
        "holdout_run": holdout_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1")
    parser.add_argument("--access-key", default=os.getenv("DEMO_ACCESS_KEY", ""))
    parser.add_argument("--publication-spec", type=Path, required=True)
    parser.add_argument(
        "--public-splits",
        type=Path,
        default=Path("data/evaluation_specs/t9446d_public_splits_v1.json"),
    )
    parser.add_argument("--private-holdout", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--release-version", default="t9446d-release-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.access_key:
        raise SystemExit("DEMO_ACCESS_KEY or --access-key is required.")
    result = verify(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "publication_batches": len(result["publication_batches"]),
                "public_runs": [
                    {
                        "dataset": item["dataset_version"],
                        "status": item["status"],
                        "pass_rate": item["aggregate_metrics"].get("pass_rate"),
                    }
                    for item in result["public_runs"]
                ],
                "holdout_status": result["holdout_run"]["status"],
                "holdout_pass_rate": result["holdout_run"]["aggregate_metrics"].get(
                    "pass_rate"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
