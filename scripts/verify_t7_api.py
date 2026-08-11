"""Verify T7 authorization, asynchronous submission, and persisted API reads."""

from __future__ import annotations

import argparse
import json
import time

import httpx


def token(client: httpx.Client, user_id: str, roles: list[str]) -> str:
    response = client.post(
        "/api/v1/auth/demo-token",
        json={
            "user_id": user_id,
            "roles": roles,
            "access_scope_keys": ["demo_engineering"],
            "fabs": ["FAB-01"],
            "products": ["P-ALPHA"],
            "tool_ids": ["ETCH-03"],
        },
    )
    response.raise_for_status()
    return response.json()["access_token"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=30) as client:
        engineer_headers = {
            "Authorization": f"Bearer {token(client, 't7_api_engineer', ['engineer'])}"
        }
        forbidden = client.get("/api/v1/evaluation-runs", headers=engineer_headers)
        if forbidden.status_code != 403:
            raise RuntimeError("A non-admin user could read evaluation runs.")

        admin_headers = {
            "Authorization": f"Bearer {token(client, 't7_api_acceptance', ['knowledge_admin'])}"
        }
        created = client.post(
            "/api/v1/evaluation-runs",
            json={"dataset_version": "t5-live-v1", "retrieval_profile": "dense"},
            headers=admin_headers,
        )
        if created.status_code != 202 or created.json()["status"] != "queued":
            raise RuntimeError(f"Evaluation submission was not queued: {created.text}")
        evaluation_run_id = created.json()["evaluation_run_id"]

        deadline = time.monotonic() + args.timeout
        run: dict[str, object] = {}
        while time.monotonic() < deadline:
            response = client.get(
                f"/api/v1/evaluation-runs/{evaluation_run_id}",
                headers=admin_headers,
            )
            response.raise_for_status()
            run = response.json()
            if run["status"] in {"completed", "failed"}:
                break
            time.sleep(1)
        if run.get("status") != "completed":
            raise RuntimeError(f"Evaluation run did not complete: {run}")

        first_case = run["case_results"][0]
        trace = client.get(
            f"/api/v1/evaluation-runs/{evaluation_run_id}/cases/{first_case['case_id']}/trace",
            headers=admin_headers,
        )
        trace.raise_for_status()
        if trace.json()["trace_id"] != first_case["trace_id"]:
            raise RuntimeError("The evaluation Case returned an unrelated retrieval trace.")

        datasets = client.get("/api/v1/evaluation-datasets", headers=admin_headers)
        datasets.raise_for_status()
        versions = {item["dataset_version"] for item in datasets.json()}
        if "t5-live-v1" not in versions:
            raise RuntimeError("The immutable dataset snapshot is not available through the API.")

        print(
            json.dumps(
                {
                    "status": "ok",
                    "evaluation_run_id": evaluation_run_id,
                    "submission_status": created.status_code,
                    "engineer_list_status": forbidden.status_code,
                    "final_status": run["status"],
                    "metrics": run["aggregate_metrics"],
                    "dataset_visible": True,
                    "case_trace_visible": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
