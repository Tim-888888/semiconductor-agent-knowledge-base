"""Run T7 profiles through Redis/Celery and verify MongoDB persistence."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from semikb.config import Settings
from semikb.contracts.models import EvaluationRun, EvaluationStatus
from semikb.evaluation.service import EvaluationService
from semikb.rag_retrieval.production_service import ProductionRetrievalService
from semikb.storage.evaluations import MongoEvaluationRepository
from semikb.workers.tasks import run_evaluation


def wait_for_run(
    repository: MongoEvaluationRepository,
    evaluation_run_id: str,
    timeout_seconds: int,
) -> EvaluationRun:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        run = repository.get_evaluation_run(evaluation_run_id)
        if run and run.status in {EvaluationStatus.COMPLETED, EvaluationStatus.FAILED}:
            return run
        time.sleep(1)
    raise TimeoutError(f"Evaluation run {evaluation_run_id} did not finish in time.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--replace-acceptance-runs",
        action="store_true",
        help="Replace only prior t7_live_acceptance runs and their referenced traces.",
    )
    args = parser.parse_args()

    settings = Settings(demo_mode=False)
    repository = MongoEvaluationRepository(settings)
    retrieval = ProductionRetrievalService(settings)
    service = EvaluationService(
        repository,
        retrieval,
        Path(__file__).resolve().parents[1] / "data" / "golden_sets",
        settings,
    )

    cleaned_runs = 0
    cleaned_traces = 0
    if args.replace_acceptance_runs:
        selector = {
            "requested_by": "t7_live_acceptance",
            "dataset_version": "t5-live-v1",
        }
        previous = list(repository.runs.find(selector, {"case_results.trace_id": 1}))
        trace_ids = [
            result["trace_id"]
            for document in previous
            for result in document.get("case_results", [])
            if result.get("trace_id")
        ]
        cleaned_runs = repository.runs.delete_many(selector).deleted_count
        if trace_ids:
            cleaned_traces = repository.database.retrieval_traces.delete_many(
                {"trace_id": {"$in": trace_ids}}
            ).deleted_count

    summaries: list[dict[str, object]] = []
    baseline_run_id: str | None = None
    for profile in ("dense", "hybrid", "reranked", "full"):
        queued = service.create_run(
            "t5-live-v1",
            baseline_run_id,
            retrieval_profile=profile,
            requested_by="t7_live_acceptance",
        )
        run_evaluation.delay(queued.evaluation_run_id)
        completed = wait_for_run(repository, queued.evaluation_run_id, args.timeout)
        if completed.status is not EvaluationStatus.COMPLETED:
            raise RuntimeError(
                f"Profile {profile} failed: {completed.safe_error_summary or completed.failure_tags}"
            )
        if completed.case_count != len(completed.case_results):
            raise RuntimeError(f"Profile {profile} did not persist every case result.")
        if any(not result.get("trace_id") for result in completed.case_results):
            raise RuntimeError(f"Profile {profile} has a case without a retrieval trace.")
        summaries.append(
            {
                "evaluation_run_id": completed.evaluation_run_id,
                "profile": profile,
                "attempt": completed.attempt,
                "metrics": completed.aggregate_metrics,
                "baseline_run_id": completed.baseline_run_id,
                "baseline_comparison": completed.baseline_comparison,
                "failure_tags": completed.failure_tags,
            }
        )
        baseline_run_id = completed.evaluation_run_id

    fresh_repository = MongoEvaluationRepository(settings)
    persisted = fresh_repository.get_evaluation_run(baseline_run_id or "")
    if persisted is None or persisted.status is not EvaluationStatus.COMPLETED:
        raise RuntimeError("A fresh repository instance could not restore the final run.")
    metrics = persisted.aggregate_metrics
    if metrics["recall_at_5"] < 0.85:
        raise RuntimeError("The full retrieval profile failed the Recall@5 threshold.")
    if metrics["no_evidence_accuracy"] < 1.0:
        raise RuntimeError("The full retrieval profile returned evidence for a negative case.")
    if metrics["image_recall_at_5"] < 1.0:
        raise RuntimeError("The full retrieval profile missed the image evidence case.")

    print(
        json.dumps(
            {
                "status": "ok",
                "dataset_version": "t5-live-v1",
                "dataset_hash": persisted.dataset_hash,
                "replaced_acceptance_runs": cleaned_runs,
                "replaced_acceptance_traces": cleaned_traces,
                "profiles": summaries,
                "restart_visibility": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
