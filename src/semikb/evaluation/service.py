"""Deterministic offline evaluation for retrieval regressions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from semikb.contracts.models import EvaluationCase, EvaluationRun, EvaluationStatus
from semikb.rag_retrieval.service import RetrievalService
from semikb.storage.memory import DemoStore


class EvaluationService:
    """Runs a fixed dataset and saves enough detail for failure-case drill-down."""

    def __init__(self, store: DemoStore, retrieval: RetrievalService, dataset_root: Path) -> None:
        self.store = store
        self.dataset_root = dataset_root
        self.retrieval = retrieval

    def run(self, dataset_version: str = "demo-v1", baseline_run_id: str | None = None) -> EvaluationRun:
        run = EvaluationRun(dataset_version=dataset_version, baseline_run_id=baseline_run_id, status=EvaluationStatus.RUNNING)
        self.store.save_evaluation_run(run)
        try:
            cases = self._load_cases(dataset_version)
            results: list[dict[str, object]] = []
            recalls: list[float] = []
            reciprocal_ranks: list[float] = []
            for case in cases:
                _, trace = self.retrieval.search(case.question, case.actor_scope, top_k=5)
                actual = trace.final_evidence_ids
                expected = set(case.expected_chunk_ids)
                hit_positions = [index + 1 for index, chunk_id in enumerate(actual) if chunk_id in expected]
                recall = 1.0 if hit_positions else 0.0
                reciprocal_rank = 1.0 / hit_positions[0] if hit_positions else 0.0
                recalls.append(recall)
                reciprocal_ranks.append(reciprocal_rank)
                results.append(
                    {
                        "case_id": case.case_id,
                        "question": case.question,
                        "expected_chunk_ids": case.expected_chunk_ids,
                        "actual_chunk_ids": actual,
                        "trace_id": trace.trace_id,
                        "recall_at_5": recall,
                        "reciprocal_rank": reciprocal_rank,
                        "failure_tags": [] if hit_positions else case.tags,
                    }
                )
            run.status = EvaluationStatus.COMPLETED
            run.case_results = results
            run.aggregate_metrics = {
                "recall_at_5": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
                "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else 0.0,
            }
            run.failure_tags = sorted({tag for result in results for tag in result["failure_tags"]})
            run.finished_at = datetime.now(UTC)
            return self.store.save_evaluation_run(run)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            run.status = EvaluationStatus.FAILED
            run.failure_tags = [type(exc).__name__]
            run.finished_at = datetime.now(UTC)
            return self.store.save_evaluation_run(run)

    def _load_cases(self, dataset_version: str) -> list[EvaluationCase]:
        path = self.dataset_root / f"{dataset_version.replace('-', '_')}.json"
        if not path.exists():
            path = self.dataset_root / "demo_v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [EvaluationCase.model_validate(item) for item in payload["cases"]]
