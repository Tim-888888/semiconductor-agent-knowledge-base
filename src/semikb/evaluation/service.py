"""Versioned offline retrieval evaluation with reproducible run evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Protocol

from semikb.config import Settings
from semikb.contracts.models import (
    ActorScope,
    Chunk,
    EvaluationCase,
    EvaluationDataset,
    EvaluationRun,
    EvaluationStatus,
    RetrievalTrace,
)
from semikb.rag_retrieval.production_service import RetrievalOptions
from semikb.storage.evaluations import EvaluationRepository

_PROFILE_OPTIONS: dict[str, dict[str, bool | None]] = {
    "dense": {"dense": True, "sparse": False, "rerank": False, "hyde": False},
    "hybrid": {"dense": True, "sparse": True, "rerank": False, "hyde": False},
    "reranked": {"dense": True, "sparse": True, "rerank": True, "hyde": False},
    "full": {"dense": True, "sparse": True, "rerank": True, "hyde": None},
}
_DATASET_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_LOWER_IS_BETTER = {"average_latency_ms", "p95_latency_ms"}


class EvaluationRetrieval(Protocol):
    def search(
        self,
        query: str,
        actor_scope: ActorScope,
        **kwargs: Any,
    ) -> tuple[list[Chunk], RetrievalTrace]: ...


class EvaluationService:
    """Create immutable evaluation snapshots and execute queued retrieval runs."""

    def __init__(
        self,
        repository: EvaluationRepository,
        retrieval: EvaluationRetrieval,
        dataset_root: Path,
        settings: Settings | None = None,
    ) -> None:
        self.repository = repository
        self.dataset_root = dataset_root
        self.retrieval = retrieval
        self.settings = settings or Settings(_env_file=None, demo_mode=True)

    def create_run(
        self,
        dataset_version: str = "demo-v2",
        baseline_run_id: str | None = None,
        *,
        retrieval_profile: str = "full",
        requested_by: str = "system",
    ) -> EvaluationRun:
        dataset = self._load_dataset(dataset_version)
        dataset = self.repository.save_evaluation_dataset(dataset)
        if retrieval_profile not in _PROFILE_OPTIONS:
            raise ValueError(f"Unsupported retrieval profile: {retrieval_profile}")
        if baseline_run_id:
            baseline = self.repository.get_evaluation_run(baseline_run_id)
            if baseline is None or baseline.status is not EvaluationStatus.COMPLETED:
                raise ValueError("The baseline evaluation run must exist and be completed.")
            if baseline.dataset_hash != dataset.dataset_hash:
                raise ValueError("The baseline run must use the same immutable dataset snapshot.")
        run = EvaluationRun(
            dataset_version=dataset.dataset_version,
            dataset_hash=dataset.dataset_hash,
            case_count=dataset.case_count,
            baseline_run_id=baseline_run_id,
            requested_by=requested_by,
            retrieval_profile=retrieval_profile,
            retrieval_config=self._retrieval_config(retrieval_profile),
            component_versions=self._configured_component_versions(),
        )
        return self.repository.save_evaluation_run(run)

    def execute(
        self,
        evaluation_run_id: str,
        *,
        execution_id: str | None = None,
    ) -> EvaluationRun:
        run = self.repository.claim_evaluation_run(evaluation_run_id, execution_id)
        if run is None:
            existing = self.repository.get_evaluation_run(evaluation_run_id)
            if existing is None:
                raise KeyError(evaluation_run_id)
            return existing
        dataset = self.repository.get_evaluation_dataset(run.dataset_version)
        if dataset is None or dataset.dataset_hash != run.dataset_hash:
            return self._fail_and_raise(run, ValueError("Evaluation dataset snapshot is unavailable."))

        try:
            case_results = [self._evaluate_case(case, run) for case in dataset.cases]
            run.case_results = case_results
            run.aggregate_metrics = self._aggregate(case_results)
            run.failure_tags = sorted(
                {
                    tag
                    for result in case_results
                    for tag in result.get("failure_tags", [])
                }
            )
            run.component_versions = self._merge_component_versions(case_results)
            if run.baseline_run_id:
                baseline = self.repository.get_evaluation_run(run.baseline_run_id)
                if baseline is None or baseline.status is not EvaluationStatus.COMPLETED:
                    raise ValueError("The baseline evaluation run is no longer available.")
                run.baseline_comparison = self._compare_with_baseline(run, baseline)
                self._annotate_case_changes(run.case_results, baseline.case_results)
            run.status = EvaluationStatus.COMPLETED
            run.finished_at = datetime.now(UTC)
            return self.repository.save_evaluation_run(run)
        except Exception as exc:
            return self._fail_and_raise(run, exc)

    def run(
        self,
        dataset_version: str = "demo-v2",
        baseline_run_id: str | None = None,
        *,
        retrieval_profile: str = "full",
        requested_by: str = "system",
    ) -> EvaluationRun:
        """Synchronous convenience path for tests and explicit CLI verification."""

        queued = self.create_run(
            dataset_version,
            baseline_run_id,
            retrieval_profile=retrieval_profile,
            requested_by=requested_by,
        )
        return self.execute(queued.evaluation_run_id)

    def get_run(self, evaluation_run_id: str) -> EvaluationRun | None:
        return self.repository.get_evaluation_run(evaluation_run_id)

    def list_runs(self) -> list[EvaluationRun]:
        return self.repository.list_evaluation_runs()

    def list_datasets(self) -> list[EvaluationDataset]:
        return self.repository.list_evaluation_datasets()

    def prepare_retry(self, evaluation_run_id: str) -> EvaluationRun:
        return self.repository.prepare_evaluation_retry(evaluation_run_id)

    def mark_queue_submission_failed(self, evaluation_run_id: str) -> EvaluationRun:
        run = self.repository.get_evaluation_run(evaluation_run_id)
        if run is None:
            raise KeyError(evaluation_run_id)
        run.status = EvaluationStatus.FAILED
        run.failure_tags = sorted({*run.failure_tags, "queue_submission_failed"})
        run.safe_error_summary = "Evaluation task queue is unavailable."
        run.finished_at = datetime.now(UTC)
        return self.repository.save_evaluation_run(run)

    def _evaluate_case(self, case: EvaluationCase, run: EvaluationRun) -> dict[str, Any]:
        options = RetrievalOptions(**_PROFILE_OPTIONS[run.retrieval_profile])
        search_kwargs: dict[str, Any] = {"top_k": 5, "options": options}
        search_kwargs["thread_id"] = f"evaluation:{run.evaluation_run_id}:{case.case_id}"
        _, trace = self.retrieval.search(case.question, case.actor_scope, **search_kwargs)
        actual = trace.final_evidence_ids[:5]
        expected = list(dict.fromkeys(case.expected_chunk_ids))
        expected_set = set(expected)
        hit_positions = [
            index + 1
            for index, chunk_id in enumerate(actual)
            if chunk_id in expected_set
        ]

        if case.expected_outcome == "no_evidence":
            passed = not actual
            recall = 0.0
            reciprocal_rank = 0.0
            ndcg = 0.0
            failure_tags = [] if passed else (
                case.failure_labels or ["unexpected_evidence_for_negative_case"]
            )
        else:
            recall = len(expected_set.intersection(actual)) / len(expected_set) if expected_set else 0.0
            reciprocal_rank = 1.0 / hit_positions[0] if hit_positions else 0.0
            ndcg = self._ndcg_at_5(actual, expected_set)
            passed = bool(hit_positions)
            failure_tags = [] if passed else (
                case.failure_labels or ["missed_expected_evidence"]
            )

        selected_candidates = [
            {
                "chunk_id": candidate.chunk_id,
                "routes": candidate.routes,
                "route_ranks": candidate.route_ranks,
                "rrf_score": candidate.rrf_score,
                "rerank_score": candidate.rerank_score,
                "selection_reason": candidate.context_selection_reason,
            }
            for candidate in trace.candidates
            if candidate.selected
        ]
        return {
            "case_id": case.case_id,
            "question": case.question,
            "tags": case.tags,
            "expected_chunk_ids": expected,
            "expected_outcome": case.expected_outcome,
            "actual_chunk_ids": actual,
            "missing_expected_chunk_ids": sorted(expected_set.difference(actual)),
            "unexpected_chunk_ids": actual if case.expected_outcome == "no_evidence" else [],
            "passed": passed,
            "recall_at_5": round(recall, 4),
            "reciprocal_rank": round(reciprocal_rank, 4),
            "ndcg_at_5": round(ndcg, 4),
            "trace_id": trace.trace_id,
            "routes": trace.routes,
            "cutoff_reason": trace.cutoff_reason,
            "selected_candidates": selected_candidates,
            "image_asset_ids": trace.image_asset_ids,
            "warnings": trace.warnings,
            "component_versions": trace.component_versions,
            "latency_ms": float(trace.timings_ms.get("total", 0.0)),
            "failure_tags": failure_tags,
        }

    @staticmethod
    def _aggregate(case_results: list[dict[str, Any]]) -> dict[str, float]:
        positive = [item for item in case_results if item["expected_outcome"] == "evidence"]
        negative = [item for item in case_results if item["expected_outcome"] == "no_evidence"]
        image_cases = [item for item in positive if "image_text" in item["tags"]]
        latencies = sorted(float(item["latency_ms"]) for item in case_results)
        p95_index = max(0, math.ceil(len(latencies) * 0.95) - 1) if latencies else 0
        return {
            "recall_at_5": round(mean(item["recall_at_5"] for item in positive), 4) if positive else 0.0,
            "mrr": round(mean(item["reciprocal_rank"] for item in positive), 4) if positive else 0.0,
            "ndcg_at_5": round(mean(item["ndcg_at_5"] for item in positive), 4) if positive else 0.0,
            "no_evidence_accuracy": round(mean(float(item["passed"]) for item in negative), 4) if negative else 1.0,
            "image_recall_at_5": round(mean(item["recall_at_5"] for item in image_cases), 4) if image_cases else 1.0,
            "pass_rate": round(mean(float(item["passed"]) for item in case_results), 4) if case_results else 0.0,
            "average_latency_ms": round(mean(latencies), 2) if latencies else 0.0,
            "p95_latency_ms": round(latencies[p95_index], 2) if latencies else 0.0,
        }

    def _retrieval_config(self, retrieval_profile: str) -> dict[str, Any]:
        return {
            "profile": retrieval_profile,
            "top_k": 5,
            **_PROFILE_OPTIONS[retrieval_profile],
            "recall_k": self.settings.retrieval_recall_k,
            "rrf_k": self.settings.retrieval_rrf_k,
            "min_evidence": self.settings.retrieval_min_evidence,
            "max_evidence": self.settings.retrieval_max_evidence,
            "score_cliff_ratio": self.settings.retrieval_score_cliff_ratio,
            "rerank_min_score": self.settings.retrieval_rerank_min_score,
        }

    def _configured_component_versions(self) -> dict[str, str]:
        return {
            "embedding": "bge-m3" if not self.settings.demo_mode else "deterministic-demo",
            "embedding_dim": str(self.settings.embedding_dim),
            "reranker": self.settings.rerank_model,
            "index_version": self.settings.milvus_index_version,
            "hyde_policy": "conditional" if self.settings.hyde_enabled else "disabled",
        }

    def _merge_component_versions(self, case_results: list[dict[str, Any]]) -> dict[str, str]:
        values: dict[str, set[str]] = {}
        for key, value in self._configured_component_versions().items():
            values.setdefault(key, set()).add(value)
        for result in case_results:
            for key, value in result.get("component_versions", {}).items():
                values.setdefault(key, set()).add(str(value))
        return {
            key: next(iter(items)) if len(items) == 1 else "mixed:" + ",".join(sorted(items))
            for key, items in sorted(values.items())
        }

    @staticmethod
    def _compare_with_baseline(run: EvaluationRun, baseline: EvaluationRun) -> dict[str, Any]:
        comparison: dict[str, Any] = {}
        for metric, current in run.aggregate_metrics.items():
            if metric not in baseline.aggregate_metrics:
                continue
            previous = baseline.aggregate_metrics[metric]
            delta = round(current - previous, 4)
            direction = "lower_better" if metric in _LOWER_IS_BETTER else "higher_better"
            signed_improvement = -delta if direction == "lower_better" else delta
            outcome = "unchanged"
            if signed_improvement > 0.0001:
                outcome = "improved"
            elif signed_improvement < -0.0001:
                outcome = "regressed"
            comparison[metric] = {
                "current": current,
                "baseline": previous,
                "delta": delta,
                "direction": direction,
                "outcome": outcome,
            }
        return comparison

    @staticmethod
    def _annotate_case_changes(
        current_results: list[dict[str, Any]],
        baseline_results: list[dict[str, Any]],
    ) -> None:
        baseline_by_id = {item["case_id"]: item for item in baseline_results}
        for result in current_results:
            baseline = baseline_by_id.get(result["case_id"])
            if baseline is None:
                result["baseline_outcome"] = "new_case"
                continue
            result["baseline_passed"] = bool(baseline.get("passed"))
            if result["passed"] == baseline.get("passed"):
                result["baseline_outcome"] = "unchanged"
            elif result["passed"]:
                result["baseline_outcome"] = "fixed"
            else:
                result["baseline_outcome"] = "regressed"

    def _load_dataset(self, dataset_version: str) -> EvaluationDataset:
        if not _DATASET_VERSION_PATTERN.fullmatch(dataset_version):
            raise ValueError("Invalid evaluation dataset version.")
        path = self.dataset_root / f"{dataset_version.replace('-', '_')}.json"
        if not path.is_file():
            raise ValueError(f"Evaluation dataset {dataset_version!r} does not exist.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        declared_version = str(payload.get("dataset_version", dataset_version))
        if declared_version != dataset_version:
            raise ValueError("Evaluation dataset filename and declared version do not match.")
        cases = [EvaluationCase.model_validate(item) for item in payload.get("cases", [])]
        if not cases:
            raise ValueError("Evaluation dataset must contain at least one case.")
        canonical = json.dumps(
            {
                "dataset_version": declared_version,
                "source_kind": payload.get("source_kind", "synthetic"),
                "description": payload.get("description", ""),
                "cases": [case.model_dump(mode="json") for case in cases],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return EvaluationDataset(
            dataset_version=declared_version,
            dataset_hash=hashlib.sha256(canonical).hexdigest(),
            source_kind=str(payload.get("source_kind", "synthetic")),
            description=str(payload.get("description", "")),
            case_count=len(cases),
            cases=cases,
        )

    @staticmethod
    def _ndcg_at_5(actual: list[str], expected: set[str]) -> float:
        if not expected:
            return 0.0
        dcg = sum(
            1.0 / math.log2(index + 2)
            for index, chunk_id in enumerate(actual[:5])
            if chunk_id in expected
        )
        ideal_hits = min(len(expected), 5)
        ideal = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
        return dcg / ideal if ideal else 0.0

    def _fail_and_raise(self, run: EvaluationRun, exc: Exception) -> Any:
        run.status = EvaluationStatus.FAILED
        run.failure_tags = sorted({*run.failure_tags, type(exc).__name__})
        run.safe_error_summary = f"Evaluation failed with {type(exc).__name__}."
        run.finished_at = datetime.now(UTC)
        self.repository.save_evaluation_run(run)
        raise exc
