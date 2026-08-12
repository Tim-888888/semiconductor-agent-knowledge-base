"""Run live T5 retrieval baselines against the governed synthetic corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from semikb.config import Settings
from semikb.contracts.models import EvaluationCase
from semikb.rag_retrieval.production_service import (
    ProductionRetrievalService,
    RetrievalOptions,
)


def load_cases(root: Path) -> list[EvaluationCase]:
    payload = json.loads(
        (root / "data" / "golden_sets" / "t5_live_v1.json").read_text(encoding="utf-8")
    )
    return [EvaluationCase.model_validate(item) for item in payload["cases"]]


def evaluate_variant(
    service: ProductionRetrievalService,
    cases: list[EvaluationCase],
    options: RetrievalOptions,
) -> dict[str, object]:
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    negative_results: list[float] = []
    latencies: list[float] = []
    case_results: list[dict[str, object]] = []
    for case in cases:
        started = perf_counter()
        _, trace = service.search(
            case.question,
            case.actor_scope,
            top_k=5,
            options=options,
        )
        latencies.append((perf_counter() - started) * 1000)
        actual = trace.final_evidence_ids
        expected = set(case.expected_chunk_ids)
        hit_positions = [index + 1 for index, chunk_id in enumerate(actual) if chunk_id in expected]
        if case.expected_outcome == "no_evidence":
            passed = not actual
            recall = 1.0 if passed else 0.0
            reciprocal_rank = 0.0
            negative_results.append(recall)
        else:
            passed = bool(hit_positions)
            recall = 1.0 if passed else 0.0
            reciprocal_rank = 1.0 / hit_positions[0] if hit_positions else 0.0
        recalls.append(recall)
        reciprocal_ranks.append(reciprocal_rank)
        case_results.append(
            {
                "case_id": case.case_id,
                "passed": passed,
                "actual_chunk_ids": actual,
                "trace_id": trace.trace_id,
                "routes": trace.routes,
                "warnings": trace.warnings,
            }
        )
    return {
        "recall_at_5": round(mean(recalls), 4),
        "mrr": round(mean(reciprocal_ranks), 4),
        "no_evidence_accuracy": round(mean(negative_results), 4) if negative_results else 1.0,
        "average_latency_ms": round(mean(latencies), 2),
        "case_results": case_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include every case result instead of only aggregate metrics.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cases = load_cases(root)
    settings = Settings(demo_mode=False)
    service = ProductionRetrievalService(settings)
    variants = {
        "dense": RetrievalOptions(dense=True, sparse=False, rerank=False, hyde=False),
        "dense_sparse": RetrievalOptions(dense=True, sparse=True, rerank=False, hyde=False),
        "dense_sparse_rerank": RetrievalOptions(
            dense=True,
            sparse=True,
            rerank=True,
            hyde=False,
        ),
        "dense_sparse_hyde_rerank": RetrievalOptions(
            dense=True,
            sparse=True,
            rerank=True,
            hyde=True,
        ),
    }
    results = {
        name: evaluate_variant(service, cases, options)
        for name, options in variants.items()
    }
    full = results["dense_sparse_hyde_rerank"]
    threshold_failed = (
        full["recall_at_5"] < 0.85 or full["no_evidence_accuracy"] < 1.0
    )
    output_results = results if args.details else {
        name: {key: value for key, value in result.items() if key != "case_results"}
        for name, result in results.items()
    }
    print(
        json.dumps(
            {
                "status": "ok",
                "dataset_version": "t5-live-v1",
                "results": output_results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if threshold_failed:
        raise RuntimeError("Full T5 retrieval pipeline failed the live acceptance threshold.")


if __name__ == "__main__":
    main()
