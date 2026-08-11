from __future__ import annotations

import pytest

from semikb.contracts.models import EvaluationStatus


def test_golden_set_evaluation_is_reproducible(seeded_services) -> None:
    _, _, _, _, evaluation = seeded_services
    run = evaluation.run("demo-v1")

    assert run.status is EvaluationStatus.COMPLETED
    assert len(run.case_results) == 3
    assert run.aggregate_metrics["recall_at_5"] == 1.0
    assert run.aggregate_metrics["mrr"] > 0


def test_expanded_golden_set_preserves_negative_case_results(seeded_services) -> None:
    _, _, _, _, evaluation = seeded_services
    run = evaluation.run("demo-v2")

    assert run.status is EvaluationStatus.COMPLETED
    assert len(run.case_results) == 8
    negative_results = [
        result for result in run.case_results if result["expected_outcome"] == "no_evidence"
    ]
    assert len(negative_results) == 3


def test_evaluation_freezes_profile_versions_and_real_baseline_comparison(
    seeded_services,
) -> None:
    store, _, _, _, evaluation = seeded_services
    baseline = evaluation.run("demo-v1", retrieval_profile="dense", requested_by="admin")
    current = evaluation.run(
        "demo-v1",
        baseline.evaluation_run_id,
        retrieval_profile="full",
        requested_by="admin",
    )

    assert len(store.list_evaluation_datasets()) == 1
    assert current.dataset_hash == baseline.dataset_hash
    assert current.retrieval_config["profile"] == "full"
    assert current.retrieval_config["hyde"] is None
    assert current.component_versions["embedding"] == "deterministic-demo"
    assert current.baseline_comparison["recall_at_5"]["outcome"] == "unchanged"
    assert all(result["baseline_outcome"] == "unchanged" for result in current.case_results)
    assert {"ndcg_at_5", "no_evidence_accuracy", "p95_latency_ms"}.issubset(
        current.aggregate_metrics
    )


def test_unknown_evaluation_dataset_does_not_silently_fall_back(seeded_services) -> None:
    _, _, _, _, evaluation = seeded_services

    with pytest.raises(ValueError, match="does not exist"):
        evaluation.create_run("missing-v99")


def test_worker_redelivery_can_only_reclaim_with_the_same_task_id(seeded_services) -> None:
    store, _, _, _, evaluation = seeded_services
    queued = evaluation.create_run("demo-v1")

    first_claim = store.claim_evaluation_run(queued.evaluation_run_id, "task-1")
    same_task_redelivery = store.claim_evaluation_run(queued.evaluation_run_id, "task-1")
    different_task = store.claim_evaluation_run(queued.evaluation_run_id, "task-2")

    assert first_claim is not None
    assert same_task_redelivery is not None
    assert different_task is None
