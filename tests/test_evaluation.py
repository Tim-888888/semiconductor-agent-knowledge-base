from __future__ import annotations

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
