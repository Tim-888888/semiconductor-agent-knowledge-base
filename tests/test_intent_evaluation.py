from __future__ import annotations

from pathlib import Path

import pytest

from semikb.agent_runtime.llm_gateway import OpenAICompatibleLLMGateway
from semikb.agent_runtime.understanding import ConversationUnderstandingService
from semikb.config import Settings
from semikb.evaluation.intent import IntentEvaluationDataset, IntentEvaluationRunner

DATASET = Path(__file__).resolve().parents[1] / "data" / "intent_sets" / "semikb_intent_v1.json"
REGRESSION_DATASET = (
    Path(__file__).resolve().parents[1] / "data" / "intent_sets" / "semikb_intent_v2.json"
)


def test_intent_dataset_is_frozen_and_covers_required_boundaries() -> None:
    dataset = IntentEvaluationDataset.load(DATASET)
    case_ids = [case.case_id for case in dataset.cases]
    tags = {tag for case in dataset.cases for tag in case.tags}

    assert dataset.dataset_version == "semikb-intent-v1"
    assert dataset.source_kind == "synthetic_review_required"
    assert len(dataset.cases) == 96
    assert len(case_ids) == len(set(case_ids))
    assert dataset.dataset_hash == "01caa8a4fa2180e2dea19df3ea04da2cb80512f10bf10076d47c42dbce53b228"
    assert {
        "history",
        "history_transform",
        "feedback",
        "knowledge",
        "data_query",
        "investigation",
        "missing_slots",
        "mixed",
        "control",
        "unsafe",
        "evidence_reuse",
    }.issubset(tags)


@pytest.mark.asyncio
async def test_deterministic_intent_baseline_has_no_hidden_route_failures() -> None:
    settings = Settings(_env_file=None, demo_mode=True)
    dataset = IntentEvaluationDataset.load(DATASET)
    runner = IntentEvaluationRunner(
        ConversationUnderstandingService(settings, OpenAICompatibleLLMGateway(settings))
    )

    result = await runner.run(dataset)

    assert result.evaluated_cases == 96
    assert result.failures == []
    assert all(
        result.metrics[name] == 1.0
        for name in (
            "interaction_mode_accuracy",
            "primary_intent_accuracy",
            "route_accuracy",
            "route_macro_precision",
            "route_macro_recall",
        )
    )
    assert all(
        result.metrics[name] == 0.0
        for name in (
            "multi_task_miss_rate",
            "unnecessary_retrieval_rate",
            "wrong_evidence_reuse_rate",
            "wrong_clarification_rate",
            "slot_correction_failure_rate",
            "dangerous_execution_miss_rate",
        )
    )
    assert result.source_counts == {"l0": 35, "deterministic_fallback": 61}


def test_intent_v2_freezes_history_recall_production_regressions() -> None:
    dataset = IntentEvaluationDataset.load(REGRESSION_DATASET)
    regression_cases = [
        case for case in dataset.cases if "production_regression" in case.tags
    ]

    assert dataset.dataset_version == "semikb-intent-v2"
    assert len(dataset.cases) == 99
    assert dataset.dataset_hash == "ad13bb6b49ff604f5e25eb1b27e85c6db2a7b455ee7b1d24fbcd9d85641b95be"
    assert len(regression_cases) == 3
    assert all(case.expected_context_message_ids == ["msg_prev_q"] for case in regression_cases)


@pytest.mark.asyncio
async def test_intent_v2_baseline_selects_correct_history_message() -> None:
    settings = Settings(_env_file=None, demo_mode=True)
    dataset = IntentEvaluationDataset.load(REGRESSION_DATASET)
    runner = IntentEvaluationRunner(
        ConversationUnderstandingService(settings, OpenAICompatibleLLMGateway(settings))
    )

    result = await runner.run(dataset)

    assert result.evaluated_cases == 99
    assert result.failures == []
    assert result.metrics["route_accuracy"] == 1.0
    assert result.metrics["unnecessary_retrieval_rate"] == 0.0
    assert result.metrics["context_reference_accuracy"] == 1.0
