from __future__ import annotations

from pathlib import Path

import pytest

from semikb.agent_runtime.intent_catalog import IntentCatalog, IntentExampleBank
from semikb.agent_runtime.llm_gateway import OpenAICompatibleLLMGateway
from semikb.agent_runtime.understanding import ConversationUnderstandingService
from semikb.config import Settings
from semikb.evaluation.intent import IntentEvaluationDataset, IntentEvaluationRunner

DATASET = Path(__file__).resolve().parents[1] / "data" / "intent_sets" / "semikb_intent_v1.json"
REGRESSION_DATASET = (
    Path(__file__).resolve().parents[1] / "data" / "intent_sets" / "semikb_intent_v2.json"
)
V3_DATASET = Path(__file__).resolve().parents[1] / "data" / "intent_sets" / "semikb_intent_v3.json"
V4_DATASET = Path(__file__).resolve().parents[1] / "data" / "intent_sets" / "semikb_intent_v4.json"
CATALOG = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "intent_catalogs"
    / "semikb_intent_catalog_v1.json"
)
EXAMPLE_BANK = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "intent_examples"
    / "intent_example_bank_v1.json"
)


def _assert_only_intended_legacy_semantic_migrations(
    failures: list[dict[str, object]],
) -> None:
    """Old sets may differ only through T9-4.11 slot or evidence-task migration."""

    assert failures
    for failure in failures:
        expected = failure["expected"]
        actual = failure["actual"]
        assert isinstance(expected, dict)
        assert isinstance(actual, dict)
        same_primary_intent = actual["primary_intent"] == expected["primary_intent"]
        conditional_slot_migration = (
            expected["route"].value == "clarify"
            and actual["route"].value in {"clarify", "tool_only", "rag_and_tool"}
            and same_primary_intent
        )
        evidence_task_migration = (
            expected["primary_intent"].value == "investigation"
            and actual["primary_intent"].value == "investigation"
            and actual["route"].value in {"clarify", "rag_and_tool"}
            and any(
                task["primary_intent"] == "data_query"
                and task["execution_policy"] == "execute"
                for task in actual["tasks"]
            )
        )
        assert conditional_slot_migration or evidence_task_migration, failure


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
    _assert_only_intended_legacy_semantic_migrations(result.failures)
    assert all(
        result.metrics[name] == 1.0
        for name in (
            "interaction_mode_accuracy",
            "primary_intent_accuracy",
        )
    )
    assert result.metrics["route_accuracy"] >= 0.95
    assert all(
        result.metrics[name] == 0.0
        for name in (
                "multi_task_miss_rate",
                "wrong_evidence_reuse_rate",
            "wrong_clarification_rate",
            "slot_correction_failure_rate",
            "dangerous_execution_miss_rate",
        )
    )
    assert result.metrics["unnecessary_retrieval_rate"] <= 0.02
    assert result.source_counts == {"l0": 35, "deterministic_fallback": 61}
    assert result.route_migration["migrated_expectation_count"] == 16
    assert result.metrics["deprecated_history_direct_emission_rate"] == 0.0


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
    _assert_only_intended_legacy_semantic_migrations(result.failures)
    assert result.metrics["route_accuracy"] >= 0.95
    assert result.metrics["wrong_clarification_rate"] == 0.0
    assert result.metrics["context_reference_accuracy"] == 1.0


def test_intent_v3_is_frozen_structured_and_isolated_from_examples() -> None:
    dataset = IntentEvaluationDataset.load(V3_DATASET)
    catalog = IntentCatalog.load(CATALOG)
    example_bank = IntentExampleBank.load(EXAMPLE_BANK)
    tags = {tag for case in dataset.cases for tag in case.tags}

    assert dataset.dataset_version == "semikb-intent-v3"
    assert dataset.catalog_version == "semikb-intent-catalog-v1"
    assert dataset.example_bank_version == "intent-example-bank-v1"
    assert len(dataset.cases) == 108
    assert dataset.dataset_hash == "e5f2f689655e0eccd945bb25eec49e167c2dc07db5436c6dc4d360fc38976679"
    assert {"confusion_pair", "multi_task", "dependency", "affect"}.issubset(tags)
    assert all(len(case.expected_tasks) == case.expected_task_count for case in dataset.cases)
    assert all(
        len(case.expected_intent_card_ids) == case.expected_task_count
        for case in dataset.cases
    )

    example_bank.validate_against_catalog(catalog)
    example_bank.assert_no_evaluation_leakage(case.utterance for case in dataset.cases)


@pytest.mark.asyncio
async def test_intent_v3_deterministic_baseline_emits_reproducible_detailed_report() -> None:
    settings = Settings(_env_file=None, demo_mode=True)
    dataset = IntentEvaluationDataset.load(V3_DATASET)
    runner = IntentEvaluationRunner(
        ConversationUnderstandingService(settings, OpenAICompatibleLLMGateway(settings))
    )

    result = await runner.run(dataset)

    assert result.evaluated_cases == 108
    _assert_only_intended_legacy_semantic_migrations(result.failures)
    assert all(
        result.metrics[name] == 1.0
        for name in (
            "primary_intent_macro_f1",
            "primary_intent_micro_f1",
            "high_risk_intent_f2",
            "confusion_pair_intent_card_exact_match_rate",
            "multi_task_exact_match_rate",
            "task_execution_policy_accuracy",
            "task_dependency_accuracy",
            "slot_operation_accuracy",
            "explicit_slot_accuracy",
            "inherited_slot_accuracy",
        )
    )
    assert result.metrics["route_micro_f1"] >= 0.95
    assert result.metrics["intent_card_micro_f1"] >= 0.97
    assert result.metrics["task_set_exact_match_rate"] >= 0.94
    assert result.metrics["target_action_joint_accuracy"] >= 0.95
    assert result.metrics["multi_task_miss_rate"] == 0.0
    assert result.metrics["spurious_task_rate"] <= 0.05
    assert result.confusion_matrices["intent_card"]
    assert result.per_class_metrics["primary_intent"]["investigation"]["f1"] == 1.0
    assert result.per_class_metrics["intent_card"]["action.prohibited_write"]["recall"] == 1.0
    assert result.capacity["intent_catalog_version"] == "semikb-intent-catalog-v5"
    assert result.capacity["active_intent_card_count"] == 15
    assert result.capacity["intent_card_selection"] == "all_active"
    assert result.capacity["llm_evaluated_cases"] == 0
    assert result.capacity["all_active_cards_injected_rate"] is None
    assert result.route_migration["applied"] is True
    assert result.route_migration["migrated_expectation_count"] == 19
    assert result.route_migration["migrated_intent_card_expectation_count"] == 4
    assert result.metrics["deprecated_history_direct_emission_rate"] == 0.0
    assert result.warnings == []


def test_intent_v4_freezes_semantic_minimal_pairs() -> None:
    dataset = IntentEvaluationDataset.load(V4_DATASET)
    case_ids = [case.case_id for case in dataset.cases]
    utterances = [case.utterance for case in dataset.cases]
    tags = {tag for case in dataset.cases for tag in case.tags}

    assert dataset.dataset_version == "semikb-intent-v4"
    assert dataset.source_kind == "synthetic_review_required"
    assert dataset.catalog_version == "semikb-intent-catalog-v5"
    assert dataset.example_bank_version == "not_used"
    assert len(dataset.cases) == 132
    assert len(case_ids) == len(set(case_ids))
    assert len(utterances) == len(set(utterances))
    assert dataset.dataset_hash == (
        "4ae12eb49a5365cc6d249bc7245f5e9d6e7779a31d74cd3f4ba047896f4ee662"
    )
    assert {
        "public_general",
        "aggregate",
        "group_by_product",
        "missing_time",
        "entity_data",
        "causal",
        "concrete_scope",
        "internal_controlled",
        "explicit_web",
    }.issubset(tags)


@pytest.mark.asyncio
async def test_intent_v4_deterministic_baseline_keeps_semantic_boundaries() -> None:
    settings = Settings(_env_file=None, demo_mode=True)
    dataset = IntentEvaluationDataset.load(V4_DATASET)
    runner = IntentEvaluationRunner(
        ConversationUnderstandingService(settings, OpenAICompatibleLLMGateway(settings))
    )

    result = await runner.run(dataset)

    assert result.evaluated_cases == 132
    assert result.failures == []
    assert all(
        result.metrics[name] == 1.0
        for name in (
            "interaction_mode_accuracy",
            "primary_intent_accuracy",
            "route_accuracy",
            "intent_card_micro_f1",
            "task_set_exact_match_rate",
            "multi_task_exact_match_rate",
            "task_dependency_accuracy",
            "explicit_slot_accuracy",
            "missing_slot_accuracy",
        )
    )
    assert result.metrics["wrong_clarification_rate"] == 0.0
    assert result.metrics["dangerous_execution_miss_rate"] == 0.0
    assert result.metrics["deprecated_history_direct_emission_rate"] == 0.0
    assert result.capacity["intent_catalog_version"] == "semikb-intent-catalog-v5"
    assert result.capacity["llm_evaluated_cases"] == 0
    assert result.warnings == []
