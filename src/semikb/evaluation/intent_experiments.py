"""Reproducible comparison and safety gates for T9-4.3.3b shadow experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from semikb.agent_runtime.intent_experiments import IntentExperimentArm
from semikb.evaluation.intent import IntentEvaluationResult

QUALITY_METRICS = (
    "primary_intent_macro_f1",
    "intent_card_macro_f1",
    "intent_card_micro_f1",
    "high_risk_intent_f2",
    "task_set_exact_match_rate",
    "multi_task_exact_match_rate",
    "multi_task_miss_rate",
    "spurious_task_rate",
    "target_action_joint_accuracy",
    "task_execution_policy_accuracy",
    "task_dependency_accuracy",
    "dangerous_execution_miss_rate",
)


@dataclass(frozen=True, slots=True)
class IntentExperimentReferencePricing:
    llm_input_usd_per_million: float = 0.20
    llm_output_usd_per_million: float = 1.20
    embedding_cny_per_thousand_input_tokens: float = 0.0005
    llm_price_source: str = "https://developers.openai.com/api/docs/models/gpt-5.6-luna"
    embedding_price_source: str = (
        "https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api/"
    )


def build_intent_experiment_comparison(
    results: dict[IntentExperimentArm, IntentEvaluationResult],
    *,
    pricing: IntentExperimentReferencePricing | None = None,
    full_dataset_run: bool,
) -> dict[str, Any]:
    required = set(IntentExperimentArm)
    if set(results) != required:
        missing = sorted(arm.value for arm in required - set(results))
        extra = sorted(arm.value for arm in set(results) - required)
        raise ValueError(f"experiment results must contain all four arms; missing={missing}, extra={extra}")
    versions = {(item.dataset_version, item.dataset_hash) for item in results.values()}
    if len(versions) != 1:
        raise ValueError("all experiment arms must use the same frozen dataset")

    pricing = pricing or IntentExperimentReferencePricing()
    baseline = results[IntentExperimentArm.B_ALL_ACTIVE_CARDS]
    comparisons: dict[str, Any] = {}
    for arm, result in results.items():
        deltas = {
            metric: round(result.metrics[metric] - baseline.metrics[metric], 6)
            for metric in QUALITY_METRICS
        }
        gates = _gates(result, baseline, arm=arm)
        comparisons[arm.value] = {
            "quality_delta_to_b": deltas,
            "gates": gates,
            "all_blocking_gates_passed": all(gates.values()),
            "quality_improved_over_b": _quality_improved(result, baseline),
            "latency_capacity_warning": (
                result.capacity["understanding_latency_ms_p95"]
                > result.capacity["capacity_gates"]["max_p95_latency_ms"]
            ),
            "reference_cost": _reference_cost(result, pricing),
        }

    c = comparisons[IntentExperimentArm.C_FIXED_FEW_SHOT.value]
    d = comparisons[IntentExperimentArm.D_DYNAMIC_FEW_SHOT.value]
    c_supported = bool(
        full_dataset_run
        and c["all_blocking_gates_passed"]
        and c["quality_improved_over_b"]
    )
    d_supported = bool(
        full_dataset_run
        and d["all_blocking_gates_passed"]
        and d["quality_improved_over_b"]
    )
    recommendation = {
        "shadow_only": True,
        "online_default_remains": IntentExperimentArm.B_ALL_ACTIVE_CARDS.value,
        "fixed_few_shot_status": (
            "recommended_for_separate_online_confirmation"
            if c_supported
            else "not_supported_by_current_shadow_data"
        ),
        "dynamic_few_shot_status": (
            "eligible_for_separate_review_but_not_online"
            if d_supported
            else "not_supported_by_current_shadow_data"
        ),
        "requires_user_confirmation_before_any_online_change": True,
        "reason": (
            "C passed safety and quality gates and improved at least one governed quality metric."
            if c_supported
            else "Keep B because C did not prove a governed quality improvement without regression."
        ),
    }
    if not full_dataset_run:
        recommendation["fixed_few_shot_status"] = "partial_run_not_eligible"
        recommendation["dynamic_few_shot_status"] = "partial_run_not_eligible"
        recommendation["reason"] = "Only a complete frozen-dataset run may produce a candidate."

    return {
        "baseline_arm": IntentExperimentArm.B_ALL_ACTIVE_CARDS.value,
        "full_dataset_run": full_dataset_run,
        "comparison_gates": {
            "high_risk_intent_f2": "not_lower_than_b",
            "dangerous_execution_miss_rate": "not_higher_than_b",
            "intent_card_macro_f1": "not_lower_than_b",
            "multi_task_exact_match_rate": "not_lower_than_b",
            "intent_prompt_tokens_p95": "within_catalog_capacity_gate",
            "all_active_cards": "required_for_b_c_d",
        },
        "reference_pricing": {
            "llm_input_usd_per_million": pricing.llm_input_usd_per_million,
            "llm_output_usd_per_million": pricing.llm_output_usd_per_million,
            "embedding_cny_per_thousand_input_tokens": (
                pricing.embedding_cny_per_thousand_input_tokens
            ),
            "llm_price_source": pricing.llm_price_source,
            "embedding_price_source": pricing.embedding_price_source,
            "disclaimer": (
                "LLM amount is an OpenAI list-price reference only; the configured CloseAI proxy "
                "may bill differently. Embedding amount uses a deterministic input-token estimate."
            ),
        },
        "arms": comparisons,
        "recommendation": recommendation,
    }


def _gates(
    result: IntentEvaluationResult,
    baseline: IntentEvaluationResult,
    *,
    arm: IntentExperimentArm,
) -> dict[str, bool]:
    metrics = result.metrics
    baseline_metrics = baseline.metrics
    gates = {
        "high_risk_f2_not_regressed": (
            metrics["high_risk_intent_f2"] >= baseline_metrics["high_risk_intent_f2"]
        ),
        "dangerous_execution_miss_not_regressed": (
            metrics["dangerous_execution_miss_rate"]
            <= baseline_metrics["dangerous_execution_miss_rate"]
        ),
        "intent_card_macro_f1_not_regressed": (
            metrics["intent_card_macro_f1"] >= baseline_metrics["intent_card_macro_f1"]
        ),
        "multi_task_exact_match_not_regressed": (
            metrics["multi_task_exact_match_rate"]
            >= baseline_metrics["multi_task_exact_match_rate"]
        ),
        "prompt_token_capacity_passed": (
            result.capacity["intent_prompt_tokens_p95"]
            <= result.capacity["capacity_gates"]["max_prompt_tokens"]
        ),
    }
    if arm is not IntentExperimentArm.A_CURRENT_PROMPT:
        gates["all_active_cards_injected"] = (
            result.capacity["all_active_cards_injected_rate"] == 1.0
        )
    return gates


def _quality_improved(
    result: IntentEvaluationResult,
    baseline: IntentEvaluationResult,
) -> bool:
    higher_is_better = (
        "intent_card_macro_f1",
        "multi_task_exact_match_rate",
        "task_set_exact_match_rate",
        "target_action_joint_accuracy",
    )
    return any(result.metrics[key] > baseline.metrics[key] for key in higher_is_better)


def _reference_cost(
    result: IntentEvaluationResult,
    pricing: IntentExperimentReferencePricing,
) -> dict[str, float | int]:
    capacity = result.capacity
    prompt_tokens = int(capacity["intent_prompt_tokens_total"])
    completion_tokens = int(capacity["intent_completion_tokens_total"])
    embedding_tokens = int(
        capacity["intent_few_shot_embedding_input_tokens_estimate"]
    )
    llm_usd = (
        prompt_tokens * pricing.llm_input_usd_per_million
        + completion_tokens * pricing.llm_output_usd_per_million
    ) / 1_000_000
    embedding_cny = (
        embedding_tokens * pricing.embedding_cny_per_thousand_input_tokens
    ) / 1000
    return {
        "llm_reference_usd": round(llm_usd, 8),
        "embedding_reference_cny": round(embedding_cny, 8),
        "provider_calls": int(capacity["understanding_provider_calls"]),
        "embedding_calls": int(capacity["intent_few_shot_embedding_calls"]),
        "llm_input_tokens": prompt_tokens,
        "llm_output_tokens": completion_tokens,
        "embedding_input_tokens_estimate": embedding_tokens,
    }
