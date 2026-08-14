from __future__ import annotations

import json
from pathlib import Path

import pytest

from semikb.agent_runtime.intent_catalog import IntentCatalog, IntentExampleBank
from semikb.agent_runtime.intent_experiments import (
    DynamicIntentExampleSelector,
    IntentExperimentArm,
    IntentExperimentProfile,
)
from semikb.agent_runtime.llm_gateway import LLMCompletion
from semikb.agent_runtime.understanding import ConversationUnderstandingService
from semikb.config import Settings
from semikb.evaluation.intent import IntentEvaluationResult
from semikb.evaluation.intent_experiments import build_intent_experiment_comparison
from semikb.rag_retrieval.encoders import HybridEmbedding

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "intent_catalogs" / "semikb_intent_catalog_v1.json"
EXAMPLE_BANK_PATH = ROOT / "data" / "intent_examples" / "intent_example_bank_v1.json"


class CapturingLLM:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    async def complete(self, messages, **kwargs):
        self.calls.append(messages)
        return LLMCompletion(
            content=json.dumps(
                {
                    "interaction_mode": "task",
                    "primary_intent": "knowledge_query",
                    "task_items": [
                        {
                            "task_id": "task_1",
                            "primary_intent": "knowledge_query",
                            "target_type": "alarm",
                            "action": "explain",
                            "depends_on": [],
                            "execution_policy": "execute",
                        }
                    ],
                    "affect": {
                        "sentiment": "neutral",
                        "urgency": "normal",
                        "complaint_signal": False,
                    },
                    "slot_operations": [],
                    "explicit_slots": [],
                    "inherited_slot_names": [],
                    "missing_slots": [],
                    "context_message_ids": [],
                    "standalone_query": "解释 ESC leakage alarm 的含义",
                    "cancel_scope": None,
                    "suggested_route": "internal_rag",
                    "confidence": 0.95,
                },
                ensure_ascii=False,
            ),
            provider="test",
            requested_model="test-model",
            reported_model="test-model",
            fallback_used=False,
            attempted_providers=("test",),
            usage={"prompt_tokens": 2000, "completion_tokens": 100},
        )


class CountingEncoder:
    model_name = "counting-test-encoder"
    sparse_encoder_version = "test-sparse-v1"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [
            HybridEmbedding(dense=self._dense(text), sparse={1: 1.0}) for text in texts
        ]

    @staticmethod
    def _dense(text: str) -> list[float]:
        lowered = text.lower()
        if "alarm" in lowered or "告警" in lowered or "报警" in lowered:
            return [1.0, 0.0]
        if "recipe" in lowered or "配方" in lowered:
            return [0.0, 1.0]
        return [2**-0.5, 2**-0.5]


def load_governance() -> tuple[IntentCatalog, IntentExampleBank]:
    return IntentCatalog.load(CATALOG_PATH), IntentExampleBank.load(EXAMPLE_BANK_PATH)


def build_service(profile: IntentExperimentProfile, llm: CapturingLLM):
    catalog, _ = load_governance()
    return ConversationUnderstandingService(
        Settings(_env_file=None, demo_mode=False),
        llm,
        intent_catalog=catalog,
        experiment_profile=profile,
    )


@pytest.mark.asyncio
async def test_arm_a_reproduces_prompt_without_catalog_or_examples() -> None:
    llm = CapturingLLM()
    service = build_service(IntentExperimentProfile.current_prompt(), llm)

    result = await service.understand("解释 ESC leakage alarm 的含义", {})

    payload = json.loads(llm.calls[0][-1]["content"])
    assert set(payload) == {
        "current_request",
        "clarification_pending",
        "conversation_context",
    }
    assert "every active intent card" not in llm.calls[0][0]["content"]
    assert result.metadata["intent_experiment_arm"] == "a_current_prompt"
    assert result.metadata["intent_card_selection"] == "none"
    assert result.metadata["intent_cards_in_prompt"] == 0


@pytest.mark.asyncio
async def test_arm_b_preserves_all_card_prompt_without_examples() -> None:
    catalog, _ = load_governance()
    llm = CapturingLLM()
    service = build_service(IntentExperimentProfile.production_baseline(), llm)

    result = await service.understand("解释 ESC leakage alarm 的含义", {})

    payload = json.loads(llm.calls[0][-1]["content"])
    assert [item["card_id"] for item in payload["intent_catalog"]["cards"]] == [
        card.card_id for card in catalog.active_cards
    ]
    assert "intent_examples" not in payload
    assert result.metadata["intent_experiment_arm"] == "b_all_active_cards"
    assert result.metadata["intent_cards_in_prompt"] == 13
    assert result.metadata["intent_completion_tokens"] == 100
    assert result.metadata["intent_total_tokens"] == 2100
    assert result.metadata["intent_usage_source"] == "provider_usage"


@pytest.mark.asyncio
async def test_arm_c_injects_complete_fixed_bank_as_classification_precedents() -> None:
    catalog, bank = load_governance()
    llm = CapturingLLM()
    service = ConversationUnderstandingService(
        Settings(_env_file=None, demo_mode=False),
        llm,
        intent_catalog=catalog,
        experiment_profile=IntentExperimentProfile.fixed_few_shot(bank),
    )

    result = await service.understand("解释 ESC leakage alarm 的含义", {})

    payload = json.loads(llm.calls[0][-1]["content"])
    assert len(payload["intent_catalog"]["cards"]) == 13
    assert len(payload["intent_examples"]) == 15
    assert "classification_precedent" in payload["intent_examples"][0]
    assert "card_id" not in payload["intent_examples"][0]["classification_precedent"]
    assert result.metadata["intent_few_shot_strategy"] == "fixed"
    assert result.metadata["intent_few_shot_example_count"] == 15
    assert result.metadata["intent_few_shot_embedding_calls"] == 0


@pytest.mark.asyncio
async def test_arm_d_selects_deterministic_examples_without_filtering_cards() -> None:
    catalog, bank = load_governance()
    encoder = CountingEncoder()
    selector = DynamicIntentExampleSelector(bank, encoder, top_k=4, batch_size=10)
    llm = CapturingLLM()
    service = ConversationUnderstandingService(
        Settings(_env_file=None, demo_mode=False),
        llm,
        intent_catalog=catalog,
        experiment_profile=IntentExperimentProfile.dynamic_few_shot(bank, selector),
    )

    first = await service.understand("解释 ESC leakage alarm 的含义", {})
    second = await service.understand("解释 ESC leakage alarm 的含义", {})

    first_payload = json.loads(llm.calls[0][-1]["content"])
    second_payload = json.loads(llm.calls[1][-1]["content"])
    first_ids = [item["example_id"] for item in first_payload["intent_examples"]]
    assert len(first_payload["intent_catalog"]["cards"]) == 13
    assert len(first_ids) == 4
    assert first_ids == [item["example_id"] for item in second_payload["intent_examples"]]
    assert first.metadata["intent_few_shot_embedding_calls"] == 3
    assert second.metadata["intent_few_shot_embedding_calls"] == 1
    assert first.metadata["intent_few_shot_embedding_model"] == "counting-test-encoder"
    assert len(encoder.calls) == 4


@pytest.mark.asyncio
async def test_l0_never_invokes_dynamic_example_embedding() -> None:
    catalog, bank = load_governance()
    encoder = CountingEncoder()
    selector = DynamicIntentExampleSelector(bank, encoder, top_k=4, batch_size=10)
    llm = CapturingLLM()
    service = ConversationUnderstandingService(
        Settings(_env_file=None, demo_mode=False),
        llm,
        intent_catalog=catalog,
        experiment_profile=IntentExperimentProfile.dynamic_few_shot(bank, selector),
    )

    result = await service.understand("你好", {})

    assert llm.calls == []
    assert encoder.calls == []
    assert result.metadata["understanding_source"] == "l0"
    assert result.metadata["intent_cards_in_prompt"] == 0
    assert result.metadata["intent_few_shot_example_count"] == 0
    assert result.metadata["intent_few_shot_embedding_calls"] == 0


def experiment_result(
    arm: str,
    *,
    card_f1: float = 0.8,
    multi_task: float = 0.4,
    high_risk_f2: float = 0.9,
    dangerous_miss: float = 0.0,
) -> IntentEvaluationResult:
    metrics = {
        "primary_intent_macro_f1": 0.9,
        "intent_card_macro_f1": card_f1,
        "intent_card_micro_f1": 0.85,
        "high_risk_intent_f2": high_risk_f2,
        "task_set_exact_match_rate": 0.6,
        "multi_task_exact_match_rate": multi_task,
        "multi_task_miss_rate": 0.1,
        "spurious_task_rate": 0.05,
        "target_action_joint_accuracy": 0.7,
        "task_execution_policy_accuracy": 0.95,
        "task_dependency_accuracy": 0.95,
        "dangerous_execution_miss_rate": dangerous_miss,
    }
    return IntentEvaluationResult(
        dataset_version="semikb-intent-v3",
        dataset_hash="a" * 64,
        evaluated_cases=108,
        metrics=metrics,
        capacity={
            "intent_experiment_arm": arm,
            "intent_prompt_tokens_p95": 5000,
            "intent_prompt_tokens_total": 350_000,
            "intent_completion_tokens_total": 10_000,
            "understanding_latency_ms_p95": 6000,
            "understanding_provider_calls": 70,
            "intent_few_shot_embedding_calls": 0,
            "intent_few_shot_embedding_input_tokens_estimate": 0,
            "all_active_cards_injected_rate": (
                0.0 if arm == "a_current_prompt" else 1.0
            ),
            "capacity_gates": {
                "max_prompt_tokens": 12000,
                "max_p95_latency_ms": 5000,
            },
        },
    )


def test_comparison_can_recommend_c_for_review_but_never_switches_online() -> None:
    results = {
        IntentExperimentArm.A_CURRENT_PROMPT: experiment_result("a_current_prompt"),
        IntentExperimentArm.B_ALL_ACTIVE_CARDS: experiment_result("b_all_active_cards"),
        IntentExperimentArm.C_FIXED_FEW_SHOT: experiment_result(
            "c_fixed_few_shot", card_f1=0.82
        ),
        IntentExperimentArm.D_DYNAMIC_FEW_SHOT: experiment_result(
            "d_dynamic_few_shot", card_f1=0.83
        ),
    }

    comparison = build_intent_experiment_comparison(results, full_dataset_run=True)

    assert comparison["recommendation"]["online_default_remains"] == "b_all_active_cards"
    assert (
        comparison["recommendation"]["fixed_few_shot_status"]
        == "recommended_for_separate_online_confirmation"
    )
    assert (
        comparison["recommendation"]["dynamic_few_shot_status"]
        == "eligible_for_separate_review_but_not_online"
    )
    assert comparison["recommendation"]["shadow_only"] is True


def test_comparison_rejects_c_when_high_risk_quality_regresses() -> None:
    results = {
        IntentExperimentArm.A_CURRENT_PROMPT: experiment_result("a_current_prompt"),
        IntentExperimentArm.B_ALL_ACTIVE_CARDS: experiment_result("b_all_active_cards"),
        IntentExperimentArm.C_FIXED_FEW_SHOT: experiment_result(
            "c_fixed_few_shot", card_f1=0.82, high_risk_f2=0.8
        ),
        IntentExperimentArm.D_DYNAMIC_FEW_SHOT: experiment_result(
            "d_dynamic_few_shot"
        ),
    }

    comparison = build_intent_experiment_comparison(results, full_dataset_run=True)

    c = comparison["arms"]["c_fixed_few_shot"]
    assert c["gates"]["high_risk_f2_not_regressed"] is False
    assert (
        comparison["recommendation"]["fixed_few_shot_status"]
        == "not_supported_by_current_shadow_data"
    )
