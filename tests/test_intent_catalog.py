from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from semikb.agent_runtime.intent_catalog import IntentCatalog, IntentExampleBank
from semikb.agent_runtime.llm_gateway import LLMCompletion
from semikb.agent_runtime.understanding import ConversationUnderstandingService
from semikb.config import Settings
from semikb.evaluation.intent import IntentEvaluationDataset

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "intent_catalogs" / "semikb_intent_catalog_v1.json"
EXAMPLE_BANK_PATH = ROOT / "data" / "intent_examples" / "intent_example_bank_v1.json"
V3_PATH = ROOT / "data" / "intent_sets" / "semikb_intent_v3.json"


class CapturingUnderstandingLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict], dict]] = []

    async def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        payload = {
            "interaction_mode": "conversation",
            "primary_intent": "conversation",
            "task_items": [
                {
                    "task_id": "task_1",
                    "primary_intent": "conversation",
                    "target_type": "general",
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
            "standalone_query": "你能协助我处理哪些工程任务？",
            "cancel_scope": None,
            "suggested_route": "chat_direct",
            "confidence": 0.96,
        }
        return LLMCompletion(
            content=json.dumps(payload, ensure_ascii=False),
            provider="test",
            requested_model="test-model",
            reported_model="test-model",
            fallback_used=False,
            attempted_providers=("test",),
            usage={"prompt_tokens": 2345, "completion_tokens": 120},
        )


def test_catalog_and_example_bank_are_frozen_and_cross_validated() -> None:
    catalog = IntentCatalog.load(CATALOG_PATH)
    bank = IntentExampleBank.load(EXAMPLE_BANK_PATH)
    dataset = IntentEvaluationDataset.load(V3_PATH)

    assert catalog.catalog_version == "semikb-intent-catalog-v1"
    assert catalog.catalog_hash == "01a5fa3ab6b12ac60a9612bde4a57de42247a72140eeb95fcf97bc38bb249f6e"
    assert len(catalog.cards) == 15
    assert len(catalog.active_cards) == 13
    assert {card.card_id for card in catalog.cards if card.status != "active"} == {
        "legacy.catch_all",
        "visual.image_similarity",
    }

    assert bank.example_bank_version == "intent-example-bank-v1"
    assert bank.example_bank_hash == "d6ead8e9e6f80a9f6390a45ec03c14b48c0b8b843d6431146505ac8be5808141"
    assert len(bank.examples) == 15
    bank.validate_against_catalog(catalog)
    bank.assert_no_evaluation_leakage(case.utterance for case in dataset.cases)


def test_catalog_rejects_unknown_confusion_reference() -> None:
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    payload["cards"][0]["confused_with"] = ["missing.intent"]

    with pytest.raises(ValidationError, match="unknown confused_with"):
        IntentCatalog.model_validate(payload)


@pytest.mark.asyncio
async def test_non_l0_understanding_injects_every_active_card_without_prefilter() -> None:
    catalog = IntentCatalog.load(CATALOG_PATH)
    llm = CapturingUnderstandingLLM()
    service = ConversationUnderstandingService(
        Settings(_env_file=None, demo_mode=False),
        llm,
        intent_catalog=catalog,
    )

    result = await service.understand("你能协助我处理哪些工程任务？", {})

    assert len(llm.calls) == 1
    messages, kwargs = llm.calls[0]
    prompt_payload = json.loads(messages[-1]["content"])
    prompt_catalog = prompt_payload["intent_catalog"]
    prompt_ids = [card["card_id"] for card in prompt_catalog["cards"]]
    assert prompt_catalog["selection_strategy"] == "all_active"
    assert prompt_ids == [card.card_id for card in catalog.active_cards]
    assert "legacy.catch_all" not in prompt_ids
    assert "visual.image_similarity" not in prompt_ids
    assert kwargs["response_schema"]
    assert kwargs["temperature"] == 0

    assert result.metadata["intent_catalog_version"] == catalog.catalog_version
    assert result.metadata["intent_catalog_hash"] == catalog.catalog_hash
    assert result.metadata["active_intent_card_count"] == 13
    assert result.metadata["intent_cards_in_prompt"] == 13
    assert result.metadata["intent_card_selection"] == "all_active"
    assert result.metadata["intent_prompt_tokens"] == 2345
    assert result.metadata["intent_prompt_tokens_source"] == "provider_usage"
    assert result.metadata["intent_catalog_capacity_warnings"] == []
    assert result.metadata["understanding_latency_ms"] >= 0
