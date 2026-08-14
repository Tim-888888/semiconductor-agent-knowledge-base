"""Shadow-only prompt strategies for governed intent understanding experiments."""

from __future__ import annotations

import asyncio
import math
import re
import time
from dataclasses import dataclass
from enum import StrEnum

from semikb.agent_runtime.intent_catalog import IntentExample, IntentExampleBank
from semikb.rag_retrieval.encoders import HybridEncoder


class IntentExperimentArm(StrEnum):
    A_CURRENT_PROMPT = "a_current_prompt"
    B_ALL_ACTIVE_CARDS = "b_all_active_cards"
    C_FIXED_FEW_SHOT = "c_fixed_few_shot"
    D_DYNAMIC_FEW_SHOT = "d_dynamic_few_shot"


class IntentFewShotStrategy(StrEnum):
    NONE = "none"
    FIXED = "fixed"
    DYNAMIC = "dynamic"


@dataclass(frozen=True, slots=True)
class IntentFewShotSelection:
    examples: tuple[IntentExample, ...] = ()
    selection_latency_ms: float = 0.0
    embedding_calls: int = 0
    embedding_input_tokens_estimate: int = 0
    embedding_model: str | None = None

    @property
    def example_ids(self) -> list[str]:
        return [example.example_id for example in self.examples]

    def prompt_payload(self) -> list[dict[str, object]]:
        return [
            {
                "example_id": example.example_id,
                "input": example.utterance,
                "classification_precedent": {
                    "interaction_mode": example.interaction_mode.value,
                    "primary_intent": example.primary_intent.value,
                    "task_items": [
                        item.model_dump(mode="json") for item in example.task_items
                    ],
                    "expected_route": example.expected_route.value,
                },
            }
            for example in self.examples
        ]


class DynamicIntentExampleSelector:
    """Select examples with an isolated in-memory index over the reviewed example bank."""

    def __init__(
        self,
        example_bank: IntentExampleBank,
        encoder: HybridEncoder,
        *,
        top_k: int = 4,
        batch_size: int = 10,
    ) -> None:
        if top_k < 1 or top_k > len(example_bank.examples):
            raise ValueError("dynamic few-shot top_k must fit inside the example bank")
        if batch_size < 1:
            raise ValueError("dynamic few-shot batch_size must be positive")
        self.example_bank = example_bank
        self.encoder = encoder
        self.top_k = top_k
        self.batch_size = batch_size
        self._bank_dense_vectors: tuple[tuple[float, ...], ...] | None = None
        self._index_lock = asyncio.Lock()

    async def select(self, request: str) -> IntentFewShotSelection:
        started = time.perf_counter()
        index_calls, index_tokens = await self._ensure_index()
        query_items = await asyncio.to_thread(self.encoder.encode, [request])
        if len(query_items) != 1:
            raise ValueError("dynamic few-shot encoder returned an unexpected query count")
        query = query_items[0].dense
        bank_vectors = self._bank_dense_vectors
        if bank_vectors is None:
            raise RuntimeError("dynamic few-shot example index was not initialized")
        ranked = sorted(
            (
                (self._dot(query, vector), example.example_id, example)
                for vector, example in zip(
                    bank_vectors,
                    self.example_bank.examples,
                    strict=True,
                )
            ),
            key=lambda item: (-item[0], item[1]),
        )
        return IntentFewShotSelection(
            examples=tuple(item[2] for item in ranked[: self.top_k]),
            selection_latency_ms=round((time.perf_counter() - started) * 1000, 3),
            embedding_calls=index_calls + 1,
            embedding_input_tokens_estimate=index_tokens + estimate_text_tokens(request),
            embedding_model=self.encoder.model_name,
        )

    async def _ensure_index(self) -> tuple[int, int]:
        if self._bank_dense_vectors is not None:
            return 0, 0
        async with self._index_lock:
            if self._bank_dense_vectors is not None:
                return 0, 0
            dense_vectors: list[tuple[float, ...]] = []
            embedding_calls = 0
            token_estimate = 0
            utterances = [example.utterance for example in self.example_bank.examples]
            for offset in range(0, len(utterances), self.batch_size):
                batch = utterances[offset : offset + self.batch_size]
                items = await asyncio.to_thread(self.encoder.encode, batch)
                if len(items) != len(batch):
                    raise ValueError(
                        "dynamic few-shot encoder returned an unexpected bank vector count"
                    )
                dense_vectors.extend(tuple(item.dense) for item in items)
                embedding_calls += 1
                token_estimate += sum(estimate_text_tokens(text) for text in batch)
            self._bank_dense_vectors = tuple(dense_vectors)
            return embedding_calls, token_estimate

    @staticmethod
    def _dot(left: list[float], right: tuple[float, ...]) -> float:
        if len(left) != len(right):
            raise ValueError("dynamic few-shot embedding dimensions do not match")
        score = sum(a * b for a, b in zip(left, right, strict=True))
        if not math.isfinite(score):
            raise ValueError("dynamic few-shot similarity is not finite")
        return score


@dataclass(frozen=True, slots=True)
class IntentExperimentProfile:
    """Explicit prompt arm; the production default is arm B."""

    arm: IntentExperimentArm
    include_catalog: bool
    few_shot_strategy: IntentFewShotStrategy = IntentFewShotStrategy.NONE
    example_bank: IntentExampleBank | None = None
    dynamic_selector: DynamicIntentExampleSelector | None = None

    def __post_init__(self) -> None:
        needs_bank = self.few_shot_strategy is not IntentFewShotStrategy.NONE
        if needs_bank != (self.example_bank is not None):
            raise ValueError("few-shot profiles must provide exactly one reviewed example bank")
        if self.few_shot_strategy is IntentFewShotStrategy.DYNAMIC:
            if self.dynamic_selector is None:
                raise ValueError("dynamic few-shot profile requires a selector")
            if self.dynamic_selector.example_bank is not self.example_bank:
                raise ValueError("dynamic selector and profile must share the same example bank")
        elif self.dynamic_selector is not None:
            raise ValueError("only the dynamic few-shot profile may provide a selector")
        if needs_bank and not self.include_catalog:
            raise ValueError("few-shot examples cannot be used without the complete active catalog")

    @classmethod
    def current_prompt(cls) -> IntentExperimentProfile:
        return cls(
            arm=IntentExperimentArm.A_CURRENT_PROMPT,
            include_catalog=False,
        )

    @classmethod
    def production_baseline(cls) -> IntentExperimentProfile:
        return cls(
            arm=IntentExperimentArm.B_ALL_ACTIVE_CARDS,
            include_catalog=True,
        )

    @classmethod
    def fixed_few_shot(cls, bank: IntentExampleBank) -> IntentExperimentProfile:
        return cls(
            arm=IntentExperimentArm.C_FIXED_FEW_SHOT,
            include_catalog=True,
            few_shot_strategy=IntentFewShotStrategy.FIXED,
            example_bank=bank,
        )

    @classmethod
    def dynamic_few_shot(
        cls,
        bank: IntentExampleBank,
        selector: DynamicIntentExampleSelector,
    ) -> IntentExperimentProfile:
        return cls(
            arm=IntentExperimentArm.D_DYNAMIC_FEW_SHOT,
            include_catalog=True,
            few_shot_strategy=IntentFewShotStrategy.DYNAMIC,
            example_bank=bank,
            dynamic_selector=selector,
        )

    async def select_examples(self, request: str) -> IntentFewShotSelection:
        if self.few_shot_strategy is IntentFewShotStrategy.NONE:
            return IntentFewShotSelection()
        if self.few_shot_strategy is IntentFewShotStrategy.FIXED:
            if self.example_bank is None:
                raise RuntimeError("fixed few-shot example bank is unavailable")
            return IntentFewShotSelection(examples=tuple(self.example_bank.examples))
        if self.dynamic_selector is None:
            raise RuntimeError("dynamic few-shot selector is unavailable")
        return await self.dynamic_selector.select(request)

    def audit_metadata(self, selection: IntentFewShotSelection) -> dict[str, object]:
        bank = self.example_bank
        return {
            "intent_experiment_arm": self.arm.value,
            "intent_few_shot_strategy": self.few_shot_strategy.value,
            "intent_example_bank_version": bank.example_bank_version if bank else None,
            "intent_example_bank_hash": bank.example_bank_hash if bank else None,
            "intent_few_shot_example_ids": selection.example_ids,
            "intent_few_shot_example_count": len(selection.examples),
            "intent_few_shot_selection_latency_ms": selection.selection_latency_ms,
            "intent_few_shot_embedding_calls": selection.embedding_calls,
            "intent_few_shot_embedding_input_tokens_estimate": (
                selection.embedding_input_tokens_estimate
            ),
            "intent_few_shot_embedding_model": selection.embedding_model,
        }


def estimate_text_tokens(text: str) -> int:
    """Stable estimate for provider calls that do not expose usage in this client."""

    return len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]", text))
