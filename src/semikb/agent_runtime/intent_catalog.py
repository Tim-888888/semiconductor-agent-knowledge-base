"""Versioned intent governance assets used by understanding and evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from semikb.contracts.models import (
    AgentRoute,
    IntentTarget,
    IntentTaskAction,
    InteractionMode,
    PrimaryIntent,
    TaskExecutionDecision,
    TaskShape,
)

VALID_SLOT_NAMES = {
    "product",
    "process_layer",
    "tool_id",
    "chamber",
    "recipe_id",
    "recipe_version",
    "time_range",
    "lot_id",
    "case_id",
}


class IntentCardStatus(StrEnum):
    ACTIVE = "active"
    DRAFT = "draft"
    DEPRECATED = "deprecated"


class IntentRiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class IntentTaskSignature(BaseModel):
    """One legal structured task shape represented by an intent card."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_intent: PrimaryIntent
    target_type: IntentTarget
    action: IntentTaskAction
    execution_policy: TaskExecutionDecision = TaskExecutionDecision.EXECUTE

    @property
    def key(self) -> str:
        return ":".join(
            (
                self.primary_intent.value,
                self.target_type.value,
                self.action.value,
                self.execution_policy.value,
            )
        )


class IntentCapacityGates(BaseModel):
    """Frozen warning-only capacity boundaries for the all-card strategy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_active_cards: int = Field(ge=1, le=500)
    max_prompt_tokens: int = Field(ge=100, le=200_000)
    max_p95_latency_ms: float = Field(gt=0, le=600_000)


class IntentDefinitionCard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    card_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    display_name: str = Field(min_length=2, max_length=80)
    status: IntentCardStatus = IntentCardStatus.ACTIVE
    primary_intent: PrimaryIntent
    handles: list[str] = Field(min_length=1, max_length=12)
    excludes: list[str] = Field(min_length=1, max_length=12)
    confused_with: list[str] = Field(default_factory=list, max_length=12)
    positive_examples: list[str] = Field(min_length=1, max_length=8)
    hard_negative_examples: list[str] = Field(min_length=1, max_length=8)
    required_slots: list[str] = Field(default_factory=list, max_length=9)
    conditional_required_slots: dict[TaskShape, list[str]] = Field(default_factory=dict)
    allowed_routes: list[AgentRoute] = Field(min_length=1, max_length=9)
    risk_level: IntentRiskLevel
    task_signatures: list[IntentTaskSignature] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_card(self) -> IntentDefinitionCard:
        invalid_slots = sorted(set(self.required_slots) - VALID_SLOT_NAMES)
        if invalid_slots:
            raise ValueError(f"unknown required slots: {', '.join(invalid_slots)}")
        invalid_conditional = sorted(
            {
                slot
                for slots in self.conditional_required_slots.values()
                for slot in slots
            }
            - VALID_SLOT_NAMES
        )
        if invalid_conditional:
            raise ValueError(
                f"unknown conditional required slots: {', '.join(invalid_conditional)}"
            )
        if self.card_id in self.confused_with:
            raise ValueError("an intent card cannot be confused with itself")
        if len(self.confused_with) != len(set(self.confused_with)):
            raise ValueError("confused_with must not contain duplicates")
        if len(self.allowed_routes) != len(set(self.allowed_routes)):
            raise ValueError("allowed_routes must not contain duplicates")
        if any(item.primary_intent is not self.primary_intent for item in self.task_signatures):
            raise ValueError("all task signatures must use the card primary_intent")
        if len({item.key for item in self.task_signatures}) != len(self.task_signatures):
            raise ValueError("task_signatures must not contain duplicates")
        return self

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "primary_intent": self.primary_intent.value,
            "handles": self.handles,
            "excludes": self.excludes,
            "confused_with": self.confused_with,
            "positive_examples": self.positive_examples,
            "hard_negative_examples": self.hard_negative_examples,
            "required_slots": self.required_slots,
            "conditional_required_slots": {
                shape.value: slots
                for shape, slots in self.conditional_required_slots.items()
            },
            "allowed_routes": [item.value for item in self.allowed_routes],
            "risk_level": self.risk_level.value,
            "task_signatures": [
                item.model_dump(mode="json") for item in self.task_signatures
            ],
        }


class IntentCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    catalog_version: str = Field(pattern=r"^semikb-intent-catalog-v\d+$")
    source_kind: str = "reviewed_semiconductor_governance"
    description: str = Field(min_length=10, max_length=1000)
    capacity_gates: IntentCapacityGates
    cards: list[IntentDefinitionCard] = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_catalog(self) -> IntentCatalog:
        by_id = {card.card_id: card for card in self.cards}
        if len(by_id) != len(self.cards):
            raise ValueError("intent card_id values must be unique")
        if not any(card.status is IntentCardStatus.ACTIVE for card in self.cards):
            raise ValueError("the catalog must contain at least one active card")

        signature_owners: dict[str, str] = {}
        for card in self.cards:
            missing = sorted(set(card.confused_with) - set(by_id))
            if missing:
                raise ValueError(
                    f"card {card.card_id} references unknown confused_with cards: {missing}"
                )
            for other_id in card.confused_with:
                if card.card_id not in by_id[other_id].confused_with:
                    raise ValueError(
                        f"confused_with must be reciprocal: {card.card_id} <-> {other_id}"
                    )
            if card.status is not IntentCardStatus.ACTIVE:
                continue
            for signature in card.task_signatures:
                owner = signature_owners.get(signature.key)
                if owner is not None:
                    raise ValueError(
                        f"active task signature {signature.key} belongs to both {owner} and {card.card_id}"
                    )
                signature_owners[signature.key] = card.card_id
        return self

    @property
    def active_cards(self) -> tuple[IntentDefinitionCard, ...]:
        return tuple(card for card in self.cards if card.status is IntentCardStatus.ACTIVE)

    @property
    def catalog_hash(self) -> str:
        payload = self.model_dump(mode="json")
        for card in payload["cards"]:
            if not card.get("conditional_required_slots"):
                card.pop("conditional_required_slots", None)
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "catalog_version": self.catalog_version,
            "catalog_hash": self.catalog_hash,
            "selection_strategy": "all_active",
            "active_card_count": len(self.active_cards),
            "cards": [card.prompt_payload() for card in self.active_cards],
        }

    def card_for_signature(self, signature: IntentTaskSignature) -> str | None:
        for card in self.active_cards:
            if signature.key in {item.key for item in card.task_signatures}:
                return card.card_id
        return None

    def capacity_warnings(
        self,
        *,
        prompt_tokens: int | None = None,
        p95_latency_ms: float | None = None,
    ) -> list[str]:
        warnings = []
        if len(self.active_cards) > self.capacity_gates.max_active_cards:
            warnings.append("active_card_count_exceeds_gate")
        if prompt_tokens is not None and prompt_tokens > self.capacity_gates.max_prompt_tokens:
            warnings.append("intent_prompt_tokens_exceed_gate")
        if (
            p95_latency_ms is not None
            and p95_latency_ms > self.capacity_gates.max_p95_latency_ms
        ):
            warnings.append("understanding_p95_latency_exceeds_gate")
        return warnings

    @classmethod
    def load(cls, path: Path) -> IntentCatalog:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "extends" not in payload:
            return cls.model_validate(payload)
        base_path = resolve_governance_path(str(payload["extends"]))
        base = json.loads(base_path.read_text(encoding="utf-8"))
        cards = {item["card_id"]: item for item in base["cards"]}
        for override in payload.get("card_overrides", []):
            card_id = str(override.get("card_id", ""))
            if card_id not in cards:
                raise ValueError(f"catalog overlay references unknown card: {card_id}")
            cards[card_id] = {**cards[card_id], **override}
        merged = {
            **base,
            "catalog_version": payload["catalog_version"],
            "source_kind": payload.get("source_kind", base.get("source_kind")),
            "description": payload.get("description", base.get("description")),
            "capacity_gates": payload.get("capacity_gates", base.get("capacity_gates")),
            "cards": list(cards.values()),
        }
        return cls.model_validate(merged)


class IntentExample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    example_id: str = Field(pattern=r"^example-[a-z0-9-]{3,80}$")
    card_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    utterance: str = Field(min_length=2, max_length=2000)
    interaction_mode: InteractionMode
    primary_intent: PrimaryIntent
    task_items: list[IntentTaskSignature] = Field(min_length=1, max_length=3)
    expected_route: AgentRoute
    tags: list[str] = Field(default_factory=list, max_length=12)


class IntentExampleBank(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    example_bank_version: str = Field(pattern=r"^intent-example-bank-v\d+$")
    catalog_version: str = Field(pattern=r"^semikb-intent-catalog-v\d+$")
    source_kind: str = "reviewed_examples_not_evaluation"
    description: str = Field(min_length=10, max_length=1000)
    examples: list[IntentExample] = Field(min_length=1, max_length=5000)

    @model_validator(mode="after")
    def validate_examples(self) -> IntentExampleBank:
        ids = [example.example_id for example in self.examples]
        if len(ids) != len(set(ids)):
            raise ValueError("intent example_id values must be unique")
        utterances = [_normalize_utterance(example.utterance) for example in self.examples]
        if len(utterances) != len(set(utterances)):
            raise ValueError("intent example utterances must be unique after normalization")
        return self

    @property
    def example_bank_hash(self) -> str:
        serialized = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def validate_against_catalog(self, catalog: IntentCatalog) -> None:
        if self.catalog_version != catalog.catalog_version:
            raise ValueError("example bank and intent catalog versions do not match")
        active_ids = {card.card_id for card in catalog.active_cards}
        invalid_ids = sorted({item.card_id for item in self.examples} - active_ids)
        if invalid_ids:
            raise ValueError(f"examples reference inactive or unknown cards: {invalid_ids}")
        for example in self.examples:
            if example.primary_intent is not example.task_items[0].primary_intent:
                raise ValueError(f"example {example.example_id} primary intent is inconsistent")
            owner = catalog.card_for_signature(example.task_items[0])
            if owner != example.card_id:
                raise ValueError(
                    f"example {example.example_id} signature belongs to {owner}, not {example.card_id}"
                )

    def assert_no_evaluation_leakage(self, utterances: Iterable[str]) -> None:
        evaluation = {_normalize_utterance(item) for item in utterances}
        leaked = sorted(
            example.example_id
            for example in self.examples
            if _normalize_utterance(example.utterance) in evaluation
        )
        if leaked:
            raise ValueError(f"example bank overlaps the frozen evaluation set: {leaked}")

    @classmethod
    def load(cls, path: Path) -> IntentExampleBank:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def _normalize_utterance(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value, flags=re.UNICODE).lower()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_governance_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root() / path


@lru_cache(maxsize=8)
def load_intent_catalog(path: str | Path) -> IntentCatalog:
    """Load and cache a validated catalog for the process lifetime."""

    return IntentCatalog.load(resolve_governance_path(path))


@lru_cache(maxsize=8)
def load_intent_example_bank(path: str | Path) -> IntentExampleBank:
    return IntentExampleBank.load(resolve_governance_path(path))
