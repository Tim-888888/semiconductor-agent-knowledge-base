"""Versioned offline evaluation for controlled conversation understanding and routing."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from semikb.agent_runtime.intent_catalog import (
    IntentCatalog,
    IntentRiskLevel,
    IntentTaskSignature,
)
from semikb.agent_runtime.routing import RoutePolicy
from semikb.agent_runtime.understanding import ConversationUnderstandingService
from semikb.contracts.models import (
    ActorScope,
    AgentRoute,
    IntentTarget,
    IntentTaskAction,
    InteractionMode,
    PrimaryIntent,
    SlotOperationKind,
    TaskExecutionDecision,
)


class ExpectedIntentTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(pattern=r"^task_[1-3]$")
    primary_intent: PrimaryIntent
    target_type: IntentTarget
    action: IntentTaskAction
    depends_on: list[str] = Field(default_factory=list, max_length=2)
    execution_policy: TaskExecutionDecision = TaskExecutionDecision.EXECUTE

    @property
    def signature(self) -> IntentTaskSignature:
        return IntentTaskSignature(
            primary_intent=self.primary_intent,
            target_type=self.target_type,
            action=self.action,
            execution_policy=self.execution_policy,
        )

    @property
    def set_key(self) -> str:
        return self.signature.key


class ExpectedSlotOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: SlotOperationKind
    slot_name: str = Field(min_length=1, max_length=64)
    value: str | None = Field(default=None, max_length=256)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.operation.value, self.slot_name, self.value or "")


class IntentEvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    utterance: str
    context: dict[str, Any] = Field(default_factory=dict)
    actor_scope: ActorScope = Field(default_factory=ActorScope)
    clarification_pending: bool = False
    expected_interaction_mode: InteractionMode
    expected_primary_intent: PrimaryIntent
    expected_route: AgentRoute
    expected_task_count: int = Field(ge=1, le=3)
    expected_refused_task_count: int = Field(default=0, ge=0, le=3)
    expected_slot_operation: SlotOperationKind | None = None
    expected_context_message_ids: list[str] = Field(default_factory=list, max_length=8)
    expected_tasks: list[ExpectedIntentTask] = Field(default_factory=list, max_length=3)
    expected_intent_card_ids: list[str] = Field(default_factory=list, max_length=3)
    expected_slot_operations: list[ExpectedSlotOperation] | None = None
    expected_explicit_slots: dict[str, str] | None = None
    expected_inherited_slots: dict[str, str] | None = None
    expected_missing_slots: list[str] | None = None
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_expected_tasks(self) -> IntentEvaluationCase:
        if self.expected_tasks and len(self.expected_tasks) != self.expected_task_count:
            raise ValueError("expected_tasks length must equal expected_task_count")
        if self.expected_intent_card_ids and len(self.expected_intent_card_ids) != self.expected_task_count:
            raise ValueError("expected_intent_card_ids length must equal expected_task_count")
        expected_ids = [f"task_{index}" for index in range(1, len(self.expected_tasks) + 1)]
        if self.expected_tasks and [item.task_id for item in self.expected_tasks] != expected_ids:
            raise ValueError("expected task IDs must be contiguous and ordered")
        valid_ids: set[str] = set()
        for task in self.expected_tasks:
            if any(item not in valid_ids for item in task.depends_on):
                raise ValueError("task dependencies may only reference earlier tasks")
            valid_ids.add(task.task_id)
        return self


class IntentEvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version: str = "semikb-intent-v1"
    source_kind: str = "synthetic_review_required"
    description: str
    catalog_version: str | None = None
    example_bank_version: str | None = None
    frozen_at: str | None = None
    cases: list[IntentEvaluationCase] = Field(min_length=80, max_length=200)

    @model_validator(mode="after")
    def validate_dataset(self) -> IntentEvaluationDataset:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("intent evaluation case IDs must be unique")
        if self.dataset_version == "semikb-intent-v3":
            if not self.catalog_version or not self.example_bank_version or not self.frozen_at:
                raise ValueError("v3 must identify its catalog, example bank, and freeze time")
            incomplete = [
                case.case_id
                for case in self.cases
                if not case.expected_tasks or not case.expected_intent_card_ids
            ]
            if incomplete:
                raise ValueError(f"v3 cases are missing structured expectations: {incomplete[:5]}")
        return self

    @property
    def dataset_hash(self) -> str:
        serialized_cases = []
        optional_empty_fields = (
            "expected_context_message_ids",
            "expected_tasks",
            "expected_intent_card_ids",
            "expected_slot_operations",
            "expected_explicit_slots",
            "expected_inherited_slots",
            "expected_missing_slots",
        )
        for case in self.cases:
            item = case.model_dump(mode="json")
            for field_name in optional_empty_fields:
                if not item[field_name]:
                    item.pop(field_name)
            serialized_cases.append(item)
        payload = json.dumps(
            serialized_cases,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def load(cls, path: Path) -> IntentEvaluationDataset:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class IntentEvaluationResult(BaseModel):
    dataset_version: str
    dataset_hash: str
    evaluated_cases: int
    metrics: dict[str, float]
    failures: list[dict[str, Any]] = Field(default_factory=list)
    source_counts: dict[str, int] = Field(default_factory=dict)
    confusion_matrices: dict[str, dict[str, dict[str, int]]] = Field(default_factory=dict)
    per_class_metrics: dict[str, dict[str, dict[str, float]]] = Field(default_factory=dict)
    capacity: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class IntentEvaluationRunner:
    def __init__(
        self,
        understanding: ConversationUnderstandingService,
        policy: RoutePolicy | None = None,
    ) -> None:
        self.understanding = understanding
        self.policy = policy or RoutePolicy()
        self.catalog: IntentCatalog = understanding.intent_catalog

    async def run(
        self,
        dataset: IntentEvaluationDataset,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> IntentEvaluationResult:
        if dataset.catalog_version and dataset.catalog_version != self.catalog.catalog_version:
            raise ValueError("evaluation dataset and active intent catalog versions do not match")
        selected = dataset.cases[offset:]
        cases = selected[:limit] if limit else selected
        correct_mode = 0
        correct_intent = 0
        expected_routes: Counter[AgentRoute] = Counter()
        actual_routes: Counter[AgentRoute] = Counter()
        true_routes: Counter[AgentRoute] = Counter()
        missed_tasks = 0
        spurious_tasks = 0
        expected_tasks = 0
        actual_tasks = 0
        unnecessary_retrieval = 0
        direct_expected = 0
        wrong_reuse = 0
        reuse_predictions = 0
        wrong_clarify = 0
        non_clarify_expected = 0
        correction_failures = 0
        correction_cases = 0
        dangerous_failures = 0
        dangerous_cases = 0
        context_reference_failures = 0
        context_reference_cases = 0
        task_set_exact = 0
        structured_task_cases = 0
        multi_task_exact = 0
        multi_task_cases = 0
        target_action_matches = 0
        target_action_cases = 0
        execution_policy_matches = 0
        dependency_matches = 0
        dependency_cases = 0
        slot_operation_matches = 0
        slot_operation_cases = 0
        explicit_slot_matches = 0
        explicit_slot_cases = 0
        inherited_slot_matches = 0
        inherited_slot_cases = 0
        missing_slot_matches = 0
        missing_slot_cases = 0
        failures: list[dict[str, Any]] = []
        sources: Counter[str] = Counter()
        expected_primary_labels: list[str] = []
        actual_primary_labels: list[str] = []
        expected_route_labels: list[str] = []
        actual_route_labels: list[str] = []
        expected_card_labels: list[list[str]] = []
        actual_card_labels: list[list[str]] = []
        card_confusion_pairs: list[tuple[str, str]] = []
        prompt_tokens: list[float] = []
        completion_tokens: list[float] = []
        total_tokens: list[float] = []
        cards_in_prompt: list[float] = []
        few_shot_examples: list[float] = []
        few_shot_selection_latencies_ms: list[float] = []
        latencies_ms: list[float] = []
        llm_cases = 0
        all_active_injected = 0
        provider_calls = 0
        few_shot_embedding_calls = 0
        few_shot_embedding_input_tokens_estimate = 0
        retrieval_routes = {
            AgentRoute.REUSE_EVIDENCE,
            AgentRoute.INTERNAL_RAG,
            AgentRoute.RAG_AND_TOOL,
            AgentRoute.RAG_AND_WEB,
        }

        for case in cases:
            result = await self.understanding.understand(
                case.utterance,
                case.context,
                clarification_pending=case.clarification_pending,
            )
            interpreted = result.understanding
            plan = self.policy.decide(
                interpreted,
                case.actor_scope,
                case.context,
                case.utterance,
            )
            metadata = result.metadata
            sources[interpreted.classifier_source] += 1
            latencies_ms.append(float(metadata.get("understanding_latency_ms", 0) or 0))
            provider_calls += int(metadata.get("understanding_calls", 0) or 0)
            few_shot_embedding_calls += int(
                metadata.get("intent_few_shot_embedding_calls", 0) or 0
            )
            few_shot_embedding_input_tokens_estimate += int(
                metadata.get("intent_few_shot_embedding_input_tokens_estimate", 0) or 0
            )
            if metadata.get("understanding_source") == "llm":
                llm_cases += 1
                prompt_tokens.append(float(metadata.get("intent_prompt_tokens", 0) or 0))
                completion_tokens.append(
                    float(metadata.get("intent_completion_tokens", 0) or 0)
                )
                total_tokens.append(float(metadata.get("intent_total_tokens", 0) or 0))
                cards_in_prompt.append(
                    float(metadata.get("intent_cards_in_prompt", 0) or 0)
                )
                few_shot_examples.append(
                    float(metadata.get("intent_few_shot_example_count", 0) or 0)
                )
                few_shot_selection_latencies_ms.append(
                    float(
                        metadata.get("intent_few_shot_selection_latency_ms", 0) or 0
                    )
                )
                all_active_injected += int(
                    metadata.get("intent_card_selection") == "all_active"
                    and metadata.get("intent_cards_in_prompt")
                    == metadata.get("active_intent_card_count")
                    == len(self.catalog.active_cards)
                )

            mode_ok = interpreted.interaction_mode is case.expected_interaction_mode
            intent_ok = interpreted.primary_intent is case.expected_primary_intent
            route_ok = plan.route is case.expected_route
            context_reference_ok = (
                not case.expected_context_message_ids
                or interpreted.context_message_ids == case.expected_context_message_ids
            )
            correct_mode += int(mode_ok)
            correct_intent += int(intent_ok)
            expected_routes[case.expected_route] += 1
            actual_routes[plan.route] += 1
            true_routes[case.expected_route] += int(route_ok)
            expected_primary_labels.append(case.expected_primary_intent.value)
            actual_primary_labels.append(interpreted.primary_intent.value)
            expected_route_labels.append(case.expected_route.value)
            actual_route_labels.append(plan.route.value)
            if case.expected_context_message_ids:
                context_reference_cases += 1
                context_reference_failures += int(not context_reference_ok)

            expected_tasks += case.expected_task_count
            actual_tasks += len(interpreted.task_items)
            missed_tasks += max(case.expected_task_count - len(interpreted.task_items), 0)
            spurious_tasks += max(len(interpreted.task_items) - case.expected_task_count, 0)
            if case.expected_route in {
                AgentRoute.HISTORY_DIRECT,
                AgentRoute.CHAT_DIRECT,
                AgentRoute.TOOL_ONLY,
                AgentRoute.CLARIFY,
                AgentRoute.REFUSE,
            }:
                direct_expected += 1
                unnecessary_retrieval += int(plan.route in retrieval_routes)
            if plan.route is AgentRoute.REUSE_EVIDENCE:
                reuse_predictions += 1
                wrong_reuse += int(case.expected_route is not AgentRoute.REUSE_EVIDENCE)
            if case.expected_route is not AgentRoute.CLARIFY:
                non_clarify_expected += 1
                wrong_clarify += int(plan.route is AgentRoute.CLARIFY)
            if case.expected_slot_operation is not None:
                correction_cases += 1
                correction_failures += int(
                    not any(
                        item.operation is case.expected_slot_operation
                        for item in interpreted.slot_operations
                    )
                )
            if case.expected_refused_task_count:
                dangerous_cases += 1
                actual_refused = sum(
                    item.decision is TaskExecutionDecision.REFUSE
                    for item in plan.task_decisions
                )
                dangerous_failures += int(actual_refused < case.expected_refused_task_count)

            structured_ok = True
            expected_cards = case.expected_intent_card_ids
            actual_cards = [self._card_for_task(item) for item in interpreted.task_items]
            if expected_cards:
                expected_card_labels.append(expected_cards)
                actual_card_labels.append(actual_cards)
                card_confusion_pairs.extend(self._aligned_pairs(expected_cards, actual_cards))

            if case.expected_tasks:
                structured_task_cases += 1
                expected_counter = Counter(item.set_key for item in case.expected_tasks)
                actual_counter = Counter(self._task_set_key(item) for item in interpreted.task_items)
                task_set_ok = expected_counter == actual_counter
                task_set_exact += int(task_set_ok)
                if case.expected_task_count > 1:
                    multi_task_cases += 1
                    multi_task_exact += int(task_set_ok)
                structured_ok &= task_set_ok

                for expected, actual in zip(case.expected_tasks, interpreted.task_items, strict=False):
                    target_action_cases += 1
                    target_action_ok = (
                        expected.primary_intent is actual.primary_intent
                        and expected.target_type is actual.target_type
                        and expected.action is actual.action
                    )
                    target_action_matches += int(target_action_ok)
                    execution_policy_matches += int(
                        expected.execution_policy is actual.execution_policy
                    )
                    dependency_cases += 1
                    dependency_matches += int(expected.depends_on == actual.depends_on)
                    structured_ok &= (
                        target_action_ok
                        and expected.execution_policy is actual.execution_policy
                        and expected.depends_on == actual.depends_on
                    )

            if case.expected_slot_operations is not None:
                slot_operation_cases += 1
                expected_operations = Counter(item.key for item in case.expected_slot_operations)
                actual_operations = Counter(
                    (item.operation.value, item.slot_name, item.value or "")
                    for item in interpreted.slot_operations
                )
                operation_ok = expected_operations == actual_operations
                slot_operation_matches += int(operation_ok)
                structured_ok &= operation_ok
            if case.expected_explicit_slots is not None:
                explicit_slot_cases += 1
                slot_ok = interpreted.explicit_slots == case.expected_explicit_slots
                explicit_slot_matches += int(slot_ok)
                structured_ok &= slot_ok
            if case.expected_inherited_slots is not None:
                inherited_slot_cases += 1
                inherited_ok = interpreted.inherited_slots == case.expected_inherited_slots
                inherited_slot_matches += int(inherited_ok)
                structured_ok &= inherited_ok
            if case.expected_missing_slots is not None:
                missing_slot_cases += 1
                missing_ok = set(plan.missing_slots) == set(case.expected_missing_slots)
                missing_slot_matches += int(missing_ok)
                structured_ok &= missing_ok

            if (
                not (mode_ok and intent_ok and route_ok and context_reference_ok and structured_ok)
                or len(interpreted.task_items) < case.expected_task_count
            ):
                failures.append(
                    {
                        "case_id": case.case_id,
                        "expected": {
                            "interaction_mode": case.expected_interaction_mode,
                            "primary_intent": case.expected_primary_intent,
                            "route": case.expected_route,
                            "task_count": case.expected_task_count,
                            "intent_card_ids": expected_cards,
                            "tasks": [item.model_dump(mode="json") for item in case.expected_tasks],
                            "context_message_ids": case.expected_context_message_ids,
                        },
                        "actual": {
                            "interaction_mode": interpreted.interaction_mode,
                            "primary_intent": interpreted.primary_intent,
                            "route": plan.route,
                            "task_count": len(interpreted.task_items),
                            "intent_card_ids": actual_cards,
                            "tasks": [
                                item.model_dump(mode="json") for item in interpreted.task_items
                            ],
                            "context_message_ids": interpreted.context_message_ids,
                            "reason_codes": plan.reason_codes,
                        },
                    }
                )

        route_precisions = [
            true_routes[route] / actual_routes[route]
            for route in AgentRoute
            if actual_routes[route]
        ]
        route_recalls = [
            true_routes[route] / expected_routes[route]
            for route in AgentRoute
            if expected_routes[route]
        ]
        primary_per_class, primary_summary = self._single_label_metrics(
            expected_primary_labels, actual_primary_labels
        )
        route_per_class, route_summary = self._single_label_metrics(
            expected_route_labels, actual_route_labels
        )
        card_per_class, card_summary = self._multi_label_metrics(
            expected_card_labels, actual_card_labels
        )
        high_risk_cards = {
            card.card_id
            for card in self.catalog.active_cards
            if card.risk_level is IntentRiskLevel.HIGH
        }
        high_risk_f2 = self._selected_f_beta(
            expected_card_labels,
            actual_card_labels,
            high_risk_cards,
            beta=2,
        )
        total = max(len(cases), 1)
        metrics = {
            "interaction_mode_accuracy": correct_mode / total,
            "primary_intent_accuracy": correct_intent / total,
            "primary_intent_macro_precision": primary_summary["macro_precision"],
            "primary_intent_macro_recall": primary_summary["macro_recall"],
            "primary_intent_macro_f1": primary_summary["macro_f1"],
            "primary_intent_micro_f1": primary_summary["micro_f1"],
            "route_accuracy": sum(true_routes.values()) / total,
            "route_macro_precision": sum(route_precisions) / max(len(route_precisions), 1),
            "route_macro_recall": sum(route_recalls) / max(len(route_recalls), 1),
            "route_macro_f1": route_summary["macro_f1"],
            "route_micro_f1": route_summary["micro_f1"],
            "intent_card_macro_f1": card_summary["macro_f1"],
            "intent_card_micro_f1": card_summary["micro_f1"],
            "high_risk_intent_f2": high_risk_f2,
            "task_set_exact_match_rate": task_set_exact / max(structured_task_cases, 1),
            "multi_task_exact_match_rate": multi_task_exact / max(multi_task_cases, 1),
            "multi_task_miss_rate": missed_tasks / max(expected_tasks, 1),
            "spurious_task_rate": spurious_tasks / max(actual_tasks, 1),
            "target_action_joint_accuracy": target_action_matches / max(target_action_cases, 1),
            "task_execution_policy_accuracy": execution_policy_matches
            / max(target_action_cases, 1),
            "task_dependency_accuracy": dependency_matches / max(dependency_cases, 1),
            "slot_operation_accuracy": slot_operation_matches / max(slot_operation_cases, 1),
            "explicit_slot_accuracy": explicit_slot_matches / max(explicit_slot_cases, 1),
            "inherited_slot_accuracy": inherited_slot_matches / max(inherited_slot_cases, 1),
            "missing_slot_accuracy": missing_slot_matches / max(missing_slot_cases, 1),
            "unnecessary_retrieval_rate": unnecessary_retrieval / max(direct_expected, 1),
            "wrong_evidence_reuse_rate": wrong_reuse / max(reuse_predictions, 1),
            "wrong_clarification_rate": wrong_clarify / max(non_clarify_expected, 1),
            "slot_correction_failure_rate": correction_failures / max(correction_cases, 1),
            "dangerous_execution_miss_rate": dangerous_failures / max(dangerous_cases, 1),
            "context_reference_accuracy": (
                1.0
                if context_reference_cases == 0
                else 1 - context_reference_failures / context_reference_cases
            ),
        }
        prompt_p50 = self._percentile(prompt_tokens, 50)
        prompt_p95 = self._percentile(prompt_tokens, 95)
        completion_p50 = self._percentile(completion_tokens, 50)
        completion_p95 = self._percentile(completion_tokens, 95)
        total_p50 = self._percentile(total_tokens, 50)
        total_p95 = self._percentile(total_tokens, 95)
        cards_p50 = self._percentile(cards_in_prompt, 50)
        cards_p95 = self._percentile(cards_in_prompt, 95)
        examples_p50 = self._percentile(few_shot_examples, 50)
        examples_p95 = self._percentile(few_shot_examples, 95)
        selection_latency_p50 = self._percentile(few_shot_selection_latencies_ms, 50)
        selection_latency_p95 = self._percentile(few_shot_selection_latencies_ms, 95)
        latency_p50 = self._percentile(latencies_ms, 50)
        latency_p95 = self._percentile(latencies_ms, 95)
        warnings = self.catalog.capacity_warnings(
            prompt_tokens=int(prompt_p95),
            p95_latency_ms=latency_p95,
        )
        profile = self.understanding.experiment_profile
        bank = profile.example_bank
        capacity = {
            "intent_catalog_version": self.catalog.catalog_version,
            "intent_catalog_hash": self.catalog.catalog_hash,
            "intent_experiment_arm": profile.arm.value,
            "intent_card_selection": (
                "all_active" if profile.include_catalog else "none"
            ),
            "active_intent_card_count": len(self.catalog.active_cards),
            "llm_evaluated_cases": llm_cases,
            "understanding_provider_calls": provider_calls,
            "all_active_cards_injected_rate": (
                all_active_injected / llm_cases if llm_cases else None
            ),
            "intent_cards_in_prompt_p50": round(cards_p50, 3),
            "intent_cards_in_prompt_p95": round(cards_p95, 3),
            "intent_prompt_tokens_p50": round(prompt_p50, 3),
            "intent_prompt_tokens_p95": round(prompt_p95, 3),
            "intent_prompt_tokens_total": int(sum(prompt_tokens)),
            "intent_completion_tokens_p50": round(completion_p50, 3),
            "intent_completion_tokens_p95": round(completion_p95, 3),
            "intent_completion_tokens_total": int(sum(completion_tokens)),
            "intent_total_tokens_p50": round(total_p50, 3),
            "intent_total_tokens_p95": round(total_p95, 3),
            "intent_total_tokens_total": int(sum(total_tokens)),
            "understanding_latency_ms_p50": round(latency_p50, 3),
            "understanding_latency_ms_p95": round(latency_p95, 3),
            "intent_few_shot_strategy": profile.few_shot_strategy.value,
            "intent_example_bank_version": (
                bank.example_bank_version if bank is not None else None
            ),
            "intent_example_bank_hash": bank.example_bank_hash if bank is not None else None,
            "intent_few_shot_example_count_p50": round(examples_p50, 3),
            "intent_few_shot_example_count_p95": round(examples_p95, 3),
            "intent_few_shot_selection_latency_ms_p50": round(
                selection_latency_p50, 3
            ),
            "intent_few_shot_selection_latency_ms_p95": round(
                selection_latency_p95, 3
            ),
            "intent_few_shot_embedding_calls": few_shot_embedding_calls,
            "intent_few_shot_embedding_input_tokens_estimate": (
                few_shot_embedding_input_tokens_estimate
            ),
            "capacity_gates": self.catalog.capacity_gates.model_dump(mode="json"),
        }
        return IntentEvaluationResult(
            dataset_version=dataset.dataset_version,
            dataset_hash=dataset.dataset_hash,
            evaluated_cases=len(cases),
            metrics={key: round(value, 6) for key, value in metrics.items()},
            failures=failures,
            source_counts=dict(sources),
            confusion_matrices={
                "primary_intent": self._confusion_matrix(
                    zip(expected_primary_labels, actual_primary_labels, strict=False)
                ),
                "route": self._confusion_matrix(
                    zip(expected_route_labels, actual_route_labels, strict=False)
                ),
                "intent_card": self._confusion_matrix(card_confusion_pairs),
            },
            per_class_metrics={
                "primary_intent": primary_per_class,
                "route": route_per_class,
                "intent_card": card_per_class,
            },
            capacity=capacity,
            warnings=warnings,
        )

    def _card_for_task(self, task: Any) -> str:
        signature = IntentTaskSignature(
            primary_intent=task.primary_intent,
            target_type=task.target_type,
            action=task.action,
            execution_policy=task.execution_policy,
        )
        return self.catalog.card_for_signature(signature) or "__unmapped__"

    @staticmethod
    def _task_set_key(task: Any) -> str:
        return IntentTaskSignature(
            primary_intent=task.primary_intent,
            target_type=task.target_type,
            action=task.action,
            execution_policy=task.execution_policy,
        ).key

    @staticmethod
    def _aligned_pairs(expected: list[str], actual: list[str]) -> list[tuple[str, str]]:
        size = max(len(expected), len(actual))
        return [
            (
                expected[index] if index < len(expected) else "__spurious__",
                actual[index] if index < len(actual) else "__missing__",
            )
            for index in range(size)
        ]

    @staticmethod
    def _confusion_matrix(
        pairs: Iterable[tuple[str, str]],
    ) -> dict[str, dict[str, int]]:
        matrix: dict[str, dict[str, int]] = {}
        for expected, actual in pairs:
            matrix.setdefault(expected, {})
            matrix[expected][actual] = matrix[expected].get(actual, 0) + 1
        return {
            expected: dict(sorted(actuals.items()))
            for expected, actuals in sorted(matrix.items())
        }

    @classmethod
    def _single_label_metrics(
        cls,
        expected: list[str],
        actual: list[str],
    ) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        labels = sorted(set(expected) | set(actual))
        counts = {}
        total_tp = total_fp = total_fn = 0
        for label in labels:
            tp = sum(e == label and a == label for e, a in zip(expected, actual, strict=False))
            fp = sum(e != label and a == label for e, a in zip(expected, actual, strict=False))
            fn = sum(e == label and a != label for e, a in zip(expected, actual, strict=False))
            counts[label] = cls._metric_row(tp, fp, fn)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        return counts, cls._metric_summary(counts, total_tp, total_fp, total_fn)

    @classmethod
    def _multi_label_metrics(
        cls,
        expected: list[list[str]],
        actual: list[list[str]],
    ) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        labels = sorted(
            {item for values in expected for item in values}
            | {item for values in actual for item in values}
        )
        counts = {}
        total_tp = total_fp = total_fn = 0
        for label in labels:
            tp = fp = fn = 0
            for expected_items, actual_items in zip(expected, actual, strict=False):
                expected_count = Counter(expected_items)[label]
                actual_count = Counter(actual_items)[label]
                tp += min(expected_count, actual_count)
                fp += max(actual_count - expected_count, 0)
                fn += max(expected_count - actual_count, 0)
            counts[label] = cls._metric_row(tp, fp, fn)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        return counts, cls._metric_summary(counts, total_tp, total_fp, total_fn)

    @staticmethod
    def _metric_row(tp: int, fp: int, fn: int) -> dict[str, float]:
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        return {
            "support": float(tp + fn),
            "predicted": float(tp + fp),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }

    @classmethod
    def _metric_summary(
        cls,
        counts: dict[str, dict[str, float]],
        tp: int,
        fp: int,
        fn: int,
    ) -> dict[str, float]:
        rows = list(counts.values())
        micro = cls._metric_row(tp, fp, fn)
        return {
            "macro_precision": sum(item["precision"] for item in rows) / max(len(rows), 1),
            "macro_recall": sum(item["recall"] for item in rows) / max(len(rows), 1),
            "macro_f1": sum(item["f1"] for item in rows) / max(len(rows), 1),
            "micro_f1": micro["f1"],
        }

    @staticmethod
    def _selected_f_beta(
        expected: list[list[str]],
        actual: list[list[str]],
        selected_labels: set[str],
        *,
        beta: float,
    ) -> float:
        tp = fp = fn = 0
        for expected_items, actual_items in zip(expected, actual, strict=False):
            expected_counter = Counter(item for item in expected_items if item in selected_labels)
            actual_counter = Counter(item for item in actual_items if item in selected_labels)
            for label in selected_labels:
                tp += min(expected_counter[label], actual_counter[label])
                fp += max(actual_counter[label] - expected_counter[label], 0)
                fn += max(expected_counter[label] - actual_counter[label], 0)
        if tp + fp + fn == 0:
            return 0.0
        beta_squared = beta * beta
        return (1 + beta_squared) * tp / max(
            (1 + beta_squared) * tp + beta_squared * fn + fp,
            1,
        )

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        rank = (len(ordered) - 1) * percentile / 100
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            return ordered[lower]
        weight = rank - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight
